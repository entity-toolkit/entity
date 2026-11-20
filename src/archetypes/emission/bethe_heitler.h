/**
 * @file archetypes/emission/bethe_heitler.h
 * @brief Emission policy for Bethe-Heitler pair production by protons on a
 *        prescribed isotropic photon field
 * @implements
 *   - arch::emission::bh::Counters
 *   - arch::emission::bh::Tables
 *   - arch::emission::bh::GetTables -> Tables&
 *   - arch::emission::bh::GetCounters -> Counters
 *   - arch::emission::BetheHeitler<>
 * @namespaces:
 *   - arch::emission::
 *   - arch::emission::bh::
 *
 * @note
 * The process is p + gamma -> p + e- + e+ on a static photon field that is
 * not a species of the simulation. An emitting particle of mass `mu_d` (in
 * units of the electron mass) and momentum `u` (in units of its own `m c`)
 * has the kinetic energy `K = mu_d u^2 / (gamma + 1)` in units of `m_e c^2`
 * and is treated as a proton of Lorentz factor `Gamma_p = 1 + K / mass_ratio`,
 * so that a reduced mass ratio in the dynamics does not change the energy
 * scale of the process. The cross sections are those of a proton (Z = 1): the
 * charge of the emitting species does not enter.
 *
 * @note
 * At each timestep, for each emitting particle:
 *   - the probability of one interaction is
 *       p = optical_depth * J(Gamma_p) * dt / pair_weight
 *     with J the tabulated interaction rate (units of alpha r_e^2 c per unit
 *     photon density) and `optical_depth = n_ph alpha r_e^2 L`, L being the
 *     physical length of one code unit of length (the unit of the extent of
 *     the grid); one uniform number decides (at most one interaction per
 *     particle per step, so `p` has to be small);
 *   - the outcome (photon energy and the two lepton momenta) is drawn from
 *     the Born cross sections in the rest frame of the proton and boosted to
 *     the simulation frame;
 *   - the particle loses `pair_weight * D` of kinetic energy along its
 *     momentum, D = E_- + E_+ - eps being the energy of the pair minus that
 *     of the absorbed photon;
 *   - one electron and one positron of weight `pair_weight * weight` are
 *     inserted at the position of the particle.
 *
 * @note
 * The new leptons are charged. The policy writes their previous position
 * equal to their position, so that the current deposit sees no displacement
 * for a lepton that is not pushed in the step in which it is created (lepton
 * species listed before the emitting species), and the pusher overwrites it
 * otherwise. Charge is conserved in both cases.
 */

#ifndef ARCHETYPES_EMISSION_BETHE_HEITLER_H
#define ARCHETYPES_EMISSION_BETHE_HEITLER_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/param_container.h"

#include "archetypes/emission/bethe_heitler/bh_born_qk.h"
#include "archetypes/emission/bethe_heitler/bh_event_host.h"
#include "archetypes/emission/bethe_heitler/bh_interpolate.h"
#include "archetypes/emission/bethe_heitler/bh_physics.h"
#include "archetypes/emission/bethe_heitler/bh_sampler_device.h"
#include "archetypes/emission/bethe_heitler/bh_tables_device.h"
#include "framework/containers/particles.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arch::emission {
  using namespace ntt;

  namespace bh {

    /**
     * @brief Event counters, accumulated on the device since the tables were built
     * @note With MPI these are the numbers of the calling rank
     */
    struct Counters {
      // pairs created
      unsigned long long pairs { 0 };
      // particle-steps above the upper edge of the tables (no interaction)
      unsigned long long above_tables { 0 };
      // particle-steps with a probability above `max_probability` (an error)
      unsigned long long probability_exceeded { 0 };
      // interactions skipped because the loss exceeded `max_loss_fraction`
      unsigned long long loss_bound_skipped { 0 };
      // interactions skipped because a lepton species was full
      unsigned long long capacity_dropped { 0 };
      // energy of the absorbed photons, sum of weight * eps [m_e c^2]
      double             absorbed_energy { 0.0 };
    };

    enum CounterIndex : int {
      cPairs               = 0,
      cAboveTables         = 1,
      cProbabilityExceeded = 2,
      cLossBoundSkipped    = 3,
      cCapacityDropped     = 4,
      cTotal               = 5
    };

    /**
     * @brief Tables of the process and its counters: built once per run on
     *        the first call of `GetTables`, kept on the device between
     *        steps, released when Kokkos finalizes
     */
    struct Tables {
      // interaction rate J(Gamma_p) on the device
      ::arch::bh::HadronicTables     rates;
      // rest-frame sampler of the outcome on the device
      ::arch::bh::SamplerTables      sampler;
      // nodes of the rate tables on the host
      ::arch::bh::HadronicTablesHost host;
      // largest photon energy of the spectrum [m_e c^2]
      double                         photon_energy_max { 0.0 };

      array_t<unsigned long long*> counters;
      array_t<double*>             absorbed_energy;

      // parameters the tables were built with, and those last checked
      std::string tables_key, checks_key;

      /**
       * @brief Interaction rate J(Gamma_p) [alpha r_e^2 c per unit photon
       *        density]; zero below the lower edge of the tables
       */
      auto rate(double gamma_p) const -> double {
        return ::arch::bh::interpolate_rate_host(host, gamma_p);
      }

      /**
       * @brief Energy-loss rate I(Gamma_p) = J <D> [m_e c^2 alpha r_e^2 c
       *        per unit photon density]
       */
      auto loss_rate(double gamma_p) const -> double {
        return ::arch::bh::interpolate_loss_host(host, gamma_p);
      }

      /**
       * @brief Mean loss per interaction <D>(Gamma_p) [m_e c^2]
       */
      auto mean_loss(double gamma_p) const -> double {
        double mean_loss { 0.0 }, mean_photon_energy { 0.0 };
        ::arch::bh::interpolate_outcome_host(host,
                                             gamma_p,
                                             mean_loss,
                                             mean_photon_energy);
        return mean_loss;
      }
    };

    namespace detail {
      inline auto tables_ptr() -> std::unique_ptr<Tables>& {
        static std::unique_ptr<Tables> tables;
        return tables;
      }

      template <typename T>
      auto get(const prm::Parameters& params, const std::string& name) -> T {
        return params.template get<T>("radiation.emission.bethe_heitler." + name);
      }

      inline auto get_double(const prm::Parameters& params,
                             const std::string&     name) -> double {
        const auto value = static_cast<double>(get<real_t>(params, name));
        // comparisons with a NaN are false: it would pass the range checks
        // and switch the process off without a word
        raise::ErrorIf(not std::isfinite(value),
                       "Bethe-Heitler " + name + " must be finite",
                       HERE);
        return value;
      }

      inline auto make_bath(const prm::Parameters& params)
        -> std::shared_ptr<const ::arch::bh::PhotonBath> {
        ::arch::bh::BathSpec spec;
        spec.type    = get<std::string>(params, "photon_spectrum");
        spec.eps0    = get_double(params, "photon_energy");
        spec.eps_min = get_double(params, "photon_energy_min");
        spec.eps_max = get_double(params, "photon_energy_max");
        spec.index   = get_double(params, "photon_index");
        spec.theta   = get_double(params, "photon_temperature");
        raise::ErrorIf(spec.type != "mono" and spec.type != "powerlaw" and
                         spec.type != "blackbody",
                       "Bethe-Heitler photon_spectrum must be mono, powerlaw "
                       "or blackbody",
                       HERE);
        return ::arch::bh::make_bath(spec);
      }

      inline auto tables_key(const prm::Parameters& params) -> std::string {
        return fmt::format("%s|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g|%.17g",
                           get<std::string>(params, "photon_spectrum").c_str(),
                           get_double(params, "photon_energy"),
                           get_double(params, "photon_energy_min"),
                           get_double(params, "photon_energy_max"),
                           get_double(params, "photon_index"),
                           get_double(params, "photon_temperature"),
                           get_double(params, "mass_ratio"),
                           get_double(params, "gamma_max"));
      }

      inline auto checks_key(const prm::Parameters& params, real_t dt)
        -> std::string {
        return fmt::format(
          "%.17g|%.17g|%.17g|%.17g|%.17g|%d",
          get_double(params, "optical_depth"),
          get_double(params, "pair_weight"),
          get_double(params, "max_probability"),
          get_double(params, "max_loss_fraction"),
          static_cast<double>(dt),
          static_cast<int>(params.template get<bool>("particles.use_weights")));
      }

      inline auto build(const prm::Parameters& params) -> std::unique_ptr<Tables> {
        auto T             = std::make_unique<Tables>();
        T->tables_key      = tables_key(params);
        // host threads of this build: the processors the calling thread may
        // run on (see `thread_limit`), and with a threaded host execution
        // space at most its threads
        unsigned n_threads = ::arch::bh::rf::thread_limit();
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
        n_threads = std::max(
          1u,
          std::min(n_threads,
                   static_cast<unsigned>(
                     Kokkos::DefaultHostExecutionSpace().concurrency())));
#endif
        try {
          // the table of the Born cross section is built at its first use:
          // here, unless something has read it before
          ::arch::bh::born::table(n_threads);
          const auto                          bath = make_bath(params);
          ::arch::bh::Quadrature              quadrature;
          const ::arch::bh::BetheHeitlerRates rates(bath, quadrature);
          T->host = ::arch::bh::build_tables(rates,
                                             get_double(params, "mass_ratio"),
                                             get_double(params, "gamma_max"));
          ::arch::bh::validate_tables(T->host);
          T->rates = ::arch::bh::mirror_tables_to_device(T->host);
          T->sampler = ::arch::bh::build_sampler_tables(*bath, T->host, n_threads);
          T->photon_energy_max = bath->eps_cut();
        } catch (const std::exception& e) {
          raise::Error(fmt::format("Bethe-Heitler tables: %s", e.what()), HERE);
        }
        T->counters = array_t<unsigned long long*> { "bh_counters", cTotal };
        T->absorbed_energy = array_t<double*> { "bh_absorbed_energy", 1 };
        return T;
      }

      /**
       * @brief Startup checks of the parameters against the tables, and the
       *        report of the numbers they are made with
       */
      inline void check(Tables& T, const prm::Parameters& params, real_t dt) {
        const auto tau         = get_double(params, "optical_depth");
        const auto pair_weight = get_double(params, "pair_weight");
        const auto p_max       = get_double(params, "max_probability");
        const auto f_max       = get_double(params, "max_loss_fraction");
        const auto mass_ratio  = get_double(params, "mass_ratio");

        raise::ErrorIf((pair_weight != 1.0) and
                         not params.template get<bool>("particles.use_weights"),
                       "Bethe-Heitler pair_weight != 1 requires "
                       "particles.use_weights = true",
                       HERE);

        // largest rate and smallest admissible pair weight over the tables
        double rate_max { 0.0 }, gamma_at_rate_max { 0.0 };
        double pair_weight_max { 0.0 };
        for (std::size_t i = 0; i < T.host.size(); ++i) {
          if (T.host.J[i] > rate_max) {
            rate_max          = T.host.J[i];
            gamma_at_rate_max = T.host.gamma[i];
          }
          // no outcome may cost more than max_loss_fraction of the kinetic energy
          const auto a_max = ::arch::bh::event::a_max(T.host.gamma[i],
                                                      T.photon_energy_max,
                                                      mass_ratio,
                                                      f_max);
          if (i == 0 or a_max < pair_weight_max) {
            pair_weight_max = a_max;
          }
        }
        const auto probability_max = tau * rate_max * static_cast<double>(dt) /
                                     pair_weight;
        info::Print(
          fmt::format(
            "Bethe-Heitler emission: tables on %.6g <= Gamma_p <= %.6g "
            "(%d nodes); probability per step <= %.4g (at Gamma_p = %.6g; "
            "max_probability = %.4g); pairs per emitter per unit time <= "
            "%.6g; pair_weight = %.4g (no outcome is skipped for pair_weight "
            "<= %.4g)",
            T.host.gamma_min,
            T.host.gamma_max,
            static_cast<int>(T.host.size()),
            probability_max,
            gamma_at_rate_max,
            p_max,
            tau * rate_max / pair_weight,
            pair_weight,
            pair_weight_max),
          false);
        raise::ErrorIf(
          probability_max > p_max,
          fmt::format("Bethe-Heitler emission probability per step reaches "
                      "%.4g > max_probability = %.4g: reduce the timestep or "
                      "the optical depth, or raise pair_weight",
                      probability_max,
                      p_max),
          HERE);
        T.checks_key = checks_key(params, dt);
      }
    } // namespace detail

    namespace detail {
      /**
       * @brief Warning in the log of this rank, the first time `condition`
       *        holds for the counter `index`
       */
      inline void warn_once(int index, bool condition, const std::string& msg) {
        static bool warned[cTotal] {};
        if (condition and not warned[index]) {
          warned[index] = true;
          raise::Warning(msg, HERE, false);
        }
      }
    } // namespace detail

    /**
     * @brief Tables of the run: built on the first call, rebuilt only if the
     *        photon spectrum, the mass ratio or the range have changed
     * @note Raises if the timestep is not finite, or if the probability of
     *       an interaction per step exceeds `max_probability` somewhere on
     *       the tables
     */
    inline auto GetTables(const prm::Parameters& params, real_t dt) -> Tables& {
      // a NaN would pass the bound on the probability (comparisons with a NaN
      // are false), and so would minus infinity
      raise::ErrorIf(not std::isfinite(dt),
                     "Bethe-Heitler: the timestep must be finite",
                     HERE);
      static bool hook_registered { false };
      if (not hook_registered) {
        // the device views must be released before Kokkos finalizes
        Kokkos::push_finalize_hook([]() {
          detail::tables_ptr().reset();
        });
        hook_registered = true;
      }
      auto& tables = detail::tables_ptr();
      if (tables == nullptr or tables->tables_key != detail::tables_key(params)) {
        tables = detail::build(params);
      }
      if (tables->checks_key != detail::checks_key(params, dt)) {
        detail::check(*tables, params, dt);
      }
      return *tables;
    }

    /**
     * @brief Counters of the run so far (zero if the policy was never used)
     */
    inline auto GetCounters() -> Counters {
      Counters    c;
      const auto& tables = detail::tables_ptr();
      if (tables == nullptr) {
        return c;
      }
      auto counters_h = Kokkos::create_mirror_view(tables->counters);
      Kokkos::deep_copy(counters_h, tables->counters);
      auto energy_h = Kokkos::create_mirror_view(tables->absorbed_energy);
      Kokkos::deep_copy(energy_h, tables->absorbed_energy);
      c.pairs                = counters_h(cPairs);
      c.above_tables         = counters_h(cAboveTables);
      c.probability_exceeded = counters_h(cProbabilityExceeded);
      c.loss_bound_skipped   = counters_h(cLossBoundSkipped);
      c.capacity_dropped     = counters_h(cCapacityDropped);
      c.absorbed_energy      = energy_h(0);
      return c;
    }

  } // namespace bh

  template <MetricClass M>
  struct BetheHeitler {
    struct Payload {
      // slot of the new pair past the current end of the lepton species
      npart_t ticket { 0u };
      // lepton momenta [m_e c] and absorbed photon energy [m_e c^2]
      real_t  u_electron[3] { ZERO, ZERO, ZERO };
      real_t  u_positron[3] { ZERO, ZERO, ZERO };
      real_t  photon_energy { ZERO };
    };

    const spidx_t electron_index, positron_index;

    // mass of the emitting species in units of the electron mass
    const double mu_d;
    // physical proton-to-electron mass ratio
    const double mass_ratio;
    // optical_depth * dt / pair_weight
    const double probability_norm;
    const double pair_weight;
    const double max_probability;
    const double max_loss_fraction;

    ::arch::bh::HadronicTables rates;
    ::arch::bh::SamplerTables  sampler;

    array_t<int*>      electron_i1, electron_i2, electron_i3;
    array_t<prtldx_t*> electron_dx1, electron_dx2, electron_dx3;
    array_t<int*>      electron_i1_prev, electron_i2_prev, electron_i3_prev;
    array_t<prtldx_t*> electron_dx1_prev, electron_dx2_prev, electron_dx3_prev;
    array_t<real_t*>   electron_ux1, electron_ux2, electron_ux3;
    array_t<real_t*>   electron_phi;
    array_t<real_t*>   electron_weight;
    array_t<short*>    electron_tag;
    array_t<npart_t**> electron_pld_i;

    array_t<int*>      positron_i1, positron_i2, positron_i3;
    array_t<prtldx_t*> positron_dx1, positron_dx2, positron_dx3;
    array_t<int*>      positron_i1_prev, positron_i2_prev, positron_i3_prev;
    array_t<prtldx_t*> positron_dx1_prev, positron_dx2_prev, positron_dx3_prev;
    array_t<real_t*>   positron_ux1, positron_ux2, positron_ux3;
    array_t<real_t*>   positron_phi;
    array_t<real_t*>   positron_weight;
    array_t<short*>    positron_tag;
    array_t<npart_t**> positron_pld_i;

    // tickets taken in this step (may exceed the capacity)
    array_t<npart_t> tickets { "bh_tickets" };
    const npart_t    electron_offset, positron_offset;
    const npart_t    electron_cntr, positron_cntr;
    // free slots common to the two lepton species
    const npart_t    capacity;
    const npart_t    domain_idx;
    const bool       electron_use_tracking, positron_use_tracking;

    array_t<unsigned long long*> counters;
    array_t<double*>             absorbed_energy;

    random_number_pool_t random_pool;

    /**
     * @note `sp_mass` and `dt` are taken as given: the timestep is checked in
     *       `bh::GetTables` and the mass of the emitting species by the
     *       factory of the policy (`kernel::sr::MakePusherPolicyEmission`)
     */
    BetheHeitler(Particles<M::Dim, M::CoordType>& electron_species,
                 Particles<M::Dim, M::CoordType>& positron_species,
                 spidx_t                          electron_index,
                 spidx_t                          positron_index,
                 float                            sp_mass,
                 real_t                           dt,
                 npart_t                          domain_idx,
                 const prm::Parameters&           params,
                 bh::Tables&                      tables,
                 random_number_pool_t&            random_pool)
      : electron_index { electron_index }
      , positron_index { positron_index }
      , mu_d { static_cast<double>(sp_mass) }
      , mass_ratio { bh::detail::get_double(params, "mass_ratio") }
      , probability_norm { bh::detail::get_double(params, "optical_depth") *
                           static_cast<double>(dt) /
                           bh::detail::get_double(params, "pair_weight") }
      , pair_weight { bh::detail::get_double(params, "pair_weight") }
      , max_probability { bh::detail::get_double(params, "max_probability") }
      , max_loss_fraction { bh::detail::get_double(params, "max_loss_fraction") }
      , rates { tables.rates }
      , sampler { tables.sampler }
      , electron_i1 { electron_species.i1 }
      , electron_i2 { electron_species.i2 }
      , electron_i3 { electron_species.i3 }
      , electron_dx1 { electron_species.dx1 }
      , electron_dx2 { electron_species.dx2 }
      , electron_dx3 { electron_species.dx3 }
      , electron_i1_prev { electron_species.i1_prev }
      , electron_i2_prev { electron_species.i2_prev }
      , electron_i3_prev { electron_species.i3_prev }
      , electron_dx1_prev { electron_species.dx1_prev }
      , electron_dx2_prev { electron_species.dx2_prev }
      , electron_dx3_prev { electron_species.dx3_prev }
      , electron_ux1 { electron_species.ux1 }
      , electron_ux2 { electron_species.ux2 }
      , electron_ux3 { electron_species.ux3 }
      , electron_phi { electron_species.phi }
      , electron_weight { electron_species.weight }
      , electron_tag { electron_species.tag }
      , electron_pld_i { electron_species.pld_i }
      , positron_i1 { positron_species.i1 }
      , positron_i2 { positron_species.i2 }
      , positron_i3 { positron_species.i3 }
      , positron_dx1 { positron_species.dx1 }
      , positron_dx2 { positron_species.dx2 }
      , positron_dx3 { positron_species.dx3 }
      , positron_i1_prev { positron_species.i1_prev }
      , positron_i2_prev { positron_species.i2_prev }
      , positron_i3_prev { positron_species.i3_prev }
      , positron_dx1_prev { positron_species.dx1_prev }
      , positron_dx2_prev { positron_species.dx2_prev }
      , positron_dx3_prev { positron_species.dx3_prev }
      , positron_ux1 { positron_species.ux1 }
      , positron_ux2 { positron_species.ux2 }
      , positron_ux3 { positron_species.ux3 }
      , positron_phi { positron_species.phi }
      , positron_weight { positron_species.weight }
      , positron_tag { positron_species.tag }
      , positron_pld_i { positron_species.pld_i }
      , electron_offset { electron_species.npart() }
      , positron_offset { positron_species.npart() }
      , electron_cntr { electron_species.counter() }
      , positron_cntr { positron_species.counter() }
      , capacity { std::min(electron_species.maxnpart() - electron_species.npart(),
                            positron_species.maxnpart() - positron_species.npart()) }
      , domain_idx { domain_idx }
      , electron_use_tracking { electron_species.use_tracking() }
      , positron_use_tracking { positron_species.use_tracking() }
      , counters { tables.counters }
      , absorbed_energy { tables.absorbed_energy }
      , random_pool { random_pool } {}

    auto emitted_species_indices() const -> std::vector<spidx_t> {
      return { electron_index, positron_index };
    }

    /**
     * @brief Number of pairs inserted in this step (tickets beyond the
     *        capacity were refused in `shouldEmit`)
     * @note Called on the host after the pusher kernel: it also raises the
     *       error for a probability above `max_probability`, and warns once
     *       when pairs start being skipped
     */
    auto numbers_injected() const -> std::vector<npart_t> {
      auto tickets_h = Kokkos::create_mirror_view(tickets);
      Kokkos::deep_copy(tickets_h, tickets);
      auto counters_h = Kokkos::create_mirror_view(counters);
      Kokkos::deep_copy(counters_h, counters);
      raise::ErrorIf(
        counters_h(bh::cProbabilityExceeded) > 0u,
        fmt::format("Bethe-Heitler emission probability per step exceeded "
                    "max_probability = %.4g for %llu particle-steps",
                    max_probability,
                    counters_h(bh::cProbabilityExceeded)),
        HERE);
      bh::detail::warn_once(
        bh::cCapacityDropped,
        counters_h(bh::cCapacityDropped) > 0u,
        fmt::format("Bethe-Heitler pairs are dropped: lepton species %d or %d "
                    "is full (increase maxnpart)",
                    static_cast<int>(electron_index),
                    static_cast<int>(positron_index)));
      bh::detail::warn_once(
        bh::cAboveTables,
        counters_h(bh::cAboveTables) > 0u,
        "Bethe-Heitler: particles above the upper edge of the tables do not "
        "interact (see gamma_max)");
      bh::detail::warn_once(
        bh::cLossBoundSkipped,
        counters_h(bh::cLossBoundSkipped) > 0u,
        "Bethe-Heitler: interactions above max_loss_fraction are skipped "
        "(reduce pair_weight)");
      const auto n_injected = std::min(tickets_h(), capacity);
      return { n_injected, n_injected };
    }

    /**
     * @brief Decide whether the particle interacts in this step, and if so
     *        draw the outcome and compute the recoil
     *
     * @param u_Ph Momentum of the particle in units of its own m c
     * @param delta_u_Ph Output: the recoil (same units), along `u_Ph`
     * @param payload Output: slot and momenta of the new leptons
     *
     * @returns (emit, recoil): both true for an interaction, both false otherwise
     */
    Inline auto shouldEmit(const coord_t<M::PrtlDim>&,
                           const coord_t<M::PrtlDim>&,
                           const vec_t<Dim::_3D>& u_Ph,
                           const vec_t<Dim::_3D>&,
                           const vec_t<Dim::_3D>&,
                           vec_t<Dim::_3D>& delta_u_Ph,
                           Payload& payload) const -> Kokkos::pair<bool, bool> {
      const double ux    = static_cast<double>(u_Ph[0]);
      const double uy    = static_cast<double>(u_Ph[1]);
      const double uz    = static_cast<double>(u_Ph[2]);
      const double u_sqr = ux * ux + uy * uy + uz * uz;
      if (not(u_sqr > 0.0)) {
        return Kokkos::make_pair(false, false);
      }
      const double u_mag   = math::sqrt(u_sqr);
      const double gamma   = math::sqrt(1.0 + u_sqr);
      // kinetic energy [m_e c^2] and Lorentz factor of the equivalent proton
      const double K       = mu_d * u_sqr / (gamma + 1.0);
      const double Gamma_p = 1.0 + K / mass_ratio;

      if (Gamma_p < rates.gamma_min) {
        return Kokkos::make_pair(false, false);
      }
      if (Gamma_p > rates.gamma_max) {
        Kokkos::atomic_inc(&counters(bh::cAboveTables));
        return Kokkos::make_pair(false, false);
      }
      const double probability = probability_norm * rates.J(Gamma_p);
      if (not(probability > 0.0)) {
        return Kokkos::make_pair(false, false);
      }
      if (probability > max_probability) {
        Kokkos::atomic_inc(&counters(bh::cProbabilityExceeded));
        return Kokkos::make_pair(false, false);
      }

      auto rand_gen = random_pool.get_state();
      if (not(rand_gen.drand() < probability)) {
        random_pool.free_state(rand_gen);
        return Kokkos::make_pair(false, false);
      }
      // outcome in the frame with z along the momentum of the particle
      ::arch::bh::SampledPair outcome;
      sampler.draw(Gamma_p, rand_gen, outcome);
      random_pool.free_state(rand_gen);

      if (pair_weight * outcome.D > max_loss_fraction * K) {
        Kokkos::atomic_inc(&counters(bh::cLossBoundSkipped));
        return Kokkos::make_pair(false, false);
      }

      const npart_t ticket = Kokkos::atomic_fetch_add(&tickets(),
                                                      static_cast<npart_t>(1));
      if (ticket >= capacity) {
        Kokkos::atomic_inc(&counters(bh::cCapacityDropped));
        return Kokkos::make_pair(false, false);
      }

      // recoil along the momentum
      const double nx = ux / u_mag, ny = uy / u_mag, nz = uz / u_mag;
      const double K_new = K - pair_weight * outcome.D;
      const double u_new = math::sqrt(K_new * (K_new + 2.0 * mu_d)) / mu_d;
      delta_u_Ph[0]      = static_cast<real_t>((u_new - u_mag) * nx);
      delta_u_Ph[1]      = static_cast<real_t>((u_new - u_mag) * ny);
      delta_u_Ph[2]      = static_cast<real_t>((u_new - u_mag) * nz);

      // rotate the lepton momenta to the simulation frame: (e1, e2, n)
      double e1x, e1y, e1z;
      if (math::abs(nx) < 0.9) {
        e1x = 1.0 - nx * nx;
        e1y = -nx * ny;
        e1z = -nx * nz;
      } else {
        e1x = -ny * nx;
        e1y = 1.0 - ny * ny;
        e1z = -ny * nz;
      }
      const double e1_norm  = math::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
      e1x                  /= e1_norm;
      e1y                  /= e1_norm;
      e1z                  /= e1_norm;
      const double e2x      = ny * e1z - nz * e1y;
      const double e2y      = nz * e1x - nx * e1z;
      const double e2z      = nx * e1y - ny * e1x;

      payload.ticket        = ticket;
      payload.u_electron[0] = static_cast<real_t>(
        outcome.pm[0] * e1x + outcome.pm[1] * e2x + outcome.pm[2] * nx);
      payload.u_electron[1] = static_cast<real_t>(
        outcome.pm[0] * e1y + outcome.pm[1] * e2y + outcome.pm[2] * ny);
      payload.u_electron[2] = static_cast<real_t>(
        outcome.pm[0] * e1z + outcome.pm[1] * e2z + outcome.pm[2] * nz);
      payload.u_positron[0] = static_cast<real_t>(
        outcome.pp[0] * e1x + outcome.pp[1] * e2x + outcome.pp[2] * nx);
      payload.u_positron[1] = static_cast<real_t>(
        outcome.pp[0] * e1y + outcome.pp[1] * e2y + outcome.pp[2] * ny);
      payload.u_positron[2] = static_cast<real_t>(
        outcome.pp[0] * e1z + outcome.pp[1] * e2z + outcome.pp[2] * nz);
      payload.photon_energy = static_cast<real_t>(outcome.eps);

      Kokkos::atomic_inc(&counters(bh::cPairs));
      return Kokkos::make_pair(true, true);
    }

    /**
     * @brief Insert the electron and the positron at the position of the
     *        emitting particle
     * @note `direction` (minus the direction of the recoil) is not used: the
     *       lepton momenta come from the payload
     */
    Inline void emit(const tuple_t<int, M::Dim>&      xi_Cd,
                     const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                     const vec_t<Dim::_3D>&,
                     real_t         weight,
                     real_t         phi,
                     const Payload& payload) const {
      const auto pair_weight_r = static_cast<real_t>(pair_weight) * weight;
      insert(electron_offset + payload.ticket,
             electron_i1,
             electron_i2,
             electron_i3,
             electron_dx1,
             electron_dx2,
             electron_dx3,
             electron_i1_prev,
             electron_i2_prev,
             electron_i3_prev,
             electron_dx1_prev,
             electron_dx2_prev,
             electron_dx3_prev,
             electron_ux1,
             electron_ux2,
             electron_ux3,
             electron_phi,
             electron_weight,
             electron_tag,
             electron_pld_i,
             xi_Cd,
             dxi_Cd,
             { payload.u_electron[0], payload.u_electron[1], payload.u_electron[2] },
             pair_weight_r,
             phi,
             electron_use_tracking,
             electron_cntr + payload.ticket);
      insert(positron_offset + payload.ticket,
             positron_i1,
             positron_i2,
             positron_i3,
             positron_dx1,
             positron_dx2,
             positron_dx3,
             positron_i1_prev,
             positron_i2_prev,
             positron_i3_prev,
             positron_dx1_prev,
             positron_dx2_prev,
             positron_dx3_prev,
             positron_ux1,
             positron_ux2,
             positron_ux3,
             positron_phi,
             positron_weight,
             positron_tag,
             positron_pld_i,
             xi_Cd,
             dxi_Cd,
             { payload.u_positron[0], payload.u_positron[1], payload.u_positron[2] },
             pair_weight_r,
             phi,
             positron_use_tracking,
             positron_cntr + payload.ticket);
      Kokkos::atomic_add(&absorbed_energy(0),
                         static_cast<double>(pair_weight_r) *
                           static_cast<double>(payload.photon_energy));
    }

  private:
    Inline void insert(npart_t                          p,
                       const array_t<int*>&             i1,
                       const array_t<int*>&             i2,
                       const array_t<int*>&             i3,
                       const array_t<prtldx_t*>&        dx1,
                       const array_t<prtldx_t*>&        dx2,
                       const array_t<prtldx_t*>&        dx3,
                       const array_t<int*>&             i1_prev,
                       const array_t<int*>&             i2_prev,
                       const array_t<int*>&             i3_prev,
                       const array_t<prtldx_t*>&        dx1_prev,
                       const array_t<prtldx_t*>&        dx2_prev,
                       const array_t<prtldx_t*>&        dx3_prev,
                       const array_t<real_t*>&          ux1,
                       const array_t<real_t*>&          ux2,
                       const array_t<real_t*>&          ux3,
                       const array_t<real_t*>&          phi_arr,
                       const array_t<real_t*>&          weight_arr,
                       const array_t<short*>&           tag,
                       const array_t<npart_t**>&        pld_i,
                       const tuple_t<int, M::Dim>&      xi_Cd,
                       const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                       const vec_t<Dim::_3D>&           u,
                       real_t                           weight,
                       real_t                           phi,
                       bool                             use_tracking,
                       npart_t                          species_cntr) const {
      if (not use_tracking) {
        kernel::InjectParticle<M::Dim, M::CoordType, false>(p,
                                                            i1,
                                                            i2,
                                                            i3,
                                                            dx1,
                                                            dx2,
                                                            dx3,
                                                            ux1,
                                                            ux2,
                                                            ux3,
                                                            phi_arr,
                                                            weight_arr,
                                                            tag,
                                                            pld_i,
                                                            xi_Cd,
                                                            dxi_Cd,
                                                            u,
                                                            weight,
                                                            phi);
      } else {
        kernel::InjectParticle<M::Dim, M::CoordType, true>(p,
                                                           i1,
                                                           i2,
                                                           i3,
                                                           dx1,
                                                           dx2,
                                                           dx3,
                                                           ux1,
                                                           ux2,
                                                           ux3,
                                                           phi_arr,
                                                           weight_arr,
                                                           tag,
                                                           pld_i,
                                                           xi_Cd,
                                                           dxi_Cd,
                                                           u,
                                                           weight,
                                                           phi,
                                                           domain_idx,
                                                           species_cntr);
      }
      // previous position = position: no displacement for the current
      // deposit until the particle is pushed for the first time
      if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        i1_prev(p)  = xi_Cd[0];
        dx1_prev(p) = dxi_Cd[0];
      }
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        i2_prev(p)  = xi_Cd[1];
        dx2_prev(p) = dxi_Cd[1];
      }
      if constexpr (M::Dim == Dim::_3D) {
        i3_prev(p)  = xi_Cd[2];
        dx3_prev(p) = dxi_Cd[2];
      }
    }
  };

} // namespace arch::emission

#endif // ARCHETYPES_EMISSION_BETHE_HEITLER_H
