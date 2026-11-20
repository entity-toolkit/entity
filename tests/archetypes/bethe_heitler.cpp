#include "archetypes/emission/bethe_heitler.h"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"
#include "utils/param_container.h"

#include "metrics/minkowski.h"

#include "framework/containers/particles.h"
#include "kernels/deposition/currents/global.hpp"
#include "kernels/injectors.hpp"
#include "kernels/pushers/context.h"
#include "kernels/pushers/sr.hpp"
#include "kernels/pushers/sr_policies.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;

namespace {

  using metric_t   = metric::Minkowski<Dim::_2D>;
  using species_t  = Particles<Dim::_2D, Coord::Cartesian>;
  using emission_t = arch::emission::BetheHeitler<metric_t>;

  constexpr ncells_t NX1 = 32u, NX2 = 24u;

  // monochromatic photons of 0.5 m_e c^2; emitters of mass 1 at Gamma_p = 10;
  // tables up to Gamma_p = 12
  const double MASS_RATIO = 1836.15267343;
  const double GAMMA_P    = 10.0;
  const double GAMMA_MAX  = 12.0;

  void check(bool condition, const std::string& msg) {
    if (not condition) {
      throw std::runtime_error(msg);
    }
  }

  auto make_params(real_t optical_depth,
                   real_t pair_weight     = ONE,
                   real_t max_probability = static_cast<real_t>(0.1),
                   bool   use_weights     = false) -> prm::Parameters {
    prm::Parameters   params;
    const std::string prefix = "radiation.emission.bethe_heitler.";
    params.set(prefix + "electron_species", static_cast<spidx_t>(1));
    params.set(prefix + "positron_species", static_cast<spidx_t>(2));
    params.set(prefix + "optical_depth", optical_depth);
    params.set(prefix + "mass_ratio", static_cast<real_t>(MASS_RATIO));
    params.set(prefix + "pair_weight", pair_weight);
    params.set(prefix + "max_probability", max_probability);
    params.set(prefix + "max_loss_fraction", static_cast<real_t>(0.04));
    params.set(prefix + "gamma_max", static_cast<real_t>(GAMMA_MAX));
    params.set(prefix + "photon_spectrum", std::string("mono"));
    params.set(prefix + "photon_energy", static_cast<real_t>(0.5));
    params.set(prefix + "photon_energy_min", ZERO);
    params.set(prefix + "photon_energy_max", ZERO);
    params.set(prefix + "photon_index", ZERO);
    params.set(prefix + "photon_temperature", ZERO);
    params.set("particles.use_weights", use_weights);
    return params;
  }

  auto make_species(spidx_t index, float mass, float charge, npart_t maxnpart)
    -> species_t {
    return species_t(index,
                     "s" + std::to_string(index),
                     mass,
                     charge,
                     maxnpart,
                     0u,
                     0u,
                     ParticlePusher::BORIS,
                     false,
                     RadiativeDrag::NONE,
                     EmissionType::NONE,
                     0u,
                     0u);
  }

  // momentum (in units of m c, for a species of mass 1) at the given Gamma_p
  auto momentum_at(double gamma_p) -> double {
    const double K     = MASS_RATIO * (gamma_p - 1.0);
    const double gamma = 1.0 + K;
    return std::sqrt((gamma - 1.0) * (gamma + 1.0));
  }

  // emitters at random positions (in one cell if `one_cell`, and then at
  // least `margin` away from its faces) with isotropic momenta of the same
  // magnitude
  void fill_emitters(species_t&    emitters,
                     npart_t       n,
                     double        u,
                     std::uint64_t seed,
                     bool          one_cell = false,
                     double        margin   = 0.0) {
    auto i1_h  = Kokkos::create_mirror_view(emitters.i1);
    auto i2_h  = Kokkos::create_mirror_view(emitters.i2);
    auto dx1_h = Kokkos::create_mirror_view(emitters.dx1);
    auto dx2_h = Kokkos::create_mirror_view(emitters.dx2);
    auto ux1_h = Kokkos::create_mirror_view(emitters.ux1);
    auto ux2_h = Kokkos::create_mirror_view(emitters.ux2);
    auto ux3_h = Kokkos::create_mirror_view(emitters.ux3);
    auto wei_h = Kokkos::create_mirror_view(emitters.weight);
    auto tag_h = Kokkos::create_mirror_view(emitters.tag);
    auto state = seed;
    auto next  = [&state]() {
      // splitmix64 -> uniform in [0, 1)
      state           += 0x9E3779B97F4A7C15ULL;
      std::uint64_t z  = state;
      z                = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z                = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z                = z ^ (z >> 31);
      return static_cast<double>(z >> 11) * 0x1.0p-53;
    };
    for (npart_t p = 0u; p < n; ++p) {
      i1_h(p)  = one_cell ? static_cast<int>(NX1 / 2u)
                          : static_cast<int>(next() * NX1);
      i2_h(p)  = one_cell ? static_cast<int>(NX2 / 2u)
                          : static_cast<int>(next() * NX2);
      dx1_h(p) = static_cast<prtldx_t>(margin + (1.0 - 2.0 * margin) * next());
      dx2_h(p) = static_cast<prtldx_t>(margin + (1.0 - 2.0 * margin) * next());
      const double cos_t = 2.0 * next() - 1.0;
      const double sin_t = std::sqrt(1.0 - cos_t * cos_t);
      const double phi   = 2.0 * constant::PI * next();
      ux1_h(p)           = static_cast<real_t>(u * sin_t * std::cos(phi));
      ux2_h(p)           = static_cast<real_t>(u * sin_t * std::sin(phi));
      ux3_h(p)           = static_cast<real_t>(u * cos_t);
      wei_h(p)           = ONE;
      tag_h(p)           = ParticleTag::alive;
    }
    Kokkos::deep_copy(emitters.i1, i1_h);
    Kokkos::deep_copy(emitters.i2, i2_h);
    Kokkos::deep_copy(emitters.dx1, dx1_h);
    Kokkos::deep_copy(emitters.dx2, dx2_h);
    Kokkos::deep_copy(emitters.ux1, ux1_h);
    Kokkos::deep_copy(emitters.ux2, ux2_h);
    Kokkos::deep_copy(emitters.ux3, ux3_h);
    Kokkos::deep_copy(emitters.weight, wei_h);
    Kokkos::deep_copy(emitters.tag, tag_h);
    emitters.set_npart(n);
  }

  auto pusher_context(const species_t& species, real_t dt)
    -> kernel::sr::PusherContext {
    return kernel::sr::PusherContext { species.index(),
                                       species.pusher(),
                                       species.radiative_drag_flags(),
                                       species.mass(),
                                       species.charge(),
                                       static_cast<simtime_t>(0),
                                       dt,
                                       ONE,
                                       static_cast<int>(NX1),
                                       static_cast<int>(NX2),
                                       1 };
  }

  // previous positions of all the slots of a species set to one point: the
  // state that particles removed earlier leave behind
  void set_previous(species_t& species, int i1, int i2, prtldx_t dx1, prtldx_t dx2) {
    Kokkos::deep_copy(species.i1_prev, i1);
    Kokkos::deep_copy(species.i2_prev, i2);
    Kokkos::deep_copy(species.dx1_prev, dx1);
    Kokkos::deep_copy(species.dx2_prev, dx2);
  }

  // control: the same policy with the insertion of the built-in policies
  // (kernel::InjectParticle alone), which leaves the previous position of
  // the new particle as it finds it
  struct WithoutPreviousPosition {
    using Payload = emission_t::Payload;

    emission_t base;

    auto emitted_species_indices() const -> std::vector<spidx_t> {
      return base.emitted_species_indices();
    }

    auto numbers_injected() const -> std::vector<npart_t> {
      return base.numbers_injected();
    }

    Inline auto shouldEmit(const coord_t<Dim::_2D>& x_Cd,
                           const coord_t<Dim::_2D>& x_Ph,
                           const vec_t<Dim::_3D>&   u_Ph,
                           const vec_t<Dim::_3D>&   ep_Ph,
                           const vec_t<Dim::_3D>&   bp_Ph,
                           vec_t<Dim::_3D>&         delta_u_Ph,
                           Payload& payload) const -> Kokkos::pair<bool, bool> {
      return base.shouldEmit(x_Cd, x_Ph, u_Ph, ep_Ph, bp_Ph, delta_u_Ph, payload);
    }

    Inline void emit(const tuple_t<int, Dim::_2D>&      xi_Cd,
                     const tuple_t<prtldx_t, Dim::_2D>& dxi_Cd,
                     const vec_t<Dim::_3D>&,
                     real_t         weight,
                     real_t         phi,
                     const Payload& payload) const {
      const auto w = static_cast<real_t>(base.pair_weight) * weight;
      kernel::InjectParticle<Dim::_2D, Coord::Cartesian, false>(
        base.electron_offset + payload.ticket,
        base.electron_i1,
        base.electron_i2,
        base.electron_i3,
        base.electron_dx1,
        base.electron_dx2,
        base.electron_dx3,
        base.electron_ux1,
        base.electron_ux2,
        base.electron_ux3,
        base.electron_phi,
        base.electron_weight,
        base.electron_tag,
        base.electron_pld_i,
        xi_Cd,
        dxi_Cd,
        { payload.u_electron[0], payload.u_electron[1], payload.u_electron[2] },
        w,
        phi);
      kernel::InjectParticle<Dim::_2D, Coord::Cartesian, false>(
        base.positron_offset + payload.ticket,
        base.positron_i1,
        base.positron_i2,
        base.positron_i3,
        base.positron_dx1,
        base.positron_dx2,
        base.positron_dx3,
        base.positron_ux1,
        base.positron_ux2,
        base.positron_ux3,
        base.positron_phi,
        base.positron_weight,
        base.positron_tag,
        base.positron_pld_i,
        xi_Cd,
        dxi_Cd,
        { payload.u_positron[0], payload.u_positron[1], payload.u_positron[2] },
        w,
        phi);
    }
  };

  template <class E>
  void push_emitters(species_t&                    emitters,
                     const E&                      emission,
                     const metric_t&               metric,
                     const ndfield_t<Dim::_2D, 6>& EB,
                     real_t                        dt) {
    const auto policy     = kernel::sr::PusherPolicy<metric_t, E> { emission };
    const auto boundaries = kernel::sr::PusherBoundaries<Dim::_2D> {
      { { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
       { PrtlBC::PERIODIC, PrtlBC::PERIODIC } }
    };
    Kokkos::parallel_for("ParticlePusher",
                         emitters.rangeActiveParticles(),
                         kernel::sr::Pusher_kernel<metric_t, decltype(policy)>(
                           pusher_context(emitters, dt),
                           boundaries,
                           emitters,
                           EB,
                           metric,
                           policy));
  }

  // one push of the emitters with the Bethe-Heitler policy (or with the
  // control that does not write the previous position); returns the number
  // of pairs inserted
  auto push_with_emission(species_t&                    emitters,
                          species_t&                    electrons,
                          species_t&                    positrons,
                          const metric_t&               metric,
                          const ndfield_t<Dim::_2D, 6>& EB,
                          const prm::Parameters&        params,
                          real_t                        dt,
                          random_number_pool_t&         random_pool,
                          bool write_previous = true) -> npart_t {
    auto&            tables = arch::emission::bh::GetTables(params, dt);
    const emission_t emission { electrons,
                                positrons,
                                electrons.index(),
                                positrons.index(),
                                emitters.mass(),
                                dt,
                                0u,
                                params,
                                tables,
                                random_pool };
    // the push, then what the engine does with the answers of the policy:
    // the species it names receive the numbers of particles it reports
    const auto       push_and_count = [&](const auto& policy) -> npart_t {
      push_emitters(emitters, policy, metric, EB, dt);
      const auto species    = policy.emitted_species_indices();
      const auto n_injected = policy.numbers_injected();
      check(species.size() == 2u and n_injected.size() == 2u and
              species[0] == electrons.index() and species[1] == positrons.index(),
            "the policy does not name the electron and the positron species");
      electrons.set_npart(electrons.npart() + n_injected[0]);
      electrons.set_counter(electrons.counter() + n_injected[0]);
      positrons.set_npart(positrons.npart() + n_injected[1]);
      positrons.set_counter(positrons.counter() + n_injected[1]);
      return n_injected[0];
    };
    return write_previous ? push_and_count(emission)
                          : push_and_count(WithoutPreviousPosition { emission });
  }

  void push_plain(species_t&                    species,
                  const metric_t&               metric,
                  const ndfield_t<Dim::_2D, 6>& EB,
                  real_t                        dt) {
    const auto policy     = kernel::sr::PusherPolicy<metric_t> {};
    const auto boundaries = kernel::sr::PusherBoundaries<Dim::_2D> {
      { { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
       { PrtlBC::PERIODIC, PrtlBC::PERIODIC } }
    };
    Kokkos::parallel_for("ParticlePusher",
                         species.rangeActiveParticles(),
                         kernel::sr::Pusher_kernel<metric_t, decltype(policy)>(
                           pusher_context(species, dt),
                           boundaries,
                           species,
                           EB,
                           metric,
                           policy));
  }

  // currents of one species, folded periodically onto the active cells:
  // J[comp][i + NX1 * j] accumulated in double on the host
  void deposit(const species_t&                  species,
               const metric_t&                   metric,
               real_t                            dt,
               std::vector<std::vector<double>>& J) {
    ndfield_t<Dim::_2D, 3> cur { "cur", NX1 + 2u * N_GHOSTS, NX2 + 2u * N_GHOSTS };
    auto scatter_cur = Kokkos::Experimental::create_scatter_view(cur);
    Kokkos::parallel_for(
      "CurrentsDeposit",
      species.rangeActiveParticles(),
      kernel::DepositCurrents_kernel<SimEngine::SRPIC, metric_t, SHAPE_ORDER>(
        scatter_cur,
        species,
        metric,
        static_cast<real_t>(species.charge()),
        dt));
    Kokkos::Experimental::contribute(cur, scatter_cur);
    auto cur_h = Kokkos::create_mirror_view(cur);
    Kokkos::deep_copy(cur_h, cur);
    const int n1 = static_cast<int>(NX1), n2 = static_cast<int>(NX2);
    const int ng = static_cast<int>(N_GHOSTS);
    for (int g1 = 0; g1 < n1 + 2 * ng; ++g1) {
      for (int g2 = 0; g2 < n2 + 2 * ng; ++g2) {
        const int i = ((g1 - ng) % n1 + n1) % n1;
        const int j = ((g2 - ng) % n2 + n2) % n2;
        for (int c = 0; c < 3; ++c) {
          J[c][i + n1 * j] += static_cast<double>(cur_h(g1, g2, c));
        }
      }
    }
  }

  // charge on the nodes with the first-order shape (the one of the deposit)
  void charge_density(const species_t& species, std::vector<double>& rho) {
    auto i1_h  = Kokkos::create_mirror_view(species.i1);
    auto i2_h  = Kokkos::create_mirror_view(species.i2);
    auto dx1_h = Kokkos::create_mirror_view(species.dx1);
    auto dx2_h = Kokkos::create_mirror_view(species.dx2);
    auto wei_h = Kokkos::create_mirror_view(species.weight);
    auto tag_h = Kokkos::create_mirror_view(species.tag);
    Kokkos::deep_copy(i1_h, species.i1);
    Kokkos::deep_copy(i2_h, species.i2);
    Kokkos::deep_copy(dx1_h, species.dx1);
    Kokkos::deep_copy(dx2_h, species.dx2);
    Kokkos::deep_copy(wei_h, species.weight);
    Kokkos::deep_copy(tag_h, species.tag);
    const int    n1 = static_cast<int>(NX1), n2 = static_cast<int>(NX2);
    const double q = static_cast<double>(species.charge());
    for (npart_t p = 0u; p < species.npart(); ++p) {
      if (tag_h(p) != ParticleTag::alive) {
        continue;
      }
      const double w  = q * static_cast<double>(wei_h(p));
      const double fx = static_cast<double>(dx1_h(p));
      const double fy = static_cast<double>(dx2_h(p));
      const int    i = (i1_h(p) % n1 + n1) % n1, ip = (i + 1) % n1;
      const int    j = (i2_h(p) % n2 + n2) % n2, jp = (j + 1) % n2;
      rho[i + n1 * j]   += w * (1.0 - fx) * (1.0 - fy);
      rho[ip + n1 * j]  += w * fx * (1.0 - fy);
      rho[i + n1 * jp]  += w * (1.0 - fx) * fy;
      rho[ip + n1 * jp] += w * fx * fy;
    }
  }

  // max over the nodes of |d rho / dt + div J|
  auto continuity_residual(const std::vector<double>&              rho_old,
                           const std::vector<double>&              rho_new,
                           const std::vector<std::vector<double>>& J,
                           double dt) -> double {
    const int n1 = static_cast<int>(NX1), n2 = static_cast<int>(NX2);
    double    residual = 0.0;
    for (int i = 0; i < n1; ++i) {
      for (int j = 0; j < n2; ++j) {
        const int    im = (i - 1 + n1) % n1, jm = (j - 1 + n2) % n2;
        const double r = (rho_new[i + n1 * j] - rho_old[i + n1 * j]) / dt +
                         (J[0][i + n1 * j] - J[0][im + n1 * j]) +
                         (J[1][i + n1 * j] - J[1][i + n1 * jm]);
        residual = std::max(residual, std::abs(r));
      }
    }
    return residual;
  }

  // kinetic energies [m c^2] of the particles of a species, in double
  auto kinetic_energies(const species_t& species) -> std::vector<double> {
    auto ux1_h = Kokkos::create_mirror_view(species.ux1);
    auto ux2_h = Kokkos::create_mirror_view(species.ux2);
    auto ux3_h = Kokkos::create_mirror_view(species.ux3);
    Kokkos::deep_copy(ux1_h, species.ux1);
    Kokkos::deep_copy(ux2_h, species.ux2);
    Kokkos::deep_copy(ux3_h, species.ux3);
    std::vector<double> K(species.npart());
    for (npart_t p = 0u; p < species.npart(); ++p) {
      const double u_sqr = static_cast<double>(ux1_h(p)) * ux1_h(p) +
                           static_cast<double>(ux2_h(p)) * ux2_h(p) +
                           static_cast<double>(ux3_h(p)) * ux3_h(p);
      K[p] = u_sqr / (std::sqrt(1.0 + u_sqr) + 1.0);
    }
    return K;
  }

  // the part of a domain that the factory of the policy uses
  struct MockDomain {
    std::deque<species_t> species;
    random_number_pool_t  pool { 1 };

    void add_species(float mass, float charge) {
      const auto index = static_cast<spidx_t>(species.size() + 1u);
      species.emplace_back(index,
                           "s" + std::to_string(index),
                           mass,
                           charge,
                           16u,
                           0u,
                           0u,
                           ParticlePusher::BORIS,
                           false,
                           RadiativeDrag::NONE,
                           EmissionType::NONE,
                           0u,
                           0u);
    }

    auto index() const -> unsigned int {
      return 0u;
    }

    auto random_pool() -> random_number_pool_t& {
      return pool;
    }
  };

  /*
   * Charge conservation of the newly created pairs. One step of 2000
   * emitters in a periodic box:
   *   - leptons not yet pushed (species listed before the emitters): their
   *     in-plane current is exactly zero, and the continuity equation holds
   *     for all the species together;
   *   - leptons pushed in the same step (species listed after the
   *     emitters): the continuity equation holds.
   */
  void test_charge() {
    const metric_t metric {
      { NX1, NX2 },
      { { 0.0, static_cast<real_t>(NX1) }, { 0.0, static_cast<real_t>(NX2) } },
      {}
    };
    const auto dt = static_cast<real_t>(0.45);
    const ndfield_t<Dim::_2D, 6> EB { "EB", NX1 + 2u * N_GHOSTS, NX2 + 2u * N_GHOSTS };
    random_number_pool_t random_pool { 12345 };

    const npart_t n_emitters = 2000u;
    auto          electrons  = make_species(1, 1.0f, -1.0f, n_emitters);
    auto          positrons  = make_species(2, 1.0f, 1.0f, n_emitters);
    auto          emitters   = make_species(3, 1.0f, 1.0f, n_emitters);
    fill_emitters(emitters, n_emitters, momentum_at(GAMMA_P), 1u);

    // optical depth giving a probability of 0.07 per step at GAMMA_P
    auto       params = make_params(static_cast<real_t>(1e-6));
    const auto rate   = arch::emission::bh::GetTables(params, dt).rate(GAMMA_P);
    params            = make_params(static_cast<real_t>(0.07 / (rate * dt)));

    const std::size_t   n_nodes = NX1 * NX2;
    std::vector<double> rho_old(n_nodes, 0.0);
    charge_density(emitters, rho_old);

    const auto n_pairs = push_with_emission(emitters,
                                            electrons,
                                            positrons,
                                            metric,
                                            EB,
                                            params,
                                            dt,
                                            random_pool);
    check(n_pairs > 100u, "charge: too few pairs created");
    check(electrons.npart() == n_pairs and positrons.npart() == n_pairs,
          "charge: wrong number of leptons");

    // round-off of the deposit: the offsets within the cell are stored (and
    // partly summed) in `prtldx_t` whatever the precision of the build
    const double eps = std::max(
      static_cast<double>(std::numeric_limits<real_t>::epsilon()),
      static_cast<double>(std::numeric_limits<prtldx_t>::epsilon()));

    double residual_unpushed { 0.0 }, residual_pushed { 0.0 };

    // leptons not yet pushed
    {
      std::vector<std::vector<double>> J_leptons(3,
                                                 std::vector<double>(n_nodes, 0.0));
      deposit(electrons, metric, dt, J_leptons);
      deposit(positrons, metric, dt, J_leptons);
      double j_max = 0.0;
      for (std::size_t n = 0; n < n_nodes; ++n) {
        j_max = std::max(j_max, std::abs(J_leptons[0][n]));
        j_max = std::max(j_max, std::abs(J_leptons[1][n]));
      }
      check(
        j_max == 0.0,
        fmt::format("charge: in-plane current of unpushed leptons %.3e != 0",
                    j_max));

      auto J_all = J_leptons;
      deposit(emitters, metric, dt, J_all);
      std::vector<double> rho_new(n_nodes, 0.0);
      charge_density(emitters, rho_new);
      charge_density(electrons, rho_new);
      charge_density(positrons, rho_new);
      residual_unpushed = continuity_residual(rho_old, rho_new, J_all, dt);
      check(residual_unpushed < 100.0 * eps,
            fmt::format("charge: continuity residual %.3e (leptons not pushed)",
                        residual_unpushed));
    }

    // leptons pushed in the step in which they are created
    {
      push_plain(electrons, metric, EB, dt);
      push_plain(positrons, metric, EB, dt);
      std::vector<std::vector<double>> J_all(3, std::vector<double>(n_nodes, 0.0));
      deposit(emitters, metric, dt, J_all);
      deposit(electrons, metric, dt, J_all);
      deposit(positrons, metric, dt, J_all);
      std::vector<double> rho_new(n_nodes, 0.0);
      charge_density(emitters, rho_new);
      charge_density(electrons, rho_new);
      charge_density(positrons, rho_new);
      residual_pushed = continuity_residual(rho_old, rho_new, J_all, dt);
      check(residual_pushed < 100.0 * eps,
            fmt::format("charge: continuity residual %.3e (leptons pushed)",
                        residual_pushed));
    }
    std::cout << fmt::format("charge: %u pairs, continuity residuals %.3e "
                             "(leptons not pushed) and %.3e (pushed)",
                             n_pairs,
                             residual_unpushed,
                             residual_pushed)
              << std::endl;
  }

  // largest in-plane current that one species deposits
  auto max_inplane_current(const species_t&                  species,
                           const metric_t&                   metric,
                           real_t                            dt,
                           std::vector<std::vector<double>>& J_all) -> double {
    std::vector<std::vector<double>> J(3, std::vector<double>(NX1 * NX2, 0.0));
    deposit(species, metric, dt, J);
    double j_max = 0.0;
    for (std::size_t n = 0; n < NX1 * NX2; ++n) {
      j_max = std::max(j_max, std::abs(J[0][n]));
      j_max = std::max(j_max, std::abs(J[1][n]));
      for (int c = 0; c < 3; ++c) {
        J_all[c][n] += J[c][n];
      }
    }
    return j_max;
  }

  /*
   * Charge conservation when the slots of the new leptons have been used
   * before. 2000 emitters in one cell; the previous positions of all the
   * lepton slots are set to one point for the electrons and to another for
   * the positrons, in two arms:
   *   - same cell: both points are in the cell of the emitters, so the
   *     offsets within the cell differ from those of the new leptons and the
   *     cell indices do not;
   *   - neighbouring cells: the two points are in the two cells diagonally
   *     opposite across the cell of the emitters, so the cell indices differ
   *     too, in both directions. The emitters stay in the middle half of
   *     their cell, which keeps every displacement that the deposit can be
   *     given, with the previous position written in full, in part or not at
   *     all, within one cell.
   * In each arm, with the leptons not yet pushed:
   *   - each lepton species alone deposits no in-plane current, and the
   *     continuity equation holds for all the species together;
   *   - control: on the same state, the insertion of the built-in policies,
   *     which does not write the previous position, must violate both (the
   *     continuity equation by more than 1000 times the limit).
   * A policy that wrote only the cell indices or only the offsets, only one
   * direction or only one species fails one of the two arms. The test is
   * two-dimensional: the previous position in the third direction is not
   * exercised, and a policy that failed to write that one alone would pass.
   */
  void test_charge_reused_slots() {
    const metric_t metric {
      { NX1, NX2 },
      { { 0.0, static_cast<real_t>(NX1) }, { 0.0, static_cast<real_t>(NX2) } },
      {}
    };
    const auto dt = static_cast<real_t>(0.45);
    const ndfield_t<Dim::_2D, 6> EB { "EB", NX1 + 2u * N_GHOSTS, NX2 + 2u * N_GHOSTS };

    auto       params = make_params(static_cast<real_t>(1e-6));
    const auto rate   = arch::emission::bh::GetTables(params, dt).rate(GAMMA_P);
    params            = make_params(static_cast<real_t>(0.07 / (rate * dt)));

    const double eps = std::max(
      static_cast<double>(std::numeric_limits<real_t>::epsilon()),
      static_cast<double>(std::numeric_limits<prtldx_t>::epsilon()));
    const std::size_t n_nodes    = NX1 * NX2;
    const npart_t     n_emitters = 2000u;
    const int         cell1      = static_cast<int>(NX1 / 2u),
              cell2              = static_cast<int>(NX2 / 2u);

    struct Arm {
      std::string   name;
      // previous position of the electron slots: cell and offsets
      int           electron_i1, electron_i2;
      double        electron_dx1, electron_dx2;
      // previous position of the positron slots
      int           positron_i1, positron_i2;
      double        positron_dx1, positron_dx2;
      // distance of the emitters from the faces of their cell, and their seed
      double        margin;
      std::uint64_t seed;
    };

    const std::vector<Arm> arms {
      {          "same cell",cell1,cell2,0.2, 0.7,cell1,cell2,0.8, 0.3,0.0, 5u       },
      { "neighbouring cells",
       cell1 - 1,
       cell2 + 1,
       0.8, 0.2,
       cell1 + 1,
       cell2 - 1,
       0.2, 0.8,
       0.25, 6u }
    };

    for (const auto& arm : arms) {
      for (const bool write_previous : { true, false }) {
        random_number_pool_t random_pool { 2468 };
        auto electrons = make_species(1, 1.0f, -1.0f, n_emitters);
        auto positrons = make_species(2, 1.0f, 1.0f, n_emitters);
        auto emitters  = make_species(3, 1.0f, 1.0f, n_emitters);
        fill_emitters(emitters,
                      n_emitters,
                      momentum_at(GAMMA_P),
                      arm.seed,
                      true,
                      arm.margin);
        set_previous(electrons,
                     arm.electron_i1,
                     arm.electron_i2,
                     static_cast<prtldx_t>(arm.electron_dx1),
                     static_cast<prtldx_t>(arm.electron_dx2));
        set_previous(positrons,
                     arm.positron_i1,
                     arm.positron_i2,
                     static_cast<prtldx_t>(arm.positron_dx1),
                     static_cast<prtldx_t>(arm.positron_dx2));

        std::vector<double> rho_old(n_nodes, 0.0);
        charge_density(emitters, rho_old);
        // all the charge is on the four nodes of one cell: the round-off of
        // the sums is measured against the largest term, not against one
        double scale = 1.0;
        for (const auto& rho : rho_old) {
          scale = std::max(scale, std::abs(rho) / static_cast<double>(dt));
        }
        const double limit   = 100.0 * eps * scale;
        const auto   n_pairs = push_with_emission(emitters,
                                                electrons,
                                                positrons,
                                                metric,
                                                EB,
                                                params,
                                                dt,
                                                random_pool,
                                                write_previous);

        std::vector<std::vector<double>> J_all(3, std::vector<double>(n_nodes, 0.0));
        const double j_electrons = max_inplane_current(electrons, metric, dt, J_all);
        const double j_positrons = max_inplane_current(positrons, metric, dt, J_all);
        deposit(emitters, metric, dt, J_all);
        std::vector<double> rho_new(n_nodes, 0.0);
        charge_density(emitters, rho_new);
        charge_density(electrons, rho_new);
        charge_density(positrons, rho_new);
        const double residual = continuity_residual(rho_old, rho_new, J_all, dt);

        // the numbers first, so that a run that fails shows them
        const std::string label = "reused slots, " + arm.name + ", " +
                                  (write_previous
                                     ? "policy"
                                     : "control without the previous position");
        std::cout
          << fmt::format(
               "%s: %u pairs, in-plane currents %.3e (electrons) and "
               "%.3e (positrons), continuity residual %.3e (limit %.3e)",
               label.c_str(),
               n_pairs,
               j_electrons,
               j_positrons,
               residual,
               limit)
          << std::endl;

        check(n_pairs > 100u, label + ": too few pairs created");
        if (write_previous) {
          check(j_electrons == 0.0 and j_positrons == 0.0,
                fmt::format("%s: in-plane current of unpushed leptons %.3e "
                            "(electrons), %.3e (positrons) != 0",
                            label.c_str(),
                            j_electrons,
                            j_positrons));
          check(residual < limit,
                fmt::format("%s: continuity residual %.3e (limit %.3e)",
                            label.c_str(),
                            residual,
                            limit));
        } else {
          check(j_electrons > 1.0 and j_positrons > 1.0 and residual > 1.0e3 * limit,
                fmt::format("%s: charge is conserved without the previous "
                            "position (currents %.3e, %.3e, continuity "
                            "residual %.3e)",
                            label.c_str(),
                            j_electrons,
                            j_positrons,
                            residual));
        }
      }
    }
  }

  /*
   * Rate and mean loss at a fixed Gamma_p. One step of 200000 emitters:
   *   - the number of pairs against the expectation p N, within 4 standard
   *     deviations of the binomial law;
   *   - the mean energy lost per interaction against the tabulated <D>,
   *     within 4 standard errors of the sample mean.
   */
  void test_rate_and_loss() {
    const metric_t metric {
      { NX1, NX2 },
      { { 0.0, static_cast<real_t>(NX1) }, { 0.0, static_cast<real_t>(NX2) } },
      {}
    };
    const auto dt = static_cast<real_t>(0.45);
    const ndfield_t<Dim::_2D, 6> EB { "EB", NX1 + 2u * N_GHOSTS, NX2 + 2u * N_GHOSTS };
    random_number_pool_t random_pool { 6789 };

    const npart_t n_emitters = 200000u;
    auto          electrons  = make_species(1, 1.0f, -1.0f, 40000u);
    auto          positrons  = make_species(2, 1.0f, 1.0f, 40000u);
    auto          emitters   = make_species(3, 1.0f, 1.0f, n_emitters);
    fill_emitters(emitters, n_emitters, momentum_at(GAMMA_P), 2u);

    auto        params = make_params(static_cast<real_t>(1e-6));
    const auto& tables = arch::emission::bh::GetTables(params, dt);
    const auto  rate   = tables.rate(GAMMA_P);
    params             = make_params(static_cast<real_t>(0.07 / (rate * dt)));
    const double probability =
      static_cast<double>(params.template get<real_t>(
        "radiation.emission.bethe_heitler.optical_depth")) *
      rate * static_cast<double>(dt);
    const double mean_loss = tables.mean_loss(GAMMA_P);

    const auto K_before = kinetic_energies(emitters);
    const auto n_pairs  = push_with_emission(emitters,
                                            electrons,
                                            positrons,
                                            metric,
                                            EB,
                                            params,
                                            dt,
                                            random_pool);
    const auto K_after  = kinetic_energies(emitters);

    const double expected = probability * n_emitters;
    const double sigma    = std::sqrt(expected * (1.0 - probability));
    check(
      std::abs(static_cast<double>(n_pairs) - expected) < 4.0 * sigma,
      fmt::format("rate: %u pairs, expected %.1f +- %.1f", n_pairs, expected, sigma));

    // losses of the emitters that interacted (the others are not touched)
    double  sum = 0.0, sum_sqr = 0.0;
    npart_t n_changed = 0u;
    for (npart_t p = 0u; p < n_emitters; ++p) {
      const double loss = K_before[p] - K_after[p];
      if (loss != 0.0) {
        sum     += loss;
        sum_sqr += loss * loss;
        ++n_changed;
      }
    }
    check(n_changed == n_pairs, "loss: recoils and pairs differ in number");
    const double mean      = sum / n_changed;
    const double variance  = sum_sqr / n_changed - mean * mean;
    const double std_error = std::sqrt(variance / n_changed);
    check(std::abs(mean - mean_loss) < 4.0 * std_error,
          fmt::format("loss: mean %.5f +- %.5f, tabulated %.5f",
                      mean,
                      std_error,
                      mean_loss));
    std::cout
      << fmt::format("rate: %u pairs, expected %.1f +- %.1f; loss: mean "
                     "%.5f +- %.5f, tabulated %.5f",
                     n_pairs,
                     expected,
                     sigma,
                     mean,
                     std_error,
                     mean_loss)
      << std::endl;
  }

  /*
   * Refusals and capacity:
   *   - full lepton species: the pairs that do not fit are not created, the
   *     corresponding emitters are not touched, and the counter says so;
   *   - a probability above `max_probability` raises, at startup from the
   *     tables and at runtime from the kernel;
   *   - `pair_weight != 1` without `particles.use_weights` raises;
   *   - a NaN optical depth or probability bound raises, and so does a
   *     timestep that is not finite;
   *   - lepton species whose charges do not cancel exactly are refused, and
   *     so is an emitting species of infinite mass.
   */
  void test_refusals_and_capacity() {
    const metric_t metric {
      { NX1, NX2 },
      { { 0.0, static_cast<real_t>(NX1) }, { 0.0, static_cast<real_t>(NX2) } },
      {}
    };
    const auto dt = static_cast<real_t>(0.45);
    const ndfield_t<Dim::_2D, 6> EB { "EB", NX1 + 2u * N_GHOSTS, NX2 + 2u * N_GHOSTS };
    random_number_pool_t random_pool { 4242 };

    auto       params = make_params(static_cast<real_t>(1e-6));
    const auto rate   = arch::emission::bh::GetTables(params, dt).rate(GAMMA_P);
    const auto optical_depth = static_cast<real_t>(0.07 / (rate * dt));
    params                   = make_params(optical_depth);

    // capacity
    {
      const npart_t n_emitters = 2000u, capacity = 64u;
      auto          electrons = make_species(1, 1.0f, -1.0f, capacity);
      auto          positrons = make_species(2, 1.0f, 1.0f, capacity + 10u);
      auto          emitters  = make_species(3, 1.0f, 1.0f, n_emitters);
      fill_emitters(emitters, n_emitters, momentum_at(GAMMA_P), 3u);
      const auto before    = arch::emission::bh::GetCounters();
      const auto K_before  = kinetic_energies(emitters);
      const auto n_pairs   = push_with_emission(emitters,
                                              electrons,
                                              positrons,
                                              metric,
                                              EB,
                                              params,
                                              dt,
                                              random_pool);
      const auto K_after   = kinetic_energies(emitters);
      const auto after     = arch::emission::bh::GetCounters();
      npart_t    n_changed = 0u;
      for (npart_t p = 0u; p < n_emitters; ++p) {
        if (K_before[p] != K_after[p]) {
          ++n_changed;
        }
      }
      check(n_pairs == capacity and electrons.npart() == capacity and
              positrons.npart() == capacity,
            "capacity: wrong number of pairs inserted");
      check(n_changed == capacity, "capacity: recoil without a pair");
      check(after.pairs - before.pairs == capacity, "capacity: wrong pair counter");
      check(after.capacity_dropped > before.capacity_dropped,
            "capacity: dropped pairs not counted");
      std::cout << "capacity: " << n_pairs << " pairs inserted, "
                << after.capacity_dropped - before.capacity_dropped
                << " dropped and counted" << std::endl;
    }

#if !defined(MPI_ENABLED)
    // with MPI an error aborts; these checks need it to throw
    auto raises = [](auto&& f) {
      try {
        f();
      } catch (const std::exception&) {
        return true;
      }
      return false;
    };

    // probability above max_probability: startup check on the tables
    check(raises([&]() {
            const auto too_dense = make_params(
              static_cast<real_t>(2) * optical_depth);
            arch::emission::bh::GetTables(too_dense, dt);
          }),
          "refusal: probability above max_probability not refused at startup");

    // ... and at runtime: tables checked with one optical depth, policy
    // built with a larger one
    check(
      raises([&]() {
        auto& tables    = arch::emission::bh::GetTables(params, dt);
        auto  electrons = make_species(1, 1.0f, -1.0f, 1000u);
        auto  positrons = make_species(2, 1.0f, 1.0f, 1000u);
        auto  emitters  = make_species(3, 1.0f, 1.0f, 100u);
        fill_emitters(emitters, 100u, momentum_at(GAMMA_P), 4u);
        const auto too_dense = make_params(static_cast<real_t>(2) * optical_depth);
        const emission_t emission { electrons,
                                    positrons,
                                    electrons.index(),
                                    positrons.index(),
                                    emitters.mass(),
                                    dt,
                                    0u,
                                    too_dense,
                                    tables,
                                    random_pool };
        const auto policy = kernel::sr::PusherPolicy<metric_t, emission_t> { emission };
        const auto boundaries = kernel::sr::PusherBoundaries<Dim::_2D> {
          { { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
           { PrtlBC::PERIODIC, PrtlBC::PERIODIC } }
        };
        Kokkos::parallel_for("ParticlePusher",
                             emitters.rangeActiveParticles(),
                             kernel::sr::Pusher_kernel<metric_t, decltype(policy)>(
                               pusher_context(emitters, dt),
                               boundaries,
                               emitters,
                               EB,
                               metric,
                               policy));
        emission.numbers_injected();
      }),
      "refusal: probability above max_probability not refused at runtime");

    // pair_weight != 1 without particle weights
    check(raises([&]() {
            const auto weighted = make_params(optical_depth,
                                              static_cast<real_t>(2),
                                              static_cast<real_t>(0.1),
                                              false);
            arch::emission::bh::GetTables(weighted, dt);
          }),
          "refusal: pair_weight != 1 accepted without use_weights");
    // a NaN optical depth or probability bound is refused (it would
    // otherwise switch the process off without a word)
    const auto not_a_number = std::numeric_limits<real_t>::quiet_NaN();
    check(raises([&]() {
            arch::emission::bh::GetTables(make_params(not_a_number), dt);
          }),
          "refusal: NaN optical_depth accepted");
    check(raises([&]() {
            arch::emission::bh::GetTables(
              make_params(optical_depth, ONE, not_a_number),
              dt);
          }),
          "refusal: NaN max_probability accepted");
    // ... and so is a timestep that is not finite: a NaN and minus infinity
    // would pass the bound on the probability
    const auto infinity = std::numeric_limits<real_t>::infinity();
    for (const auto bad_dt : { not_a_number, infinity, -infinity }) {
      check(raises([&]() {
              arch::emission::bh::GetTables(params, bad_dt);
            }),
            fmt::format("refusal: timestep %g accepted",
                        static_cast<double>(bad_dt)));
    }

    // the charges of the two lepton species must cancel exactly: a positron
    // charge one unit in the last place above that is refused by the factory
    const auto make_policy = [&](float positron_charge, float emitter_mass = 1.0f) {
      MockDomain domain;
      domain.add_species(1.0f, -1.0f);
      domain.add_species(1.0f, positron_charge);
      domain.add_species(emitter_mass, 1.0f);
      kernel::sr::MakePusherPolicyEmission<metric_t, MockDomain, EmissionType::BETHE_HEITLER>(
        domain,
        params,
        pusher_context(domain.species[2], dt));
    };
    check(not raises([&]() {
      make_policy(1.0f);
    }),
          "refusal: exactly opposite lepton charges refused");
    check(raises([&]() {
            make_policy(std::nextafter(1.0f, 2.0f));
          }),
          "refusal: lepton charges that do not cancel exactly accepted");
    // the mass of the emitting species must be finite
    check(raises([&]() {
            make_policy(1.0f, std::numeric_limits<float>::infinity());
          }),
          "refusal: emitting species of infinite mass accepted");
    std::cout << "refusals: raised as expected" << std::endl;
#endif
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  try {
    test_charge();
    test_charge_reused_slots();
    test_rate_and_loss();
    test_refusals_and_capacity();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return 0;
}
