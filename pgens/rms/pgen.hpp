#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/utils.h"
#include "framework/containers/particles.h"
#include "framework/domain/metadomain.h"

#include <cstdint>
#include <utility>

namespace user {
  using namespace ntt;

  enum Component : idx_t {
    comp_n      = 0u,
    comp_rho    = 1u,
    comp_rho_vx = 2u,
    comp_rho_vy = 3u,
    comp_rho_vz = 4u,
    comp_n_t    = 5u,
  };

  struct ComputeN {
    static constexpr uint8_t N = 1;

    Inline void operator()(const ParticleArrays& /* prtls */,
                           float /* mass */,
                           float /* charge */,
                           prtlidx_t /* p */,
                           list_t<real_t, N>& contribs) const {
      contribs[0] = ONE;
    }
  };

  struct ComputeRhoV {
    static constexpr uint8_t N = 5;

    Inline void operator()(const ParticleArrays& prtls,
                           float                 mass,
                           float /* charge */,
                           prtlidx_t          p,
                           list_t<real_t, N>& contribs) const {
      contribs[0]      = ONE;
      contribs[1]      = mass;
      const auto gamma = U2GAMMA(prtls.ux1(p), prtls.ux2(p), prtls.ux3(p));
      contribs[2]      = mass * prtls.ux1(p) / gamma;
      contribs[3]      = mass * prtls.ux2(p) / gamma;
      contribs[4]      = mass * prtls.ux3(p) / gamma;
    }
  };

  struct ComputePhotonTmunu {
    static constexpr uint8_t N = 5;

    Inline void operator()(const ParticleArrays& prtls,
                           float /* mass */,
                           float /* charge */,
                           prtlidx_t          p,
                           list_t<real_t, N>& contribs) const {
      const auto energy = math::sqrt(
        SQR(prtls.ux1(p)) + SQR(prtls.ux2(p)) + SQR(prtls.ux3(p)));
      contribs[0] = ONE;
      contribs[1] = energy;
      contribs[2] = prtls.ux1(p);
      contribs[3] = prtls.ux2(p);
      contribs[4] = prtls.ux3(p);
    }
  };

  template <int C>
  struct ComputeN_Rho_Vi {
    static constexpr uint8_t N = 1;

    Inline void operator()(const ParticleArrays& prtls,
                           float                 mass,
                           float /* charge */,
                           prtlidx_t          p,
                           list_t<real_t, N>& contribs) const {
      const auto gamma = U2GAMMA(prtls.ux1(p), prtls.ux2(p), prtls.ux3(p));
      if constexpr (C == -1) {
        contribs[0] = ONE;
      } else if constexpr (C == 0) {
        contribs[0] = mass;
      } else if constexpr (C == 1) {
        contribs[0] = mass * prtls.ux1(p) / gamma;
      } else if constexpr (C == 2) {
        contribs[0] = mass * prtls.ux2(p) / gamma;
      } else if constexpr (C == 3) {
        contribs[0] = mass * prtls.ux3(p) / gamma;
      }
    }
  };

  template <Dimension D>
  struct Normalize {
    ndfield_t<D, 6> buffer;
    const idx_t     comp, comp_norm;

    Normalize(const ndfield_t<D, 6>& buff, idx_t comp, idx_t comp_norm)
      : buffer { buff }
      , comp { comp }
      , comp_norm { comp_norm } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (buffer(i1, comp_norm) < 1e-6) {
          buffer(i1, comp) = ZERO;
        } else {
          buffer(i1, comp) /= buffer(i1, comp_norm);
        }
      } else {
        raise::KernelError(HERE, "Normalize is only implemented for 1D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        if (buffer(i1, i2, comp_norm) < 1e-6) {
          buffer(i1, i2, comp) = ZERO;
        } else {
          buffer(i1, i2, comp) /= buffer(i1, i2, comp_norm);
        }
      } else {
        raise::KernelError(HERE, "Normalize is only implemented for 2D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        if (buffer(i1, i2, i3, comp_norm) < 1e-6) {
          buffer(i1, i2, i3, comp) = ZERO;
        } else {
          buffer(i1, i2, i3, comp) /= buffer(i1, i2, i3, comp_norm);
        }
      } else {
        raise::KernelError(HERE, "Normalize is only implemented for 3D");
      }
    }
  };

  template <Dimension D>
  struct ComputePressure {
    static constexpr uint8_t N = 1;
    ndfield_t<D, 6>          buffer;

    const idx_t rho, rho_vx, rho_vy, rho_vz;

    ComputePressure(const ndfield_t<D, 6>& buff,
                    idx_t                  rho    = comp_rho,
                    idx_t                  rho_vx = comp_rho_vx,
                    idx_t                  rho_vy = comp_rho_vy,
                    idx_t                  rho_vz = comp_rho_vz)
      : buffer { buff }
      , rho { rho }
      , rho_vx { rho_vx }
      , rho_vy { rho_vy }
      , rho_vz { rho_vz } {}

    Inline void operator()(const ParticleArrays& prtls,
                           float                 mass,
                           float,
                           prtlidx_t          p,
                           list_t<real_t, N>& contribs) const {
      real_t Vx1 { ZERO }, Vx2 { ZERO }, Vx3 { ZERO };
      if constexpr (D == Dim::_1D) {
        Vx1 = buffer(prtls.i1(p) + N_GHOSTS, rho_vx) /
              buffer(prtls.i1(p) + N_GHOSTS, rho);
        Vx2 = buffer(prtls.i1(p) + N_GHOSTS, rho_vy) /
              buffer(prtls.i1(p) + N_GHOSTS, rho);
        Vx3 = buffer(prtls.i1(p) + N_GHOSTS, rho_vz) /
              buffer(prtls.i1(p) + N_GHOSTS, rho);
      } else if constexpr (D == Dim::_2D) {
        Vx1 = buffer(prtls.i1(p) + N_GHOSTS, prtls.i2(p) + N_GHOSTS, rho_vx) /
              buffer(prtls.i1(p) + N_GHOSTS, prtls.i2(p) + N_GHOSTS, rho);
        Vx2 = buffer(prtls.i1(p) + N_GHOSTS, prtls.i2(p) + N_GHOSTS, rho_vy) /
              buffer(prtls.i1(p) + N_GHOSTS, prtls.i2(p) + N_GHOSTS, rho);
        Vx3 = buffer(prtls.i1(p) + N_GHOSTS, prtls.i2(p) + N_GHOSTS, rho_vz) /
              buffer(prtls.i1(p) + N_GHOSTS, prtls.i2(p) + N_GHOSTS, rho);
      } else if constexpr (D == Dim::_3D) {
        Vx1 = buffer(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     rho_vx) /
              buffer(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     rho);
        Vx2 = buffer(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     rho_vy) /
              buffer(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     rho);
        Vx3 = buffer(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     rho_vz) /
              buffer(prtls.i1(p) + N_GHOSTS,
                     prtls.i2(p) + N_GHOSTS,
                     prtls.i3(p) + N_GHOSTS,
                     rho);
      }
      const auto Gamma = ONE / math::sqrt(ONE - SQR(Vx1) - SQR(Vx2) - SQR(Vx3));
      const auto gamma = U2GAMMA(prtls.ux1(p), prtls.ux2(p), prtls.ux3(p));
      contribs[0]      = mass *
                    (SQR(Gamma) * SQR(gamma - Vx1 * prtls.ux1(p) -
                                      Vx2 * prtls.ux2(p) - Vx3 * prtls.ux3(p)) -
                     ONE) /
                    (THREE * gamma);
    }
  };

  template <Dimension D>
  struct ComputePhotonTemperature {
    const ndfield_t<D, 6> t_array;
    const idx_t comp_t, comp_n, comp_t00, comp_t01, comp_t02, comp_t03;

    ComputePhotonTemperature(ndfield_t<D, 6>& t_array,
                             idx_t            comp_t,
                             idx_t            comp_n,
                             idx_t            comp_t00,
                             idx_t            comp_t01,
                             idx_t            comp_t02,
                             idx_t            comp_t03)
      : t_array { t_array }
      , comp_t { comp_t }
      , comp_n { comp_n }
      , comp_t00 { comp_t00 }
      , comp_t01 { comp_t01 }
      , comp_t02 { comp_t02 }
      , comp_t03 { comp_t03 } {}

    Inline auto Gamma(real_t T00_Sqr, real_t T0i_Sqr) const -> real_t {
      return HALF * math::sqrt(T0i_Sqr) *
             math::sqrt(
               ONE / (T0i_Sqr + math::sqrt(T00_Sqr) *
                                  (math::sqrt(FOUR * T00_Sqr - THREE * T0i_Sqr) -
                                   TWO * math::sqrt(T00_Sqr))));
    }

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto T0i_Sqr  = (SQR(t_array(i1, comp_t01)) +
                              SQR(t_array(i1, comp_t02)) +
                              SQR(t_array(i1, comp_t03)));
        const auto T00_Sqr  = SQR(t_array(i1, comp_t00));
        t_array(i1, comp_t) = (math::sqrt(FOUR * T00_Sqr - THREE * T0i_Sqr) -
                               math::sqrt(T00_Sqr)) /
                              (t_array(i1, comp_n) * Gamma(T00_Sqr, T0i_Sqr) *
                               static_cast<real_t>(2.7));
      } else {
        raise::KernelError(HERE, "ComputePhotonTemperature is only implemented for 1D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        const auto T0i_Sqr = (SQR(t_array(i1, i2, comp_t01)) +
                              SQR(t_array(i1, i2, comp_t02)) +
                              SQR(t_array(i1, i2, comp_t03)));
        const auto T00_Sqr = SQR(t_array(i1, i2, comp_t00));
        t_array(i1, i2, comp_t) = (math::sqrt(FOUR * T00_Sqr - THREE * T0i_Sqr) -
                                   math::sqrt(T00_Sqr)) /
                                  (t_array(i1, i2, comp_n) *
                                   Gamma(T00_Sqr, T0i_Sqr) *
                                   static_cast<real_t>(2.7));
      } else {
        raise::KernelError(HERE, "ComputePhotonTemperature is only implemented for 2D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        const auto T0i_Sqr = (SQR(t_array(i1, i2, i3, comp_t01)) +
                              SQR(t_array(i1, i2, i3, comp_t02)) +
                              SQR(t_array(i1, i2, i3, comp_t03)));
        const auto T00_Sqr = SQR(t_array(i1, i2, i3, comp_t00));
        t_array(i1, i2, i3, comp_t) = (math::sqrt(FOUR * T00_Sqr - THREE * T0i_Sqr) -
                                       math::sqrt(T00_Sqr)) /
                                      (t_array(i1, i2, i3, comp_n) *
                                       Gamma(T00_Sqr, T0i_Sqr) *
                                       static_cast<real_t>(2.7));
      } else {
        raise::KernelError(HERE, "ComputePhotonTemperature is only implemented for 3D");
      }
    }
  };

  template <CartesianMetricClass M>
  struct PhotonSpatialDistribution {
    const ndfield_t<M::Dim, 6> n_array;

    const M      metric;
    const real_t nmax { 4.0 };

    PhotonSpatialDistribution(const ndfield_t<M::Dim, 6>& n_array, const M& metric)
      : n_array { n_array }
      , metric { metric } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      coord_t<M::Dim> x_Cd { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);
      if constexpr (M::Dim == Dim::_1D) {
        return n_array(static_cast<int>(x_Cd[0]) + N_GHOSTS, comp_n) / nmax;
      } else if constexpr (M::Dim == Dim::_2D) {
        return n_array(static_cast<int>(x_Cd[0]) + N_GHOSTS,
                       static_cast<int>(x_Cd[1]) + N_GHOSTS,
                       comp_n) /
               nmax;
      } else if constexpr (M::Dim == Dim::_3D) {
        return n_array(static_cast<int>(x_Cd[0]) + N_GHOSTS,
                       static_cast<int>(x_Cd[1]) + N_GHOSTS,
                       static_cast<int>(x_Cd[2]) + N_GHOSTS,
                       comp_n) /
               nmax;
      }
    }
  };

  template <Dimension D>
  struct PlanckDistribution {
    const real_t T_ph_inj;
    const real_t boost_beta;

    random_number_pool_t random_pool;

    PlanckDistribution(real_t T_ph_inj, real_t boost_beta, random_number_pool_t& pool)
      : T_ph_inj { T_ph_inj }
      , boost_beta { boost_beta }
      , random_pool { pool } {}

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& k) const {
      real_t     prob { ZERO }, n { ZERO };
      auto       gen   = random_pool.get_state();
      const auto rnd   = Random<real_t>(gen);
      const auto rnd1  = Random<real_t>(gen);
      const auto rnd2  = Random<real_t>(gen);
      const auto rnd3  = Random<real_t>(gen);
      const auto rndth = Random<real_t>(gen);
      const auto rndph = Random<real_t>(gen);
      random_pool.free_state(gen);

      while ((prob < rnd) and (n < 40)) {
        n    += ONE;
        prob += ONE / (static_cast<real_t>(1.20206) * CUBE(n));
      }
      const auto energy = -static_cast<real_t>(2.7) * T_ph_inj *
                          math::log(
                            rnd1 * rnd2 * rnd3 + static_cast<real_t>(1e-16)) /
                          n;
      const auto costh = TWO * rndth - ONE;
      const auto phi   = static_cast<real_t>(constant::TWO_PI) * rndph;

      k[0] = energy * math::sqrt(ONE - SQR(costh)) * math::cos(phi);
      k[1] = energy * math::sqrt(ONE - SQR(costh)) * math::sin(phi);
      k[2] = energy * costh;

      // boost the photon momentum in the -x direction
      const auto gamma = ONE / math::sqrt(ONE - SQR(boost_beta));
      const auto kx    = k[0];
      const auto ky    = k[1];
      const auto kz    = k[2];
      k[0]             = gamma * (kx - boost_beta * energy);
      k[1]             = ky;
      k[2]             = kz;
    }
  };

  template <Dimension D>
  struct InitFields {
    /*
      Sets up magnetic and electric field components for the simulation.
      Must satisfy E = -v x B for Lorentz Force to be zero.

      @param bmag: magnetic field scaling
      @param thetaB: Bx = bmag * cos(thetaB)
      @param beta_upstream: drift three-velocity in the x direction
    */
    InitFields(real_t bmag, real_t thetaB, real_t beta_upstream)
      : Bmag { bmag }
      , thetaB { thetaB * static_cast<real_t>(convert::deg2rad) }
      , beta_upstream { beta_upstream } {}

    // magnetic field components
    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return Bmag * math::cos(thetaB);
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return Bmag * math::sin(thetaB);
    }

    // electric field components
    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return -beta_upstream * Bmag * math::sin(thetaB);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t Bmag, thetaB, beta_upstream;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };
    const SimulationParams& params;
    Metadomain<S, M>&       metadomain;

    // domain properties
    const real_t global_xmin, global_xmax;
    // gas properties
    const real_t beta_upstream, Te, Te_ovr_Ti;
    // magnetic field properties
    const real_t Bmag, thetaB;
    // photon properties
    // const real_t photon_inj_rate; // units of n0 / time
    const real_t photon_density; // units of n0
    const real_t T_ph_inj;       // photon injection temperature
    // plasma injector properties
    const real_t filling_fraction, beta_injector;
    const int    injection_interval;

    InitFields<D> init_flds;

    PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , global_xmin { metadomain.mesh().extent(in::x1).first }
      , global_xmax { metadomain.mesh().extent(in::x1).second }
      , beta_upstream { params.template get<real_t>("setup.beta_upstream") }
      , Te { params.template get<real_t>("setup.Te") }
      , Te_ovr_Ti { params.template get<real_t>("setup.Te_ovr_Ti", ONE) }
      , Bmag { params.template get<real_t>("setup.Bmag", ZERO) }
      , thetaB { params.template get<real_t>("setup.thetaB", ZERO) }
      // , photon_inj_rate { params.template get<real_t>("setup.photon_inj_rate", ZERO) }
      , photon_density { params.template get<real_t>("setup.photon_density", ZERO) }
      , T_ph_inj { params.template get<real_t>("setup.T_ph_inj") }
      , filling_fraction { params.template get<real_t>("setup.filling_fraction",
                                                       1.0) }
      , beta_injector { params.template get<real_t>("setup.beta_injector", 1.0) }
      , injection_interval { params.template get<int>(
          "setup.injection_interval",
          100) }
      , init_flds { Bmag, thetaB, beta_upstream } {}

    auto MatchFields(simtime_t) const -> InitFields<D> {
      return init_flds;
    }

    auto FixFieldsConst(const bc_in&, const em& comp) const
      -> std::pair<real_t, bool> {
      if (comp == em::ex1) {
        return { init_flds.ex1({ ZERO }), true };
      } else if ((comp == em::ex2) or (comp == em::ex3)) {
        return { ZERO, true };
      } else if (comp == em::bx1) {
        return { init_flds.bx1({ ZERO }), true };
      } else if (comp == em::bx2) {
        return { init_flds.bx2({ ZERO }), true };
      } else if (comp == em::bx3) {
        return { init_flds.bx3({ ZERO }), true };
      } else {
        raise::Error("Invalid component", HERE);
        return { ZERO, false };
      }
    }

    void InitPrtls(Domain<S, M>& domain) {
      /*
       *  Plasma setup as partially filled box
       *
       *  Plasma setup:
       *
       * global_xmin                            global_xmax
       * |                                      |
       * V                                      V
       * |:::::::::::|..........................|
       *             ^
       *             |
       *        filling_fraction
       */

      // minimum and maximum position of particles
      real_t xg_min = global_xmin;
      real_t xg_max = global_xmin + filling_fraction * (global_xmax - global_xmin);

      // define box to inject into
      boundaries_t<real_t> box;
      // loop over all dimensions
      for (auto d { 0u }; d < (unsigned int)M::Dim; ++d) {
        // compute the range for the x-direction
        if (d == static_cast<decltype(d)>(in::x1)) {
          box.emplace_back(xg_min, xg_max);
        } else {
          // inject into full range in other directions
          box.push_back(Range::All);
        }
      }

      const auto gamma_upstream = ONE / math::sqrt(ONE - SQR(beta_upstream));

      // inject particles
      arch::InjectUniformMaxwellians<S, M>(
        params,
        domain,
        TWO,
        std::make_pair(Te, Te / Te_ovr_Ti),
        { 1, 2 },
        std::make_pair(
          std::vector<real_t> { -gamma_upstream * beta_upstream, ZERO, ZERO },
          std::vector<real_t> { -gamma_upstream * beta_upstream, ZERO, ZERO }),
        false,
        box);

      const auto planck_dist = PlanckDistribution<M::Dim>(T_ph_inj,
                                                          beta_upstream,
                                                          domain.random_pool());
      arch::InjectUniform(params, domain, 3, planck_dist, photon_density, false, box);
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      const auto dt = params.template get<real_t>("algorithms.timestep.dt");

      if (step % injection_interval == 0) {
        /*
         *  Replenish plasma in a moving injector
         *
         *  Injector setup:
         *
         * global_xmin           purge/replenish  global_xmax
         * |         x_init            |          |
         * V           v               V          V
         * |:::::::::::;::::::::::|\\\\\\\\|......|
         *                       xmin    xmax
         *                                 ^
         *                                 |
         *                           moving injector
         */

        // initial position of injector
        const auto x_init = global_xmin +
                            filling_fraction * (global_xmax - global_xmin);

        // compute the position of the injector after the current timestep
        const auto xmax = std::min<real_t>(x_init + beta_injector * (step + 1) * dt,
                                           global_xmax);

        // compute the beginning of the injected region
        const auto xmin = (step == 0)
                            ? std::max<real_t>(x_init - beta_upstream * dt,
                                               global_xmin)
                            : xmax - injection_interval * dt * beta_injector -
                                (injection_interval + 1) * dt * beta_upstream;

        // define indice range to reset fields
        boundaries_t<bool> incl_ghosts;
        for (auto d = 0; d < M::Dim; ++d) {
          incl_ghosts.emplace_back(false, false);
        }

        // define box to reset fields
        boundaries_t<real_t> purge_box;
        // loop over all dimension
        for (auto d = 0u; d < M::Dim; ++d) {
          if (d == 0) {
            purge_box.emplace_back(xmin, global_xmax);
          } else {
            purge_box.push_back(Range::All);
          }
        }

        const auto extent = domain.mesh.ExtentToRange(purge_box, incl_ghosts);
        tuple_t<ncells_t, M::Dim> x_min { 0 }, x_max { 0 };
        for (auto d = 0; d < M::Dim; ++d) {
          x_min[d] = extent[d].first;
          x_max[d] = extent[d].second;
        }

        Kokkos::parallel_for("ResetFields",
                             CreateRangePolicy<M::Dim>(x_min, x_max),
                             arch::SetEMFields_kernel<S, M, decltype(init_flds)> {
                               domain.fields.em,
                               init_flds,
                               domain.mesh.metric });
        metadomain.CommunicateFields(domain, Comm::E | Comm::B);

        /*
          tag particles inside the injection zone as dead
        */
        // const auto& mesh = domain.mesh;

        // loop over particle species
        // for (auto& species : domain.species) {
        //   // get particle properties
        //   auto  i1      = species.i1;
        //   auto  dx1     = species.dx1;
        //   auto  tag     = species.tag;

        //   Kokkos::parallel_for(
        //     "RemoveParticles",
        //     species.rangeActiveParticles(),
        //     Lambda(prtlidx_t p) {
        //       // check if the particle is already dead
        //       if (tag(p) == ParticleTag::dead) {
        //         return;
        //       }
        //       const auto x_Cd = static_cast<real_t>(i1(p)) +
        //                         static_cast<real_t>(dx1(p));
        //       const auto x_Ph = mesh.metric.template convert<1, Crd::Cd, Crd::XYZ>(
        //         x_Cd);

        //       if (x_Ph > xmin) {
        //         tag(p) = ParticleTag::dead;
        //       }
        //     });
        // }

        // define box to inject into
        boundaries_t<real_t> inj_box;
        // loop over all dimension
        for (auto d = 0u; d < M::Dim; ++d) {
          if (d == 0) {
            inj_box.emplace_back(xmin, xmax);
          } else {
            inj_box.push_back(Range::All);
          }
        }

        const auto gamma_upstream = ONE / math::sqrt(ONE - SQR(beta_upstream));

        // same maxwell distribution as above
        arch::InjectUniformMaxwellians<S, M>(
          params,
          domain,
          TWO,
          std::make_pair(Te, Te / Te_ovr_Ti),
          { 1, 2 },
          std::make_pair(
            std::vector<real_t> { -gamma_upstream * beta_upstream, ZERO, ZERO },
            std::vector<real_t> { -gamma_upstream * beta_upstream, ZERO, ZERO }),
          false,
          inj_box);

        // replenish photons
        const auto planck_dist = PlanckDistribution<M::Dim>(T_ph_inj,
                                                            beta_upstream,
                                                            domain.random_pool());
        arch::InjectUniform(params, domain, 3, planck_dist, photon_density, false, inj_box);
      }

      // {
      //   /*
      //    * Inject photons
      //    */
      //   auto compute_n = ComputeN {};
      //   arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_n), 6>(
      //     params,
      //     domain,
      //     { 1, 2 },
      //     domain.fields.bckp,
      //     { comp_n },
      //     compute_n);
      //
      //   // inject photons with a Planck distribution in energy and spatial distribution following the plasma density
      //   const auto energy_dist = PlanckDistribution<M::Dim>(T_ph_inj,
      //                                                       domain.random_pool());
      //   const auto spatial_dist = PhotonSpatialDistribution<M>(domain.fields.bckp,
      //                                                          domain.mesh.metric);
      //   arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(spatial_dist)>(
      //     params,
      //     domain,
      //     3,
      //     energy_dist,
      //     spatial_dist,
      //     static_cast<real_t>(photon_inj_rate * dt));
      // }
    }

    void CustomFieldOutput(const std::string& label,
                           ndfield_t<D, 6>&   buff,
                           cellidx_t          buff_idx,
                           timestep_t,
                           simtime_t,
                           const Domain<S, M>& domain) {
      const uint8_t smoothing_order = 2u * N_GHOSTS;
      if (label == "Vbx") {
        /**
         * buff_idx + 1 -> rho_e + rho_i
         */
        auto compute_rho = ComputeN_Rho_Vi<0> {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_rho), 6>(
          params,
          domain,
          { 1, 2 },
          buff,
          { (idx_t)((buff_idx + 1) % 6) },
          compute_rho,
          smoothing_order);
        /**
         * buff_idx -> rho_e * vxe + rho_i * vxi
         */
        auto compute_rho_vx = ComputeN_Rho_Vi<1> {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_rho_vx), 6>(
          params,
          domain,
          { 1, 2 },
          buff,
          { (idx_t)buff_idx },
          compute_rho_vx,
          smoothing_order);
        /**
         * buff_idx -> vx = (rho_e * vxe + rho_i * vxi) / (rho_e + rho_i)
         */
        Kokkos::parallel_for(
          "ComputeVx",
          domain.mesh.rangeActiveCells(),
          Normalize<D> { buff, (idx_t)(buff_idx), (idx_t)((buff_idx + 1) % 6) });
      } else if (label == "Vby") {
        /**
         * buff_idx + 1 -> rho_e + rho_i
         */
        auto compute_rho = ComputeN_Rho_Vi<0> {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_rho), 6>(
          params,
          domain,
          { 1, 2 },
          buff,
          { (idx_t)((buff_idx + 1) % 6) },
          compute_rho,
          smoothing_order);
        /**
         * buff_idx -> rho_e * vye + rho_i * vyi
         */
        auto compute_rho_vy = ComputeN_Rho_Vi<2> {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_rho_vy), 6>(
          params,
          domain,
          { 1, 2 },
          buff,
          { (idx_t)(buff_idx) },
          compute_rho_vy,
          smoothing_order);
        /**
         * buff_idx -> vy = (rho_e * vye + rho_i * vyi) / (rho_e + rho_i)
         */
        Kokkos::parallel_for(
          "ComputeVy",
          domain.mesh.rangeActiveCells(),
          Normalize<D> { buff, (idx_t)(buff_idx), (idx_t)((buff_idx + 1) % 6) });
      } else if (label == "Vbz") {
        /**
         * buff_idx + 1 -> rho_e + rho_i
         */
        auto compute_rho = ComputeN_Rho_Vi<0> {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_rho), 6>(
          params,
          domain,
          { 1, 2 },
          buff,
          { (idx_t)((buff_idx + 1) % 6) },
          compute_rho,
          smoothing_order);
        /**
         * buff_idx -> rho_e * vye + rho_i * vyi
         */
        auto compute_rho_vz = ComputeN_Rho_Vi<3> {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_rho_vz), 6>(
          params,
          domain,
          { 1, 2 },
          buff,
          { (idx_t)buff_idx },
          compute_rho_vz,
          smoothing_order);
        /**
         * buff_idx -> vy = (rho_e * vye + rho_i * vyi) / (rho_e + rho_i)
         */
        Kokkos::parallel_for(
          "ComputeVy",
          domain.mesh.rangeActiveCells(),
          Normalize<D> { buff, (idx_t)(buff_idx), (idx_t)((buff_idx + 1) % 6) });
      } else if (label == "Temperature_e") {
        /**
         * buff_idx + 1 -> n_e + n_i
         * buff_idx + 2 -> rho_e + rho_i
         * buff_idx + 3 -> rho_e * vxe + rho_i * vxi
         * buff_idx + 4 -> rho_e * vye + rho_i * vyi
         * buff_idx + 5 -> rho_e * vze + rho_i * vzi
         */
        auto compute_rho_v = ComputeRhoV {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_rho_v), 6>(
          params,
          domain,
          { 1, 2 },
          buff,
          { (idx_t)((buff_idx + 1) % 6),
            (idx_t)((buff_idx + 2) % 6),
            (idx_t)((buff_idx + 3) % 6),
            (idx_t)((buff_idx + 4) % 6),
            (idx_t)((buff_idx + 5) % 6) },
          compute_rho_v,
          smoothing_order);
        /**
         * buff_idx -> n_e * T_e
         */
        auto compute_pressure = ComputePressure<D> { buff,
                                                     (idx_t)((buff_idx + 2) % 6),
                                                     (idx_t)((buff_idx + 3) % 6),
                                                     (idx_t)((buff_idx + 4) % 6),
                                                     (idx_t)((buff_idx + 5) % 6) };
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_pressure), 6>(
          params,
          domain,
          { 1 },
          buff,
          { (idx_t)buff_idx },
          compute_pressure,
          smoothing_order);
        /**
         * buff_idx + 1 -> n_e
         */
        auto compute_n = ComputeN_Rho_Vi<-1> {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_n), 6>(
          params,
          domain,
          { 1 },
          buff,
          { (idx_t)((buff_idx + 1) % 6) },
          compute_n,
          smoothing_order);
        /**
         * buff_idx -> T_e
         */
        Kokkos::parallel_for(
          "ComputeTe",
          domain.mesh.rangeActiveCells(),
          Normalize<D> { buff, (idx_t)(buff_idx), (idx_t)((buff_idx + 1) % 6) });
      } else if (label == "Temperature_i") {
        /**
         * buff_idx + 1 -> n_e + n_i
         * buff_idx + 2 -> rho_e + rho_i
         * buff_idx + 3 -> rho_e * vxe + rho_i * vxi
         * buff_idx + 4 -> rho_e * vye + rho_i * vyi
         * buff_idx + 5 -> rho_e * vze + rho_i * vzi
         */
        auto compute_rho_v = ComputeRhoV {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_rho_v), 6>(
          params,
          domain,
          { 1, 2 },
          buff,
          { (idx_t)((buff_idx + 1) % 6),
            (idx_t)((buff_idx + 2) % 6),
            (idx_t)((buff_idx + 3) % 6),
            (idx_t)((buff_idx + 4) % 6),
            (idx_t)((buff_idx + 5) % 6) },
          compute_rho_v,
          smoothing_order);
        /**
         * buff_idx -> n_i * T_i
         */
        auto compute_pressure = ComputePressure<D> { buff,
                                                     (idx_t)((buff_idx + 2) % 6),
                                                     (idx_t)((buff_idx + 3) % 6),
                                                     (idx_t)((buff_idx + 4) % 6),
                                                     (idx_t)((buff_idx + 5) % 6) };
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_pressure), 6>(
          params,
          domain,
          { 2 },
          buff,
          { (idx_t)buff_idx },
          compute_pressure,
          smoothing_order);
        /**
         * buff_idx + 1 -> n_i
         */
        auto compute_n = ComputeN_Rho_Vi<-1> {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_n), 6>(
          params,
          domain,
          { 2 },
          buff,
          { (idx_t)((buff_idx + 1) % 6) },
          compute_n,
          smoothing_order);
        /**
         * buff_idx -> T_i
         */
        Kokkos::parallel_for(
          "ComputeTi",
          domain.mesh.rangeActiveCells(),
          Normalize<D> { buff, (idx_t)buff_idx, (idx_t)((buff_idx + 1) % 6) });
      } else if (label == "Temperature_ph") {
        /**
         * buff_idx + 1 -> n_ph
         * buff_idx + 2 -> T^00_ph
         * buff_idx + 3 -> T^0x_ph
         * buff_idx + 4 -> T^0y_ph
         * buff_idx + 5 -> T^0z_ph
         */
        auto compute_t_munu = ComputePhotonTmunu {};
        arch::ComputeMomentWithSpeciesNew<S, M, decltype(compute_t_munu), 6>(
          params,
          domain,
          { 3 },
          buff,
          { (idx_t)((buff_idx + 1) % 6),
            (idx_t)((buff_idx + 2) % 6),
            (idx_t)((buff_idx + 3) % 6),
            (idx_t)((buff_idx + 4) % 6),
            (idx_t)((buff_idx + 5) % 6) },
          compute_t_munu,
          smoothing_order);
        /**
         * buff_idx -> T_ph
         */
        Kokkos::parallel_for(
          "ComputeT_ph",
          domain.mesh.rangeActiveCells(),
          ComputePhotonTemperature<D> { buff,
                                        (idx_t)buff_idx,
                                        (idx_t)((buff_idx + 1) % 6),
                                        (idx_t)((buff_idx + 2) % 6),
                                        (idx_t)((buff_idx + 3) % 6),
                                        (idx_t)((buff_idx + 4) % 6),
                                        (idx_t)((buff_idx + 5) % 6) });
      }
    }
  };
} // namespace user

#endif // PROBLEM_GENERATOR_H
