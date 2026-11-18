#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "traits/pgen.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

namespace user {
  using namespace ntt;

  template <class M, Dimension D>
  struct InitFields {
    InitFields(M metric_, real_t m_eps) : metric { metric_ }, m_eps { m_eps } {}

    Inline auto A_3(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<3, 3>(x_Cd) +
                     TWO * metric.spin() * metric.template h_<1, 3>(x_Cd) *
                       metric.beta1(x_Cd));
    }

    Inline auto A_1(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<1, 3>(x_Cd) +
                     TWO * metric.spin() * metric.template h_<1, 1>(x_Cd) *
                       metric.beta1(x_Cd));
    }

    Inline auto A_0(const coord_t<D>& x_Cd) const -> real_t {
      real_t g_00 { -metric.alpha(x_Cd) * metric.alpha(x_Cd) +
                    metric.template h_<1, 1>(x_Cd) * metric.beta1(x_Cd) *
                      metric.beta1(x_Cd) };
      return HALF * (metric.template h_<1, 3>(x_Cd) * metric.beta1(x_Cd) +
                     TWO * metric.spin() * g_00);
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i , j + HALF )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF * m_eps;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF * m_eps;

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };

      if (cmp::AlmostZero(x_Ph[1])) {
        return ONE;
      } else {
        return (A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP / m_eps;
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i + HALF , j )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0] - HALF * m_eps;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF * m_eps;
      x0p[1] = xi[1];

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      if (cmp::AlmostZero(x_Ph[1])) {
        return ZERO;
      } else {
        return -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP / m_eps;
      }
    }

    Inline auto bx3(const coord_t<D>& /*x_Ph*/) const -> real_t {
      return ZERO;
    }

    Inline auto dx1(const coord_t<D>& /*x_Ph*/) const -> real_t {
      return ZERO;
    }

    Inline auto dx2(const coord_t<D>& /*x_Ph*/) const -> real_t {
      return ZERO;
    }

    Inline auto dx3(const coord_t<D>& /*x_Ph*/) const -> real_t {
      return ZERO;
    }

  private:
    const M      metric;
    const real_t m_eps;
  };

  template <GRMetricClass M>
  struct PointDistribution {
    PointDistribution(const std::vector<real_t>&   xi_min,
                      const std::vector<real_t>&   xi_max,
                      const real_t                 sigma_thr,
                      const real_t                 dens_thr,
                      const SimulationParams&      params,
                      Domain<SimEngine::GRPIC, M>* domain_ptr)
      : metric { domain_ptr->mesh.metric }
      , EM { domain_ptr->fields.em }
      , density { domain_ptr->fields.buff }
      , sigma_thr { sigma_thr }
      , dens_thr { dens_thr }
      , inv_n0 { ONE / params.template get<real_t>("scales.n0") } {
      std::copy(xi_min.begin(), xi_min.end(), x_min);
      std::copy(xi_max.begin(), xi_max.end(), x_max);

      std::vector<spidx_t> specs {};
      for (auto& sp : domain_ptr->species) {
        if (sp.mass() > 0) {
          specs.push_back(sp.index());
        }
      }

      arch::ComputeMomentWithSpecies<SimEngine::GRPIC, M, FldsID::Rho, 3>(
        params,
        *domain_ptr,
        specs,
        density);
    }

    Inline auto sigma_crit(const coord_t<M::Dim>& x_Ph) const -> bool {
      coord_t<M::Dim> xi { ZERO };
      if constexpr (M::Dim == Dim::_2D) {
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        const auto i1 = static_cast<int>(xi[0]) + static_cast<int>(N_GHOSTS);
        const auto i2 = static_cast<int>(xi[1]) + static_cast<int>(N_GHOSTS);
        const vec_t<Dim::_3D> B_cntrv { EM(i1, i2, em::bx1),
                                        EM(i1, i2, em::bx2),
                                        EM(i1, i2, em::bx3) };
        const vec_t<Dim::_3D> D_cntrv { EM(i1, i2, em::dx1),
                                        EM(i1, i2, em::dx2),
                                        EM(i1, i2, em::dx3) };
        vec_t<Dim::_3D>       B_cov { ZERO };
        metric.template transform<Idx::U, Idx::D>(xi, B_cntrv, B_cov);
        const auto bsqr =
          DOT(B_cntrv[0], B_cntrv[1], B_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        const auto dens = density(i1, i2, 0);
        return (bsqr > sigma_thr * dens) || (dens < dens_thr);
      }
      return false;
    }

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      auto fill = true;
      for (auto d = 0u; d < M::Dim; ++d) {
        fill &= x_Ph[d] > x_min[d] and x_Ph[d] < x_max[d] and sigma_crit(x_Ph);
      }
      return fill ? ONE : ZERO;
    }

  private:
    const M                 metric;
    ndfield_t<M::Dim, 6>    EM;
    ndfield_t<M::Dim, 3>    density;
    tuple_t<real_t, M::Dim> x_min { ZERO };
    tuple_t<real_t, M::Dim> x_max { ZERO };
    const real_t            sigma_thr;
    const real_t            dens_thr;
    const real_t            inv_n0;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::GRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Kerr_Schild, Metric::QKerr_Schild, Metric::Kerr_Schild_0> {}
    };
    static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D> {} };

    const SimulationParams& params;

    const std::vector<real_t> xi_min;
    const std::vector<real_t> xi_max;
    const real_t sigma0, sigma_max, multiplicity, nGJ, temperature, m_eps;

    InitFields<M, D>        init_flds;
    const Metadomain<S, M>* metadomain;

    PGen(SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , xi_min { params.template get<std::vector<real_t>>("setup.xi_min") }
      , xi_max { params.template get<std::vector<real_t>>("setup.xi_max") }
      , sigma_max { params.template get<real_t>("setup.sigma_max") }
      , sigma0 { params.template get<real_t>("scales.sigma0") }
      , multiplicity { params.template get<real_t>("setup.multiplicity") }
      , nGJ { params.template get<real_t>("scales.B0") *
              SQR(params.template get<real_t>("scales.skindepth0")) }
      , temperature { params.template get<real_t>("setup.temperature") }
      , m_eps { params.template get<real_t>("setup.m_eps") }
      , init_flds { m.mesh().metric, m_eps }
      , metadomain { &m } {}

    void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        local_domain.random_pool(),
        temperature);
      const auto spatial_dist = PointDistribution<M>(xi_min,
                                                     xi_max,
                                                     sigma_max / sigma0,
                                                     multiplicity * nGJ,
                                                     params,
                                                     &local_domain);

      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(spatial_dist)>(
        params,
        local_domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        spatial_dist,
        ONE,
        true);
    }

    void CustomPostStep(timestep_t /*step*/,
                        simtime_t /*time*/,
                        Domain<S, M>& local_domain) {
      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        local_domain.random_pool(),
        temperature);
      const auto spatial_dist = PointDistribution<M>(xi_min,
                                                     xi_max,
                                                     sigma_max / sigma0,
                                                     multiplicity * nGJ,
                                                     params,
                                                     &local_domain);
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(spatial_dist)>(
        params,
        local_domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        spatial_dist,
        ONE,
        true);
    }
  };

} // namespace user

#endif
