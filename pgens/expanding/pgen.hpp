#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"           
#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"         
#include "archetypes/utils.h"                

#include "utils/numeric.h"
#include "utils/formatting.h"
#include "metrics/metric_box.h"

#include <plog/Log.h>

namespace user {
  using namespace ntt;

  // expanding-box initialization + empty fields
  template <Dimension D>
  struct InitFieldsExpBox {
    Inline auto ex1(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto ex2(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto ex3(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto bx1(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto bx2(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto bx3(const coord_t<D>&) const -> real_t { return ZERO; }
  };


  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines {
      traits::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Box>::value
    };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_2D, Dim::_3D>::value
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    // user parameters
    const real_t qx;
    const real_t qy;
    const real_t qz;
    const real_t sx;
    const real_t sy;
    const real_t sz;
    const real_t n0_bg;
    const real_t T_bg;

    InitFieldsExpBox<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& mdom)
      : arch::ProblemGenerator<S, M> { p }
      , qx { p.template get<real_t>("metric_box.qx", ZERO) }
      , qy { p.template get<real_t>("metric_box.qy", ZERO) }
      , qz { p.template get<real_t>("metric_box.qz", ZERO) }
      , sx { p.template get<real_t>("metric_box.sx", ZERO) }
      , sy { p.template get<real_t>("metric_box.sy", ZERO) }
      , sz { p.template get<real_t>("metric_box.sz", ZERO) } 
      , n0_bg { p.template get<real_t>("setup.n0_bg", ONE) }
      , T_bg  { p.template get<real_t>("setup.T_bg", 1.0e-3) }
    {
      PLOGI << fmt::format("PGen<Box>: q=({}, {}, {}), s=({}, {}, {}), n0={}, T={}",
                           qx, qy, qz, sx, sy, sz, n0_bg, T_bg); 
      (void) mdom;
    }

    // metric parameters at t=0 and set midstep values.
    inline void InitMetric(Domain<S, M>& domain) {
      auto& box = domain.mesh.metric;
      box.q = { qx, qy, qz };
      box.s = { sx, sy, sz };
      // set a_i, H_i, Delta at midstep
      const real_t dt = params.template get<real_t>("algorithms.timestep.dt");
      box.update(HALF * dt);
      PLOGI << fmt::format("InitMetric: a=({}, {}, {}), H=({}, {}, {}), Δ={}",
                           box.a[0], box.a[1], box.a[2],
                           box.H[0], box.H[1], box.H[2], box.Delta);
    }

    inline void InitFlds(Domain<S, M>& domain) {
     InitMetric(domain);

      Kokkos::deep_copy(domain.fields.em, ZERO);

      auto& box = domain.mesh.metric;

      const real_t Ex_phys = ZERO;   
      const real_t Ey_phys = ZERO; 
      const real_t Ez_phys = ZERO;
      const real_t Bx_phys = ZERO;
      const real_t By_phys = ZERO;
      const real_t Bz_phys = ZERO;  // set if needed

      const real_t Ex_p = Ex_phys * box.Linv(0);  // E′ = E / a
      const real_t Ey_p = Ey_phys * box.Linv(1);
      const real_t Ez_p = Ez_phys * box.Linv(2);
      const real_t Bx_p = Bx_phys * box.Linv(0);  // B′ = B / a
      const real_t By_p = By_phys * box.Linv(1);
      const real_t Bz_p = Bz_phys * box.Linv(2);

      auto em = domain.fields.em;

      if constexpr (M::Dim == Dim::_2D) {
       const int ni = domain.mesh.Ni1_tot();
      const int nj = domain.mesh.Ni2_tot();
        Kokkos::parallel_for(
          "InitEM_primed_2D",
         Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ni, nj}),
          KOKKOS_LAMBDA(const int i, const int j) {
            em(i, j, em::ex1) = Ex_p;
            em(i, j, em::ex2) = Ey_p;
            em(i, j, em::ex3) = Ez_p;
            em(i, j, em::bx1) = Bx_p;
            em(i, j, em::bx2) = By_p;
            em(i, j, em::bx3) = Bz_p;
         });
      } else if constexpr (M::Dim == Dim::_3D) {
        const int ni = domain.mesh.Ni1_tot();
        const int nj = domain.mesh.Ni2_tot();
        const int nk = domain.mesh.Ni3_tot();
        Kokkos::parallel_for(
         "InitEM_primed_3D",
         Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {ni, nj, nk}),
         KOKKOS_LAMBDA(const int i, const int j, const int k) {
           em(i, j, k, em::ex1) = Ex_p;
           em(i, j, k, em::ex2) = Ey_p;
           em(i, j, k, em::ex3) = Ez_p;
           em(i, j, k, em::bx1) = Bx_p;
           em(i, j, k, em::bx2) = By_p;
           em(i, j, k, em::bx3) = Bz_p;
         });
     }
    }


    inline void InitPrtls(Domain<S, M>& local_domain) {
     const auto energy_dist = arch::Maxwellian<S, M>(
        local_domain.mesh.metric,
       local_domain.random_pool,
       T_bg);

    const auto spatial_dist = arch::UniformInjector<S, M, arch::Maxwellian>(
       energy_dist,
        { 1, 2 });

      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
       local_domain,
       spatial_dist,
       n0_bg);
    }

    inline void CustomPostStep(timestep_t, simtime_t t, Domain<S, M>& domain) {
      // to update the metric to mid-step time
      const auto dt = params.template get<real_t>("algorithms.timestep.dt");
      domain.mesh.metric.update(t + HALF * dt);
    }
  };

} // namespace user

#endif
