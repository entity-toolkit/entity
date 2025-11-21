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
    explicit InitFieldsExpBox(real_t bx0, real_t by0, real_t bz0)
      : m_bx{bx0}, m_by{by0}, m_bz{bz0} {}

    Inline auto ex1(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto ex2(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto ex3(const coord_t<D>&) const -> real_t { return ZERO; }

    Inline auto bx1(const coord_t<D>&) const -> real_t { return m_bx; }
    Inline auto bx2(const coord_t<D>&) const -> real_t { return m_by; }
    Inline auto bx3(const coord_t<D>&) const -> real_t { return m_bz; }

  private:
    const real_t m_bx, m_by, m_bz;
  };


  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits
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
    const real_t qx, qy, qz;
    const real_t sx, sy, sz;
    const real_t n0_bg, T_bg;
    const real_t B0_x, B0_y, B0_z;

    InitFieldsExpBox<M::Dim> init_flds;

    // -------- pgen
    inline PGen(const SimulationParams& p, const Metadomain<S, M>& mdom)
      : arch::ProblemGenerator<S, M>{p}
      , qx{p.template get<real_t>("metric_box.qx", ZERO)}
      , qy{p.template get<real_t>("metric_box.qy", ZERO)}
      , qz{p.template get<real_t>("metric_box.qz", ZERO)}
      , sx{p.template get<real_t>("metric_box.sx", ZERO)}
      , sy{p.template get<real_t>("metric_box.sy", ZERO)}
      , sz{p.template get<real_t>("metric_box.sz", ZERO)}
      , n0_bg{p.template get<real_t>("setup.n0_bg", ONE)}
      , T_bg{p.template get<real_t>("setup.T_bg", 1.0e-3)}
      , B0_x{p.template get<real_t>("setup.B0_x", ZERO)}
      , B0_y{p.template get<real_t>("setup.B0_y", ZERO)}
      , B0_z{p.template get<real_t>("setup.B0_z", ONE)}
      , init_flds{B0_x, B0_y, B0_z}   
    {
      PLOGI << fmt::format(
        "PGen<Box>: q=({}, {}, {}), s=({}, {}, {}), n0={}, T={}, B=({}, {}, {})",
        qx, qy, qz, sx, sy, sz, n0_bg, T_bg, B0_x, B0_y, B0_z);
      
      PLOGI << "PGen<Box>: EB on the mesh is treated as E' and B' (grid variables); "
         "physical E,B for the pusher are obtained via metric.transform_xyz.";

      (void) mdom;
    }




    inline void InitPrtls(Domain<S, M>& local_domain) {
      // Maxwellian energy distribution (same for both species 1 and 2)
      const auto energy_dist = arch::Maxwellian<S, M>(
        local_domain.mesh.metric,
        local_domain.random_pool,
        T_bg);

      const auto spatial_dist =
        arch::UniformInjector<S, M, arch::Maxwellian>(energy_dist, { 1, 2 });

      arch::InjectUniform<S, M,
                          arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        spatial_dist,
        n0_bg);
    }
    
    // maybe we'll put it outside
//    inline void CustomPostStep(timestep_t dt, simtime_t t, Domain<S, M>& domain) {
    // Update the metric at the mid‑point of the next time interval
//    domain.mesh.metric.update(static_cast<real_t>(t) + 0.5 * static_cast<real_t>(dt));
//}
    inline void CustomPostStep(timestep_t, simtime_t t, Domain<S, M>& domain) {
      domain.mesh.metric.update(t);  // not HALF time
    }
  };

} // namespace user

#endif
