#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <fstream>
#include <iostream>

namespace user {
  using namespace ntt;

  template <class M, Dimension D>
  struct InitFields {
    InitFields(real_t Bg, real_t drift) 
      : Bg { Bg }
      , drift { drift } 
      , drift_gamma { ONE / math::sqrt(ONE - SQR(drift)) } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return -math::sin(x_Ph[1]);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return math::sin(TWO * x_Ph[0]);
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return Bg;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      vec_t<Dim::_3D> v { ZERO };
      v[0] = - drift * math::sin(x_Ph[1]);
      v[1] = + drift * math::sin(x_Ph[0]);
      return -CROSS_x1(v[0], v[1], v[2], bx1(x_Ph), bx2(x_Ph), bx3(x_Ph));
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      vec_t<Dim::_3D> v { ZERO };
      v[0] = - drift * math::sin(x_Ph[1]);
      v[1] = + drift * math::sin(x_Ph[0]);
      return -CROSS_x2(v[0], v[1], v[2], bx1(x_Ph), bx2(x_Ph), bx3(x_Ph));
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      vec_t<Dim::_3D> v { ZERO };
      v[0] = - drift * math::sin(x_Ph[1]);
      v[1] = + drift * math::sin(x_Ph[0]);
      return -CROSS_x3(v[0], v[1], v[2], bx1(x_Ph), bx2(x_Ph), bx3(x_Ph));
    }

  private:
    const real_t Bg, drift, drift_gamma;
  };

  template <SimEngine::type S, class M>
  struct Drifting_background : public arch::EnergyDistribution<S, M> {
    Drifting_background(const M&                   metric,
                        random_number_pool_t&      pool,
                        const real_t               temperature,
                        const real_t               drift)
      : arch::EnergyDistribution<S, M> { metric }
      , metric { metric }
      , temperature { temperature }
      , drift {drift}
      , maxwellian { metric, pool, temperature } 
      { }

    Inline void operator()(const coord_t<M::Dim>& x_Cd,
                           vec_t<Dim::_3D>&       v,
                           unsigned short         sp) const override {
      coord_t<M::Dim> x_Ph { ZERO };
      metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
      
      vec_t<Dim::_3D> v_th; // first used as a helped function. then takes maxwellian distribution
      // counter-clock wise rotation of x-aligned vector + thermal velocity
      v_th[0] = - drift * math::sin(x_Ph[1]);
      v_th[1] = + drift * math::sin(x_Ph[0]);
      const real_t gamma = ONE / std::sqrt(ONE - NORM_SQR(v_th[0], v_th[1], ZERO));
      v_th[0] *= gamma;
      v_th[1] *= gamma;
      const real_t u  = NORM(v_th[0], v_th[1], ZERO);
      const real_t cos_h = v_th[0] / u;
      const real_t sin_h = v_th[1] / u;
      
      v_th[0] = ZERO;
      v_th[1] = u;
      v_th[2] = ZERO;
      maxwellian(x_Ph, v_th); // the coordinate is not used

      v[0] = cos_h * v_th[0] - sin_h * v_th[1];
      v[1] = sin_h * v_th[0] + cos_h * v_th[1];
      v[2] = v_th[2];

      // v[1] = ZERO;
      // maxwellian(x_Ph, v); // the coordinate is not used
      // const real_t vx = - drift * math::sin(x_Ph[1]);
      // const real_t vy = + drift * math::sin(x_Ph[0]);
      // const real_t gamma = ONE / std::sqrt(ONE - NORM_SQR(vx, vy, ZERO));
      // v[0] += gamma * vx;
      // v[1] += gamma * vy;
    }

  private:
    const M      metric;
    // random_number_pool_t &pool;
    const real_t temperature, drift;
    const arch::Maxwellian_coord<S, M> maxwellian;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t temperature, Bg, drift;

    InitFields<M, D> init_flds;

    inline PGen(const SimulationParams& params, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { params }
      , temperature { params.template get<real_t>("setup.temperature") }
      , drift { params.template get<real_t>("setup.drift") }
      , Bg { params.template get<real_t>("setup.Bg") }
      , init_flds { Bg, drift } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = Drifting_background<S, M>(local_domain.mesh.metric,
                                                         local_domain.random_pool,
                                                         temperature,
                                                         drift);
      const auto injector = arch::UniformInjector<S, M, Drifting_background>(energy_dist,
                                                                             { 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, Drifting_background>>(params,
                                                                                  local_domain,
                                                                                  injector,
                                                                                  1.0);
    }

    // void CustomFieldOutput(const std::string&    name,
    //                       ndfield_t<M::Dim, 6> buffer,
    //                       std::size_t          index,
    //                       const Domain<S, M>&  domain) {
    //   if (name == "Pressure") {
    //     if constexpr (M::Dim == Dim::_2D) {
    //       // kernel::ParticleMoments_kernel<S, M, FldsID::Rho, 3>
    //     }
    //   } else {
    //     raise::Error("Custom output not provided", HERE);
    //   }
    // }
  };

} // namespace user

#endif