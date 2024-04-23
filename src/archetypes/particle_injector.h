/**
 * @file archetypes/particle_injector.h
 * @brief Particle injector routines
 * ...
 */

#ifndef ARCHETYPES_PARTICLE_INJECTOR_H
#define ARCHETYPES_PARTICLE_INJECTOR_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "framework/domain/domain.h"

#include <Kokkos_Core.hpp>

#include <tuple>
#include <utility>
#include <vector>

namespace arch {
  using namespace ntt;
  using spidx_t = unsigned short;

  template <SimEngine::type S, class M, template <SimEngine::type, class> class E>
  struct ParticleInjector {
    using energy_dist_t = E<S, M>;
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(energy_dist_t::is_energy_dist,
                  "E must be an energy distribution class");
    static constexpr bool      is_injector { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    const energy_dist_t               energy_dist;
    const std::pair<spidx_t, spidx_t> species;

    ParticleInjector(const energy_dist_t&               energy_dist,
                     const std::pair<spidx_t, spidx_t>& species)
      : energy_dist { energy_dist }
      , species { species } {}

    ~ParticleInjector() = default;
  };

  template <SimEngine::type S, class M, class ED>
  struct Injector_kernel {
    static_assert(ED::is_energy_dist, "ED must be an energy distribution class");
    static_assert(M::is_metric, "M must be a metric class");

    const spidx_t spidx1, spidx2;

    array_t<int*>      i1s_1, i2s_1, i3s_1;
    array_t<prtldx_t*> dx1s_1, dx2s_1, dx3s_1;
    array_t<real_t*>   ux1s_1, ux2s_1, ux3s_1;
    array_t<real_t*>   phis_1;
    array_t<real_t*>   weights_1;
    array_t<short*>    tags_1;

    array_t<int*>         i1s_2, i2s_2, i3s_2;
    array_t<prtldx_t*>    dx1s_2, dx2s_2, dx3s_2;
    array_t<real_t*>      ux1s_2, ux2s_2, ux3s_2;
    array_t<real_t*>      phis_2;
    array_t<real_t*>      weights_2;
    const array_t<short*> tags_2;

    std::size_t            offset1, offset2;
    const M                metric;
    const array_t<real_t*> ni;
    const ED               energy_dist;
    const real_t           inv_V0;
    random_number_pool_t   random_pool;

    Injector_kernel(spidx_t                          spidx1,
                    spidx_t                          spidx2,
                    Particles<M::Dim, M::CoordType>& species1,
                    Particles<M::Dim, M::CoordType>& species2,
                    std::size_t                      offset1,
                    std::size_t                      offset2,
                    const M&                         metric,
                    const array_t<real_t*>&          ni,
                    const ED&                        energy_dist,
                    real_t                           inv_V0,
                    random_number_pool_t&            random_pool)
      : spidx1 { spidx1 }
      , spidx2 { spidx2 }
      , i1s_1 { species1.i1 }
      , i2s_1 { species1.i2 }
      , i3s_1 { species1.i3 }
      , dx1s_1 { species1.dx1 }
      , dx2s_1 { species1.dx2 }
      , dx3s_1 { species1.dx3 }
      , ux1s_1 { species1.ux1 }
      , ux2s_1 { species1.ux2 }
      , ux3s_1 { species1.ux3 }
      , phis_1 { species1.phi }
      , weights_1 { species1.weight }
      , tags_1 { species1.tag }
      , i1s_2 { species2.i1 }
      , i2s_2 { species2.i2 }
      , i3s_2 { species2.i3 }
      , dx1s_2 { species2.dx1 }
      , dx2s_2 { species2.dx2 }
      , dx3s_2 { species2.dx3 }
      , ux1s_2 { species2.ux1 }
      , ux2s_2 { species2.ux2 }
      , ux3s_2 { species2.ux3 }
      , phis_2 { species2.phi }
      , weights_2 { species2.weight }
      , tags_2 { species2.tag }
      , offset1 { offset1 }
      , offset2 { offset2 }
      , metric { metric }
      , ni { ni }
      , energy_dist { energy_dist }
      , inv_V0 { inv_V0 }
      , random_pool { random_pool } {}

    Inline void operator()(index_t p) const {
      coord_t<M::Dim> x_Cd { ZERO };
      vec_t<Dim::_3D> v1 { ZERO }, v2 { ZERO };
      { // generate a random coordinate
        auto rand_gen = random_pool.get_state();
        x_Cd[0]       = Random<real_t>(rand_gen) * ni(0);
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          x_Cd[1] = Random<real_t>(rand_gen) * ni(1);
        }
        if constexpr (M::Dim == Dim::_3D) {
          x_Cd[2] = Random<real_t>(rand_gen) * ni(2);
        }
        random_pool.free_state(rand_gen);
      }
      { // generate the velocity
        coord_t<M::Dim> x_Ph { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
        if constexpr (M::CoordType == Coord::Cart) {
          coord_t<M::PrtlDim> x_Ph_ { ZERO };
          vec_t<Dim::_3D>     v_Ph { ZERO };
          energy_dist(x_Ph, v_Ph, spidx1);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Ph_, v_Ph, v1);
          energy_dist(x_Ph, v_Ph, spidx2);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Ph, v_Ph, v2);
        } else if constexpr (S == SimEngine::SRPIC) {
          coord_t<M::PrtlDim> x_Ph_ { ZERO };
          x_Ph_[0] = x_Ph[0];
          x_Ph_[1] = x_Ph[1];
          x_Ph_[2] = ZERO; // phi = 0
          vec_t<Dim::_3D> v_Ph { ZERO };
          energy_dist(x_Ph, v_Ph, spidx1);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Ph_, v_Ph, v1);
          energy_dist(x_Ph, v_Ph, spidx2);
          metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Ph_, v_Ph, v2);
        } else if constexpr (S == SimEngine::GRPIC) {
          vec_t<Dim::_3D> v_Ph { ZERO };
          energy_dist(x_Ph, v_Ph, spidx1);
          metric.template transform<Idx::T, Idx::D>(x_Ph, v_Ph, v1);
          energy_dist(x_Ph, v_Ph, spidx2);
          metric.template transform<Idx::T, Idx::D>(x_Ph, v_Ph, v2);
        } else {
          raise::KernelError(HERE, "Unknown simulation engine");
        }
      }
      // inject
      i1s_1(p + offset1)  = static_cast<int>(x_Cd[0]);
      dx1s_1(p + offset1) = static_cast<prtldx_t>(
        x_Cd[0] - static_cast<real_t>(i1s_1(p + offset1)));
      i1s_2(p + offset2)  = static_cast<int>(x_Cd[0]);
      dx1s_2(p + offset2) = static_cast<prtldx_t>(
        x_Cd[0] - static_cast<real_t>(i1s_2(p + offset2)));
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        i2s_1(p + offset1)  = static_cast<int>(x_Cd[1]);
        dx2s_1(p + offset1) = static_cast<prtldx_t>(
          x_Cd[1] - static_cast<real_t>(i2s_1(p + offset1)));
        i2s_2(p + offset2)  = static_cast<int>(x_Cd[1]);
        dx2s_2(p + offset2) = static_cast<prtldx_t>(
          x_Cd[1] - static_cast<real_t>(i2s_2(p + offset2)));
        if constexpr (S == SimEngine::SRPIC && M::CoordType != Coord::Cart) {
          phis_1(p + offset1) = ZERO;
          phis_2(p + offset2) = ZERO;
        }
      }
      if constexpr (M::Dim == Dim::_3D) {
        i3s_1(p + offset1)  = static_cast<int>(x_Cd[2]);
        dx3s_1(p + offset1) = static_cast<prtldx_t>(
          x_Cd[2] - static_cast<real_t>(i3s_1(p + offset1)));
        i3s_2(p + offset2)  = static_cast<int>(x_Cd[2]);
        dx3s_2(p + offset2) = static_cast<prtldx_t>(
          x_Cd[2] - static_cast<real_t>(i3s_2(p + offset2)));
      }
      ux1s_1(p + offset1) = v1[0];
      ux2s_1(p + offset1) = v1[1];
      ux3s_1(p + offset1) = v1[2];
      ux1s_2(p + offset2) = v2[0];
      ux2s_2(p + offset2) = v2[1];
      ux3s_2(p + offset2) = v2[2];
      tags_1(p + offset1) = ParticleTag::alive;
      tags_2(p + offset2) = ParticleTag::alive;
      if constexpr (M::CoordType == Coord::Cart) {
        weights_1(p + offset1) = ONE;
        weights_2(p + offset2) = ONE;
      } else {
        const auto sqrt_det_h  = metric.sqrt_det_h(x_Cd);
        weights_1(p + offset1) = sqrt_det_h * inv_V0;
        weights_2(p + offset2) = sqrt_det_h * inv_V0;
      }
    }
  };

  /**
   * @brief Injects uniform number density of particles everywhere in the domain
   * @param domain Domain object
   * @param injector Particle injector object
   * @param number_density Total number density (in units of n0)
   * @param use_weights Use weights
   * @tparam S Simulation engine type
   * @tparam M Metric type
   * @tparam I Injector type
   */
  template <SimEngine::type S, class M, class I>
  inline void InjectUniformNumberDensity(const SimulationParams& params,
                                         Domain<S, M>&           domain,
                                         const I&                injector,
                                         real_t                  number_density,
                                         bool use_weights = false) {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(I::is_injector, "I must be a particle injector class");
    raise::ErrorIf((M::CoordType != Coord::Cart) && (not use_weights),
                   "Weights must be used for non-Cartesian coordinates",
                   HERE);
    raise::ErrorIf((M::CoordType == Coord::Cart) && use_weights,
                   "Weights should not be used for Cartesian coordinates",
                   HERE);
    raise::ErrorIf(params.template get<bool>("particles.use_weights") != use_weights,
                   "Weights must be enabled from the input file to use them in "
                   "the injector",
                   HERE);
    if (domain.species[injector.species.first - 1].charge() +
          domain.species[injector.species.second - 1].charge() !=
        0.0f) {
      raise::Warning("Total charge of the injected species is non-zero", HERE);
    }

    {
      auto             ppc0 = params.template get<real_t>("particles.ppc0");
      array_t<real_t*> ni { "ni", M::Dim };
      auto             ni_h   = Kokkos::create_mirror_view(ni);
      std::size_t      ncells = 1;
      for (auto d = 0; d < M::Dim; ++d) {
        ni_h(d)  = domain.mesh.n_active()[d];
        ncells  *= domain.mesh.n_active()[d];
      }
      Kokkos::deep_copy(ni, ni_h);
      const auto nparticles = static_cast<std::size_t>(
        (long double)(ppc0 * number_density * 0.5) * (long double)(ncells));

      Kokkos::parallel_for("InjectUniformNumberDensity",
                           nparticles,
                           Injector_kernel<S, M, typename I::energy_dist_t>(
                             injector.species.first,
                             injector.species.second,
                             domain.species[injector.species.first - 1],
                             domain.species[injector.species.second - 1],
                             domain.species[injector.species.first - 1].npart(),
                             domain.species[injector.species.second - 1].npart(),
                             domain.mesh.metric,
                             ni,
                             injector.energy_dist,
                             ONE / params.template get<real_t>("scales.V0"),
                             domain.random_pool));
      domain.species[injector.species.first - 1].set_npart(
        domain.species[injector.species.first - 1].npart() + nparticles);
      domain.species[injector.species.second - 1].set_npart(
        domain.species[injector.species.second - 1].npart() + nparticles);
    }
  }

} // namespace arch

#endif // ARCHETYPES_PARTICLE_INJECTOR_H