/**
 * @file archetypes/spatial_dist.hpp
 * @brief Spatial distribution class passed to injectors
 * @implements
 *   - arch::SpatialDistribution<>
 *   - arch::Uniform<> : arch::SpatialDistribution<>
 *   - arch::Replenish<> : arch::SpatialDistribution<>
 *   - arch::ReplenishConst<> : arch::SpatialDistribution<>
 * @namespace
 *   - arch::
 * @note
 * Instances of these functors take coordinate position in code units
 * and return a number between 0 and 1 that represents the spatial distribution
 */

#ifndef ARCHETYPES_SPATIAL_DIST_HPP
#define ARCHETYPES_SPATIAL_DIST_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace arch {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct SpatialDistribution {
    static constexpr bool is_spatial_dist { true };
    static_assert(M::is_metric, "M must be a metric class");

    SpatialDistribution(const M& metric) : metric { metric } {}

  protected:
    const M metric;
  };

  template <SimEngine::type S, class M>
  struct Uniform : public SpatialDistribution<S, M> {
    Uniform(const M& metric) : SpatialDistribution<S, M> { metric } {}

    Inline auto operator()(const coord_t<M::Dim>&) const -> real_t {
      return ONE;
    }
  };

  template <SimEngine::type S, class M, class T, int N = 6>
  struct Replenish : public SpatialDistribution<S, M> {
    using SpatialDistribution<S, M>::metric;
    const ndfield_t<M::Dim, 6> density;
    const idx_t                idx;

    const T      target_density;
    const real_t target_max_density;

    const real_t fill_ratio;

    Replenish(const M&                    metric,
              const ndfield_t<M::Dim, N>& density,
              idx_t                       idx,
              const T&                    target_density,
              real_t                      target_max_density,
              real_t                      fill_ratio = 0.9)
      : SpatialDistribution<S, M> { metric }
      , density { density }
      , idx { idx }
      , target_density { target_density }
      , target_max_density { target_max_density }
      , fill_ratio { fill_ratio } {
      raise::ErrorIf(target_max_density <= ZERO,
                     "Target max density must be positive",
                     HERE);
      raise::ErrorIf(fill_ratio <= ZERO or fill_ratio > ONE,
                     "Fill ratio must be in (0, 1]",
                     HERE);
      raise::ErrorIf(idx >= N, "Index for density field is out of bounds", HERE);
    }

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      coord_t<M::Dim> x_Cd { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);
      real_t dens { ZERO };
      if constexpr (M::Dim == Dim::_1D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS, idx);
      } else if constexpr (M::Dim == Dim::_2D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       idx);
      } else if constexpr (M::Dim == Dim::_3D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[2]) + N_GHOSTS,
                       idx);
      } else {
        raise::KernelError(HERE, "Invalid dimension");
      }
      const auto target = target_density(x_Ph);
      if (fill_ratio * target > dens) {
        return (target - dens) / target_max_density;
      } else {
        return ZERO;
      }
    }
  };

  template <SimEngine::type S, class M, int N = 6>
  struct ReplenishConst : public SpatialDistribution<S, M> {
    using SpatialDistribution<S, M>::metric;
    const ndfield_t<M::Dim, N> density;
    const idx_t                idx;

    const real_t target_density;
    const real_t fill_ratio;

    ReplenishConst(const M&                    metric,
                   const ndfield_t<M::Dim, N>& density,
                   idx_t                       idx,
                   real_t                      target_density,
                   real_t                      fill_ratio = 0.9)
      : SpatialDistribution<S, M> { metric }
      , density { density }
      , idx { idx }
      , target_density { target_density }
      , fill_ratio { fill_ratio } {
      raise::ErrorIf(target_density <= ZERO, "Target density must be positive", HERE);
      raise::ErrorIf(fill_ratio <= ZERO or fill_ratio > ONE,
                     "Fill ratio must be in (0, 1]",
                     HERE);
      raise::ErrorIf(idx >= N, "Index for density field is out of bounds", HERE);
    }

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      coord_t<M::Dim> x_Cd { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);
      real_t dens { ZERO };
      if constexpr (M::Dim == Dim::_1D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS, idx);
      } else if constexpr (M::Dim == Dim::_2D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       idx);
      } else if constexpr (M::Dim == Dim::_3D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[2]) + N_GHOSTS,
                       idx);
      } else {
        raise::KernelError(HERE, "Invalid dimension");
      }
      if (fill_ratio * target_density > dens) {
        return (target_density - dens) / target_density;
      } else {
        return ZERO;
      }
    }
  };

} // namespace arch

#endif // ARCHETYPES_SPATIAL_DIST_HPP
