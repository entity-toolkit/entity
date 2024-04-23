/**
 * @file archetypes/spatial_dist.hpp
 * @brief Spatial distribution class passed to injectors
 * @implements
 *   - arch::SpatialDistribution<>
 *   - arch::UniformDist<> : arch::SpatialDistribution<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/kokkos_aliases.h
 *   - utils/numeric.h
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
#include "utils/numeric.h"

namespace arch {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct SpatialDistribution {
    static constexpr bool is_spatial_dist { true };
    static_assert(M::is_metric, "M must be a metric class");

    SpatialDistribution(const M& metric) : metric { metric } {}

    Inline virtual auto operator()(const coord_t<M::Dim>&) const -> real_t {
      return ONE;
    }

  protected:
    const M metric;
  };

  template <SimEngine::type S, class M>
  struct UniformDist : public SpatialDistribution<S, M> {
    UniformDist(const M& metric) : SpatialDistribution<S, M> { metric } {}

    Inline auto operator()(const coord_t<M::Dim>&) const -> real_t override {
      return ONE;
    }
  };

} // namespace arch

#endif // ARCHETYPES_SPATIAL_DIST_HPP