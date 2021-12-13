#ifndef OBJECTS_GEOMETRY_CARTESIAN_H
#define OBJECTS_GEOMETRY_CARTESIAN_H

#include "global.h"
#include "coord_system.h"

namespace ntt {

template <Dimension D>
struct CartesianSystem : public CoordinateSystem<D> {
  CartesianSystem() : CoordinateSystem<D>{"cartesian"} {}
  ~CartesianSystem() = default;

  // coordinate transformations
  // ... curvilinear -> cartesian
  Inline auto transform_x1TOx(const real_t&) const -> real_t override;
  Inline auto transform_x1x2TOxy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> override;
  Inline auto transform_x1x2x3TOxyz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> override;

  // ... cartesian -> curvilinear
  Inline auto transform_xTOx1(const real_t&) const -> real_t override;
  Inline auto transform_xyTOx1x2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> override;
  Inline auto transform_xyzTOx1x2x3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> override;

  // vector transformations
  // ... curvilinear -> cartesian
  Inline auto transform_ux1TOux(const real_t&) const -> real_t override;
  Inline auto transform_ux1ux2TOuxuy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> override;
  Inline auto transform_ux1ux2ux3TOuxuyuz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> override;

  // ... cartesian -> curvilinear
  Inline auto transform_uxTOux1(const real_t&) const -> real_t override;
  Inline auto transform_uxuyTOux1ux2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> override;
  Inline auto transform_uxuyuzTOux1ux2ux3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> override;
};

}


#endif
