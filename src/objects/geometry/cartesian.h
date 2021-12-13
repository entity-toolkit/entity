#ifndef OBJECTS_GEOMETRY_CARTESIAN_H
#define OBJECTS_GEOMETRY_CARTESIAN_H

#include "global.h"
#include "coord_system.h"

namespace ntt {

template <Dimension D>
struct CartesianSystem : public CoordinateSystem<D> {
  CartesianSystem() : CoordinateSystem<D>{"cartesian"} {}
  ~CartesianSystem() = default;

  Inline auto convert_x1TOx(const real_t&) const -> real_t override;
  Inline auto convert_x1x2TOxy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> override;
  Inline auto convert_x1x2x3TOxyz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> override;

  Inline auto convert_xTOx1(const real_t&) const -> real_t override;
  Inline auto convert_xyTOx1x2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> override;
  Inline auto convert_xyzTOx1x2x3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> override;

  // velocity conversion
  Inline auto convert_ux1TOux(const real_t&) const -> real_t override;
  Inline auto convert_ux1ux2TOuxuy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> override;
  Inline auto convert_ux1ux2ux3TOuxuyuz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> override;

  Inline auto convert_uxTOux1(const real_t&) const -> real_t override;
  Inline auto convert_uxuyTOux1ux2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> override;
  Inline auto convert_uxuyuzTOux1ux2ux3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> override;
};

}


#endif
