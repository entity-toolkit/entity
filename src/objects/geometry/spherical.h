#ifndef OBJECTS_GEOMETRY_SPHERICAL_H
#define OBJECTS_GEOMETRY_SPHERICAL_H

#include "global.h"
#include "coord_system.h"

#include <tuple>
#include <cassert>

namespace ntt {

  template <Dimension D>
  struct SphericalSystem : public CoordinateSystem<D> {
    SphericalSystem() : CoordinateSystem<D> {"spherical"} {}
    ~SphericalSystem() = default;

    // coordinate transformations
    // ... curvilinear -> cartesian
    Inline auto transform_x1TOx(const real_t&) const -> real_t override {
      assert(false);
      return -1.0;
    }
    Inline auto transform_x1x2TOxy(const real_t& x1, const real_t& x2) const -> std::tuple<real_t, real_t> override {
      return {x1, x2};
    }
    Inline auto transform_x1x2x3TOxyz(const real_t& x1, const real_t& x2, const real_t& x3) const -> std::tuple<real_t, real_t, real_t> override {
      return {x1, x2, x3};
    }

    // ... cartesian -> curvilinear
    Inline auto transform_xTOx1(const real_t&) const -> real_t override {
      assert(false);
      return -1.0;
    }
    Inline auto transform_xyTOx1x2(const real_t& x, const real_t& y) const -> std::tuple<real_t, real_t> override {
      return {x, y};
    }
    Inline auto transform_xyzTOx1x2x3(const real_t& x, const real_t& y, const real_t& z) const -> std::tuple<real_t, real_t, real_t> override {
      return {x, y, z};
    }

    // vector transformations
    // ... curvilinear -> cartesian
    Inline auto transform_ux1TOux(const real_t&) const -> real_t override {
      assert(false);
      return -1.0;
    }
    Inline auto transform_ux1ux2TOuxuy(const real_t& ux1, const real_t& ux2) const -> std::tuple<real_t, real_t> override {
      return {ux1, ux2};
    }
    Inline auto transform_ux1ux2ux3TOuxuyuz(const real_t& ux1, const real_t& ux2, const real_t& ux3) const -> std::tuple<real_t, real_t, real_t> override {
      return {ux1, ux2, ux3};
    }

    // ... cartesian -> curvilinear
    Inline auto transform_uxTOux1(const real_t&) const -> real_t override {
      assert(false);
      return -1.0;
    }
    Inline auto transform_uxuyTOux1ux2(const real_t& ux, const real_t& uy) const -> std::tuple<real_t, real_t> override {
      return {ux, uy};
    }
    Inline auto transform_uxuyuzTOux1ux2ux3(const real_t& ux, const real_t& uy, const real_t& uz) const -> std::tuple<real_t, real_t, real_t> override {
      return {ux, uy, uz};
    }

    // metric coefficients
    Inline auto hx1(const real_t&) const -> real_t {
      assert(false);
      return -1.0;
    }
    Inline auto hx1(const real_t&, const real_t&) const -> real_t {
      return 1.0;
    }
    Inline auto hx1(const real_t&, const real_t&, const real_t&) const -> real_t {
      return 1.0;
    }

    Inline auto hx2(const real_t&) const -> real_t {
      assert(false);
      return -1.0;
    }
    Inline auto hx2(const real_t& x1, const real_t&) const -> real_t {
      return x1 * x1;
    }
    Inline auto hx2(const real_t& x1, const real_t&, const real_t&) const -> real_t {
      return x1 * x1;
    }

    Inline auto hx3(const real_t&) const -> real_t {
      assert(false);
      return -1.0;
    }
    Inline auto hx3(const real_t& x1, const real_t& x2) const -> real_t {
      real_t sin_x2 {std::sin(x2)};
      return x1 * x1 * sin_x2 * sin_x2;
    }
    Inline auto hx3(const real_t& x1, const real_t& x2, const real_t&) const -> real_t {
      real_t sin_x2 {std::sin(x2)};
      return x1 * x1 * sin_x2 * sin_x2;
    }

    // det of metric
    Inline auto sqrt_det_h(const real_t&) const -> real_t {
      assert(false);
      return -1.0;
    }
    Inline auto sqrt_det_h(const real_t& x1, const real_t& x2) const -> real_t {
      return x1 * x1 * std::sin(x2);
    }
    Inline auto sqrt_det_h(const real_t& x1, const real_t& x2, const real_t&) const -> real_t {
      return x1 * x1 * std::sin(x2);
    }

    // area at poles
    Inline auto polar_area(const real_t& x1, const real_t& dx2) const -> real_t {
      return x1 * x1 * (ONE - std::cos(dx2 * 0.5));
    }
  };

} // namespace ntt

#endif
