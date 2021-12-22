#ifndef OBJECTS_GEOMETRY_COORD_SYSTEM_H
#define OBJECTS_GEOMETRY_COORD_SYSTEM_H

#include "global.h"

#include <tuple>
#include <string>

namespace ntt {

  template <Dimension D>
  struct CoordinateSystem {
    real_t m_parameters[10];

    CoordinateSystem() = default;
    CoordinateSystem(const std::string& label) : m_label {label} {}
    [[nodiscard]] auto get_label() const -> std::string { return m_label; }
    virtual ~CoordinateSystem() = default;

    virtual Inline auto transform_x1TOx(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto transform_x1x2TOxy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
    virtual Inline auto transform_x1x2x3TOxyz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    virtual Inline auto transform_xTOx1(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto transform_xyTOx1x2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
    virtual Inline auto transform_xyzTOx1x2x3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    // velocity conversion
    virtual Inline auto transform_ux1TOux(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto transform_ux1ux2TOuxuy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
    virtual Inline auto transform_ux1ux2ux3TOuxuyuz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    virtual Inline auto transform_uxTOux1(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto transform_uxuyTOux1ux2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; };
    virtual Inline auto transform_uxuyuzTOux1ux2ux3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    virtual Inline auto hx1(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto hx1(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto hx1(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto hx2(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto hx2(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto hx2(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto hx3(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto hx3(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto hx3(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto sqrt_det_h(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto sqrt_det_h(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto sqrt_det_h(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto polar_area(const real_t&, const real_t&) const -> real_t { return -1.0; }

    // conversion to spherical
    virtual Inline auto getSpherical_r(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto getSpherical_r(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto getSpherical_r(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto getSpherical_theta(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto getSpherical_theta(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto getSpherical_theta(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto getSpherical_phi(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto getSpherical_phi(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto getSpherical_phi(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    // CNT -> contravariant (upper index)
    // CVR -> covariant (lower index)
    // LOC -> local orthonormal (hatted index)
    //
    // CNT -> LOC
    Inline auto convert_CNT_to_LOC_x1(const real_t& ax1, const real_t& x1) -> real_t {
      return std::sqrt(hx1(x1)) * ax1;
    }
    Inline auto convert_CNT_to_LOC_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
      return std::sqrt(hx1(x1, x2)) * ax1;
    }
    Inline auto convert_CNT_to_LOC_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return std::sqrt(hx1(x1, x2, x3)) * ax1;
    }

    Inline auto convert_CNT_to_LOC_x2(const real_t& ax2, const real_t& x1) -> real_t {
      return std::sqrt(hx2(x1)) * ax2;
    }
    Inline auto convert_CNT_to_LOC_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
      return std::sqrt(hx2(x1, x2)) * ax2;
    }
    Inline auto convert_CNT_to_LOC_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return std::sqrt(hx2(x1, x2, x3)) * ax2;
    }

    Inline auto convert_CNT_to_LOC_x3(const real_t& ax3, const real_t& x1) -> real_t {
      return std::sqrt(hx3(x1)) * ax3;
    }
    Inline auto convert_CNT_to_LOC_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
      return std::sqrt(hx3(x1, x2)) * ax3;
    }
    Inline auto convert_CNT_to_LOC_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return std::sqrt(hx3(x1, x2, x3)) * ax3;
    }

    // LOC -> CNT
    Inline auto convert_LOC_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t {
      return ax1 / std::sqrt(hx1(x1));
    }
    Inline auto convert_LOC_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
      return ax1 / std::sqrt(hx1(x1, x2));
    }
    Inline auto convert_LOC_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return ax1 / std::sqrt(hx1(x1, x2, x3));
    }

    Inline auto convert_LOC_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t {
      return ax2 / std::sqrt(hx2(x1));
    }
    Inline auto convert_LOC_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
      return ax2 / std::sqrt(hx2(x1, x2));
    }
    Inline auto convert_LOC_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return ax2 / std::sqrt(hx2(x1, x2, x3));
    }

    Inline auto convert_LOC_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t {
      return ax3 / std::sqrt(hx3(x1));
    }
    Inline auto convert_LOC_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
      return ax3 / std::sqrt(hx3(x1, x2));
    }
    Inline auto convert_LOC_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return ax3 / std::sqrt(hx3(x1, x2, x3));
    }

  protected:
    std::string m_label;
  };

} // namespace ntt

#endif
