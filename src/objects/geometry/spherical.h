#ifndef OBJECTS_GEOMETRY_SPHERICAL_H
#define OBJECTS_GEOMETRY_SPHERICAL_H

#include "global.h"
#include "grid.h"

#include <tuple>
#include <cassert>

namespace ntt {

  template <Dimension D>
  struct SphericalSystem : public CoordinateGrid<D> {
  protected:
    const real_t dr, dtheta, dphi;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;

  public:
    SphericalSystem(std::vector<std::size_t> resolution, std::vector<real_t> extent)
      : CoordinateGrid<D> {"spherical", resolution, extent},
        dr((this->x1_max - this->x1_min) / (real_t)(this->Nx1)),
        dtheta((this->x2_max - this->x2_min) / (real_t)(this->Nx2)),
        dphi((this->x3_max - this->x3_min) / (real_t)(this->Nx3)),
        dr_sqr(dr * dr),
        dtheta_sqr(dtheta * dtheta),
        dphi_sqr(dphi * dphi) {}
    ~SphericalSystem() = default;

    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    // Inline auto coord_CU_to_Cart(const real_t& x1) const -> real_t override { return x1 * dx + this->x1_min; }
    Inline auto coord_CU_to_Cart(const real_t& x1, const real_t& x2) const -> std::tuple<real_t, real_t> override {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta + this->x2_min};
      return {r * std::sin(theta), r * std::cos(theta)};
    }
    Inline auto coord_CU_to_Cart(const real_t& x1, const real_t& x2, const real_t& x3) const
      -> std::tuple<real_t, real_t, real_t> override {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta + this->x2_min};
      auto phi {x3 * dphi + this->x3_min};
      return {r * std::sin(theta) * std::cos(phi), r * std::sin(theta) * std::sin(phi), r * std::cos(theta)};
    }

    // // conversion from cartesian (Cart) to code units (CU)
    // Inline auto coord_CART_to_CU(const real_t&) const -> real_t override {
    //   assert(false);
    //   return -1.0;
    // }
    // Inline auto coord_CART_to_CU(const real_t& x, const real_t& y) const -> std::tuple<real_t, real_t> override {
    //   return {x, y};
    // }
    // Inline auto coord_CART_to_CU(const real_t& x, const real_t& y, const real_t& z) const
    //     -> std::tuple<real_t, real_t, real_t> override {
    //   return {x, y, z};
    // }

    // conversion to spherical
    Inline auto coord_CU_to_Sph(const real_t& x1, const real_t& x2) const -> std::tuple<real_t, real_t> {
      return {x1 * dr + this->x1_min, x2 * dtheta + this->x2_min};
    }
    Inline auto coord_CU_to_Sph(const real_t& x1, const real_t& x2, const real_t& x3) const
      -> std::tuple<real_t, real_t, real_t> {
      return {x1 * dr + this->x1_min, x2 * dtheta + this->x2_min, x3 * dphi + this->x3_min};
    }

    // // vector transformations
    // // ... curvilinear -> cartesian
    // Inline auto transform_ux1TOux(const real_t&) const -> real_t override {
    //   assert(false);
    //   return -1.0;
    // }
    // Inline auto transform_ux1ux2TOuxuy(const real_t& ux1, const real_t& ux2) const -> std::tuple<real_t, real_t>
    // override {
    //   return {ux1, ux2};
    // }
    // Inline auto transform_ux1ux2ux3TOuxuyuz(const real_t& ux1, const real_t& ux2, const real_t& ux3) const ->
    // std::tuple<real_t, real_t, real_t> override {
    //   return {ux1, ux2, ux3};
    // }

    // // ... cartesian -> curvilinear
    // Inline auto transform_uxTOux1(const real_t&) const -> real_t override {
    //   assert(false);
    //   return -1.0;
    // }
    // Inline auto transform_uxuyTOux1ux2(const real_t& ux, const real_t& uy) const -> std::tuple<real_t, real_t>
    // override {
    //   return {ux, uy};
    // }
    // Inline auto transform_uxuyuzTOux1ux2ux3(const real_t& ux, const real_t& uy, const real_t& uz) const ->
    // std::tuple<real_t, real_t, real_t> override {
    //   return {ux, uy, uz};
    // }

    // metric coefficients
    // Inline auto h11(const real_t&) const -> real_t {
    //   assert(false);
    //   return -1.0;
    // }
    Inline auto h11(const real_t&, const real_t&) const -> real_t { return dr_sqr; }
    Inline auto h11(const real_t&, const real_t&, const real_t&) const -> real_t { return dr_sqr; }

    // Inline auto h22(const real_t&) const -> real_t {
    //   assert(false);
    //   return -1.0;
    // }
    Inline auto h22(const real_t& x1, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      return dtheta_sqr * r * r;
    }
    Inline auto h22(const real_t& x1, const real_t&, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      return dtheta_sqr * r * r;
    }

    // Inline auto h33(const real_t&) const -> real_t {
    //   assert(false);
    //   return -1.0;
    // }
    Inline auto h33(const real_t& x1, const real_t& x2) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta + this->x2_min};
      real_t sin_theta {std::sin(theta)};
      return r * r * sin_theta * sin_theta;
    }
    Inline auto h33(const real_t& x1, const real_t& x2, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta + this->x2_min};
      auto sin_theta {std::sin(theta)};
      return dphi_sqr * r * r * sin_theta * sin_theta;
    }

    // det of metric
    // Inline auto sqrt_det_h(const real_t&) const -> real_t {
    //   assert(false);
    //   return -1.0;
    // }
    Inline auto sqrt_det_h(const real_t& x1, const real_t& x2) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta + this->x2_min};
      return dr * dtheta * r * r * std::sin(theta);
    }
    Inline auto sqrt_det_h(const real_t& x1, const real_t& x2, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta + this->x2_min};
      return dr * dtheta * dphi * r * r * std::sin(theta);
    }

    // area at poles
    Inline auto polar_area(const real_t& x1, const real_t& x2) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto del_theta {x2 * dtheta + this->x2_min};
      return r * r * (ONE - std::cos(del_theta));
    }
    Inline auto polar_area(const real_t& x1, const real_t& x2, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto del_theta {x2 * dtheta + this->x2_min};
      return dphi * r * r * (ONE - std::cos(del_theta));
    }
  };
} // namespace ntt

#endif
