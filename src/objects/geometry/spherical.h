#ifndef OBJECTS_GEOMETRY_SPHERICAL_H
#define OBJECTS_GEOMETRY_SPHERICAL_H

#include "global.h"
#include "grid.h"

#include <tuple>
#include <cassert>

namespace ntt {

  // r, theta, phi
  template <Dimension D>
  struct SphericalSystem : public Grid<D> {
  protected:
    const real_t dr, dtheta, dphi;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;

  public:
    SphericalSystem(std::vector<std::size_t> resolution, std::vector<real_t> extent)
      : Grid<D> {"spherical", resolution, extent},
        dr((this->x1_max - this->x1_min) / (real_t)(this->Nx1)),
        dtheta(PI / (real_t)(this->Nx2)),
        dphi(TWO_PI / (real_t)(this->Nx3)),
        dr_sqr(dr * dr),
        dtheta_sqr(dtheta * dtheta),
        dphi_sqr(dphi * dphi) {}
    ~SphericalSystem() = default;

    auto findSmallestCell() const -> real_t {
      if constexpr (D == TWO_D) {
        auto dx1 {dr};
        auto dx2 {this->x1_min * dtheta};
        return ONE / std::sqrt(ONE / (dx1 * dx1) + ONE / (dx2 * dx2));
      } else {
        throw std::logic_error("# Error: min cell finding not implemented for 3D spherical.");
      }
    }

    // * * * * * * * * * * * * * * *
    // 2D:
    // * * * * * * * * * * * * * * *
    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    Inline auto coord_CU_to_Cart(const real_t& x1, const real_t& x2) const -> std::tuple<real_t, real_t> override {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta};
      return {r * std::sin(theta), r * std::cos(theta)};
    }

    // // conversion from cartesian (Cart) to code units (CU)
    // Inline auto coord_CART_to_CU(const real_t& x, const real_t& y) const -> std::tuple<real_t, real_t> override {
    //   return {x, y};
    // }

    // conversion to spherical
    Inline auto coord_CU_to_Sph(const real_t& x1, const real_t& x2) const -> std::tuple<real_t, real_t> {
      return {x1 * dr + this->x1_min, x2 * dtheta};
    }

    // metric coefficients
    Inline auto h11(const real_t&, const real_t&) const -> real_t { return dr_sqr; }

    Inline auto h22(const real_t& x1, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      return dtheta_sqr * r * r;
    }

    Inline auto h33(const real_t& x1, const real_t& x2) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta};
      auto sin_theta {std::sin(theta)};
      return r * r * sin_theta * sin_theta;
    }

    // det of metric
    Inline auto sqrt_det_h(const real_t& x1, const real_t& x2) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta};
      return dr * dtheta * r * r * std::sin(theta);
    }

    // area at poles
    Inline auto polar_area(const real_t& x1, const real_t& x2) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto del_theta {x2 * dtheta};
      return r * r * (ONE - std::cos(del_theta));
    }

    // * * * * * * * * * * * * * * *
    // 3D:
    // * * * * * * * * * * * * * * *
    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    Inline auto coord_CU_to_Cart(const real_t& x1, const real_t& x2, const real_t& x3) const
      -> std::tuple<real_t, real_t, real_t> override {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta};
      auto phi {x3 * dphi};
      return {r * std::sin(theta) * std::cos(phi), r * std::sin(theta) * std::sin(phi), r * std::cos(theta)};
    }

    // // conversion from cartesian (Cart) to code units (CU)
    // Inline auto coord_CART_to_CU(const real_t& x, const real_t& y, const real_t& z) const
    //     -> std::tuple<real_t, real_t, real_t> override {
    //   return {x, y, z};
    // }

    // conversion to spherical
    Inline auto coord_CU_to_Sph(const real_t& x1, const real_t& x2, const real_t& x3) const
      -> std::tuple<real_t, real_t, real_t> {
      return {x1 * dr + this->x1_min, x2 * dtheta, x3 * dphi};
    }

    // metric coefficients
    Inline auto h11(const real_t&, const real_t&, const real_t&) const -> real_t { return dr_sqr; }
    Inline auto h22(const real_t& x1, const real_t&, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      return dtheta_sqr * r * r;
    }
    Inline auto h33(const real_t& x1, const real_t& x2, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta};
      auto sin_theta {std::sin(theta)};
      return dphi_sqr * r * r * sin_theta * sin_theta;
    }

    // det of metric
    Inline auto sqrt_det_h(const real_t& x1, const real_t& x2, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto theta {x2 * dtheta};
      return dr * dtheta * dphi * r * r * std::sin(theta);
    }

    // area at poles
    Inline auto polar_area(const real_t& x1, const real_t& x2, const real_t&) const -> real_t {
      auto r {x1 * dr + this->x1_min};
      auto del_theta {x2 * dtheta};
      return r * r * (ONE - std::cos(del_theta));
    }
  };
} // namespace ntt

#endif
