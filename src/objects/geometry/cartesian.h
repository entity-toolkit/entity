#ifndef OBJECTS_GEOMETRY_CARTESIAN_H
#define OBJECTS_GEOMETRY_CARTESIAN_H

#include "global.h"
#include "grid.h"

#include <tuple>

namespace ntt {
  // in cartesian system dx = dy = dz always
  template <Dimension D>
  struct CartesianSystem : public CoordinateGrid<D> {
  protected:
    const real_t dx, inv_dx;
    const real_t x_min, y_min, z_min;

  public:
    CartesianSystem(std::vector<std::size_t> resolution, std::vector<real_t> extent)
        : CoordinateGrid<D> {"cartesian", resolution, extent},
          dx((extent[1] - extent[0]) / (real_t)(resolution[0])),
          inv_dx(ONE / dx),
          x_min(resolution.size() > 0 ? extent[0] : ZERO),
          y_min(resolution.size() > 1 ? extent[2] : ZERO),
          z_min(resolution.size() > 2 ? extent[4] : ZERO) {}
    ~CartesianSystem() = default;

    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    Inline auto coord_CU_to_Cart(const real_t &x1) const -> real_t override { 
      return x1 * dx + x_min;
    }
    Inline auto coord_CU_to_Cart(const real_t &x1, const real_t &x2) const -> std::tuple<real_t, real_t> override {
      return {x1 * dx + x_min, x2 * dx + y_min};
    }
    Inline auto coord_CU_to_Cart(const real_t &x1, const real_t &x2, const real_t &x3) const -> std::tuple<real_t, real_t, real_t> override { 
      return {x1 * dx + x_min, x2 * dx + y_min, x3 * dx + z_min};
    }

    // conversion from cartesian (Cart) to code units (CU)
    Inline auto coord_CART_to_CU(const real_t &x) const -> real_t override { 
      return (x - x_min) * inv_dx;
    }
    Inline auto coord_CART_to_CU(const real_t &x, const real_t &y) const -> std::tuple<real_t, real_t> override {
      return {(x - x_min) * inv_dx, (y - y_min) * inv_dx};
    }
    Inline auto coord_CART_to_CU(const real_t &x, const real_t &y, const real_t &z) const -> std::tuple<real_t, real_t, real_t> override {
      return {(x - x_min) * inv_dx, (y - y_min) * inv_dx, (z - z_min) * inv_dx};
    }

    // // vector transformations
    // // ... curvilinear -> cartesian
    // Inline auto transform_ux1TOux(const real_t& ux1) const -> real_t override { return ux1; }
    // Inline auto transform_ux1ux2TOuxuy(const real_t& ux1, const real_t& ux2) const -> std::tuple<real_t, real_t> override { return {ux1, ux2}; }
    // Inline auto transform_ux1ux2ux3TOuxuyuz(const real_t& ux1, const real_t& ux2, const real_t& ux3) const -> std::tuple<real_t, real_t, real_t> override { return {ux1, ux2, ux3}; }

    // // ... cartesian -> curvilinear
    // Inline auto transform_uxTOux1(const real_t& ux) const -> real_t override { return ux; }
    // Inline auto transform_uxuyTOux1ux2(const real_t& ux, const real_t& uy) const -> std::tuple<real_t, real_t> override { return {ux, uy}; }
    // Inline auto transform_uxuyuzTOux1ux2ux3(const real_t& ux, const real_t& uy, const real_t& uz) const -> std::tuple<real_t, real_t, real_t> override { return {ux, uy, uz}; }

    Inline auto h11(const real_t &) const -> real_t { 
      return dx;
    }
    Inline auto h11(const real_t &, const real_t &) const -> real_t { 
      return dx;
    }
    Inline auto h11(const real_t &, const real_t &, const real_t &) const -> real_t { 
      return dx;
    }

    Inline auto h22(const real_t&) const -> real_t { 
      return dx;
    }
    Inline auto h22(const real_t&, const real_t&) const -> real_t { 
      return dx;
    }
    Inline auto h22(const real_t&, const real_t&, const real_t&) const -> real_t { 
      return dx;
    }

    Inline auto h33(const real_t&) const -> real_t { 
      return dx;
    }
    Inline auto h33(const real_t&, const real_t&) const -> real_t { 
      return dx;
    }
    Inline auto h33(const real_t&, const real_t&, const real_t&) const -> real_t { 
      return dx;
    }

    Inline auto sqrt_det_h(const real_t&) const -> real_t { 
      return dx * dx * dx;
    }
    Inline auto sqrt_det_h(const real_t&, const real_t&) const -> real_t { 
      return dx * dx * dx;
    }
    Inline auto sqrt_det_h(const real_t&, const real_t&, const real_t&) const -> real_t { 
      return dx * dx * dx;
    }

    Inline auto polar_area(const real_t&, const real_t&) const -> real_t {
      assert(false);
      return -1.0;
    }

    // // conversion to spherical
    // // TODO: not implemented yet
    // Inline auto getSpherical_r(const real_t&) const -> real_t { return -1.0; }
    // Inline auto getSpherical_r(const real_t&, const real_t&) const -> real_t { return -1.0; }
    // Inline auto getSpherical_r(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    // Inline auto getSpherical_theta(const real_t&) const -> real_t { return -1.0; }
    // Inline auto getSpherical_theta(const real_t&, const real_t&) const -> real_t { return -1.0; }
    // Inline auto getSpherical_theta(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    // Inline auto getSpherical_phi(const real_t&) const -> real_t { return -1.0; }
    // Inline auto getSpherical_phi(const real_t&, const real_t&) const -> real_t { return -1.0; }
    // Inline auto getSpherical_phi(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }
  };

} // namespace ntt

#endif
