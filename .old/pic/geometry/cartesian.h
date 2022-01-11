#ifndef OBJECTS_GEOMETRY_CARTESIAN_H
#define OBJECTS_GEOMETRY_CARTESIAN_H

#include "global.h"
#include "grid.h"

#include <tuple>

namespace ntt {
  // in cartesian system dx = dy = dz always
  template <Dimension D>
  struct CartesianSystem : public Grid<D> {
  protected:
    const real_t dx, dx_sqr, inv_dx;

  public:
    CartesianSystem(std::vector<std::size_t> resolution, std::vector<real_t> extent)
        : Grid<D> {"cartesian", resolution, extent},
          dx((this->x1_max - this->x1_min) / (real_t)(this->Nx1)),
          dx_sqr(dx * dx),
          inv_dx(ONE / dx) {}
    ~CartesianSystem() = default;

    auto findSmallestCell() const -> real_t {
      return dx / std::sqrt(static_cast<real_t>(D));
    }

    // * * * * * * * * * * * * * * *
    // 1D:
    // * * * * * * * * * * * * * * *
    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    Inline auto coord_CU_to_Cart(const real_t &x1) const -> real_t override {
      return x1 * dx + this->x1_min;
    }
    Inline auto h11(const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto h22(const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto h33(const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto sqrt_det_h(const real_t&) const -> real_t {
      return dx * dx * dx;
    }
    // conversion to global cartesian basis
    Inline auto vec_LOC_to_HAT(const real_t& ax, const real_t& ay, const real_t& az, const real_t&)
      -> std::tuple<real_t, real_t, real_t> {
      return {ax, ay, az};
    }
    Inline auto vec_HAT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t&)
      -> std::tuple<real_t, real_t, real_t> {
      return {ax1, ax2, ax3};
    }

    // * * * * * * * * * * * * * * *
    // 2D:
    // * * * * * * * * * * * * * * *
    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    Inline auto coord_CU_to_Cart(const real_t &x1, const real_t &x2) const -> std::tuple<real_t, real_t> override {
      return {x1 * dx + this->x1_min, x2 * dx + this->x2_min};
    }
    Inline auto h11(const real_t&, const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto h22(const real_t&, const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto h33(const real_t&, const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto sqrt_det_h(const real_t&, const real_t&) const -> real_t {
      return dx * dx * dx;
    }
    // conversion to global cartesian basis
    Inline auto vec_LOC_to_HAT(const real_t& ax, const real_t& ay, const real_t& az, const real_t&, const real_t&)
      -> std::tuple<real_t, real_t, real_t> {
      return {ax, ay, az};
    }
    Inline auto vec_HAT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t&, const real_t&)
      -> std::tuple<real_t, real_t, real_t> {
      return {ax1, ax2, ax3};
    }

    // * * * * * * * * * * * * * * *
    // 3D:
    // * * * * * * * * * * * * * * *
    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    Inline auto coord_CU_to_Cart(const real_t &x1, const real_t &x2, const real_t &x3) const -> std::tuple<real_t, real_t, real_t> override {
      return {x1 * dx + this->x1_min, x2 * dx + this->x2_min, x3 * dx + this->x3_min};
    }
    Inline auto h11(const real_t&, const real_t&, const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto h22(const real_t&, const real_t&, const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto h33(const real_t&, const real_t&, const real_t&) const -> real_t {
      return dx_sqr;
    }
    Inline auto sqrt_det_h(const real_t&, const real_t&, const real_t&) const -> real_t {
      return dx * dx * dx;
    }
    // conversion to global cartesian basis
    Inline auto
    vec_LOC_to_HAT(const real_t& ax, const real_t& ay, const real_t& az, const real_t&, const real_t&, const real_t&)
      -> std::tuple<real_t, real_t, real_t> {
      return {ax, ay, az};
    }
    Inline auto
    vec_HAT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t&, const real_t&, const real_t&)
      -> std::tuple<real_t, real_t, real_t> {
      return {ax1, ax2, ax3};
    }
  };

} // namespace ntt

#endif
