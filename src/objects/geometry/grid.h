#ifndef OBJECTS_GEOMETRY_GRID_H
#define OBJECTS_GEOMETRY_GRID_H

#include "global.h"

#include <tuple>
#include <string>

namespace ntt {

  template <Dimension D>
  struct CoordinateGrid {
  protected:
    std::string m_label;
    std::vector<std::size_t> m_resolution;
    std::vector<real_t> m_extent_PHU;

  public:
    // real_t m_parameters[10];

    CoordinateGrid(const std::string& label, std::vector<std::size_t> resolution, std::vector<real_t> extent)
        : m_label {label}, m_resolution {resolution}, m_extent_PHU {extent} {}
    virtual ~CoordinateGrid() = default;

    // getters
    Inline auto label() const -> std::string { 
      return m_label;
    }
    Inline auto x1min_PHU() const -> real_t { 
      return m_extent_PHU[0];
    }
    Inline auto x1max_PHU() const -> real_t { 
      return m_extent_PHU[1];
    }
    Inline auto x2min_PHU() const -> real_t { 
      return m_extent_PHU[2];
    }
    Inline auto x2max_PHU() const -> real_t { 
      return m_extent_PHU[3];
    }
    Inline auto x3min_PHU() const -> real_t { 
      return m_extent_PHU[4];
    }
    Inline auto x3max_PHU() const -> real_t { 
      return m_extent_PHU[5];
    }

    // Inline auto Nx1() const -> std::size_t { 
    //   return m_resolution[0];
    // }
    // Inline auto Nx2() const -> std::size_t { 
    //   return m_resolution[1];
    // }
    // Inline auto Nx3() const -> std::size_t { 
    //   return m_resolution[2];
    // }

    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    virtual Inline auto coord_CU_to_Cart(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto coord_CU_to_Cart(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> {
      return {-1.0, -1.0}; }
    virtual Inline auto coord_CU_to_Cart(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    // conversion from cartesian (Cart) to code units (CU)
    virtual Inline auto coord_CART_to_CU(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto coord_CART_to_CU(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
    virtual Inline auto coord_CART_to_CU(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    // conversion from code units (CU) to spherical (Sph)
    virtual Inline auto coord_CU_to_Sph(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
    virtual Inline auto coord_CU_to_Sph(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    Inline auto CU_to_Idi(const real_t& xi) const -> std::pair<long int, float> {
      // TODO: this is a hack
      auto i {static_cast<long int>(xi + N_GHOSTS)};
      float di {static_cast<float>(xi) - static_cast<float>(i)};
      return {i, di};
    }

    // // velocity conversion
    // virtual Inline auto transform_ux1TOux(const real_t&) const -> real_t { return -1.0; }
    // virtual Inline auto transform_ux1ux2TOuxuy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
    // virtual Inline auto transform_ux1ux2ux3TOuxuyuz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    // virtual Inline auto transform_uxTOux1(const real_t&) const -> real_t { return -1.0; }
    // virtual Inline auto transform_uxuyTOux1ux2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; };
    // virtual Inline auto transform_uxuyuzTOux1ux2ux3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

    virtual Inline auto h11(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto h11(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto h11(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto h22(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto h22(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto h22(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto h33(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto h33(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto h33(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto sqrt_det_h(const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto sqrt_det_h(const real_t&, const real_t&) const -> real_t { return -1.0; }
    virtual Inline auto sqrt_det_h(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    virtual Inline auto polar_area(const real_t&, const real_t&) const -> real_t { return -1.0; }

    // CNT -> contravariant (upper index)
    // CVR -> covariant (lower index)
    // HAT -> local orthonormal (hatted index)
    //
    // CNT -> HAT
    Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1) -> real_t {
      return std::sqrt(h11(x1)) * ax1;
    }
    Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
      return std::sqrt(h11(x1, x2)) * ax1;
    }
    Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return std::sqrt(h11(x1, x2, x3)) * ax1;
    }

    Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1) -> real_t {
      return std::sqrt(h22(x1)) * ax2;
    }
    Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
      return std::sqrt(h22(x1, x2)) * ax2;
    }
    Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return std::sqrt(h22(x1, x2, x3)) * ax2;
    }

    Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1) -> real_t {
      return std::sqrt(h33(x1)) * ax3;
    }
    Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
      return std::sqrt(h33(x1, x2)) * ax3;
    }
    Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return std::sqrt(h33(x1, x2, x3)) * ax3;
    }

    // LOC -> CNT
    Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t {
      return ax1 / std::sqrt(h11(x1));
    }
    Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
      return ax1 / std::sqrt(h11(x1, x2));
    }
    Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return ax1 / std::sqrt(h11(x1, x2, x3));
    }

    Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t {
      return ax2 / std::sqrt(h22(x1));
    }
    Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
      return ax2 / std::sqrt(h22(x1, x2));
    }
    Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return ax2 / std::sqrt(h22(x1, x2, x3));
    }

    Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t {
      return ax3 / std::sqrt(h33(x1));
    }
    Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
      return ax3 / std::sqrt(h33(x1, x2));
    }
    Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
      return ax3 / std::sqrt(h33(x1, x2, x3));
    }
  };

} // namespace ntt

#endif
