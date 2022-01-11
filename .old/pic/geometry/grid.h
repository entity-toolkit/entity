#ifndef OBJECTS_GEOMETRY_GRID_H
#  define OBJECTS_GEOMETRY_GRID_H

#  include "global.h"

#  include <tuple>
#  include <string>

namespace ntt {
  /**
   * Arbitrary coordinate grid with a diagonal h_ij.
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  struct Grid {
    const std::string label;
    const std::size_t Nx1, Nx2, Nx3;
    const real_t x1_min, x1_max;
    const real_t x2_min, x2_max;
    const real_t x3_min, x3_max;

    Grid(const std::string& label_, std::vector<std::size_t> resolution, std::vector<real_t> extent)
      : label {label_},
        Nx1 {resolution.size() > 0 ? resolution[0] : 1},
        Nx2 {resolution.size() > 1 ? resolution[1] : 1},
        Nx3 {resolution.size() > 2 ? resolution[2] : 1},
        x1_min {resolution.size() > 0 ? extent[0] : ZERO},
        x1_max {resolution.size() > 0 ? extent[1] : ZERO},
        x2_min {resolution.size() > 1 ? extent[2] : ZERO},
        x2_max {resolution.size() > 1 ? extent[3] : ZERO},
        x3_min {resolution.size() > 2 ? extent[4] : ZERO},
        x3_max {resolution.size() > 2 ? extent[5] : ZERO} {}
    virtual ~Grid() = default;

    /**
     * Compute minimum effective cell size for a given coordinate grid (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    virtual auto findSmallestCell() const -> real_t { return -1.0; }

    /**
     * Convert `real_t` type code unit coordinate to cell index + displacement.
     *
     * @returns A pair of `long int` and `float`: cell index + displacement.
     */
    Inline auto CU_to_Idi(const real_t& xi) const -> std::pair<long int, float> {
      // TODO: this is a hack
      auto i {static_cast<long int>(xi + N_GHOSTS)};
      float di {static_cast<float>(xi) - static_cast<float>(i)};
      return {i, di};
    }

    /**
     * Convert 1d coordinate from code units to global Cartesian frame.
     *
     * @param x1 [code units].
     * @returns x coordinate [physical units].
     */
    virtual Inline auto coord_CU_to_Cart(const real_t&) const -> real_t { return -1.0; }

    /**
     * Convert 2d coordinate from code units to global Cartesian frame.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @returns x and y coordinates [physical units].
     */
    virtual Inline auto coord_CU_to_Cart(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> {
      return {-1.0, -1.0};
    }

    /**
     * Convert 3d coordinate from code units to global Cartesian frame.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @param x3 [code units].
     * @returns x, y and z coordinates [physical units].
     */
    virtual Inline auto coord_CU_to_Cart(const real_t&, const real_t&, const real_t&) const
      -> std::tuple<real_t, real_t, real_t> {
      return {-1.0, -1.0, -1.0};
    }

    /**
     * Convert 2d coordinate from code units to global spherical frame.
     *
     * @param x1 [code units].
     * @param x2 [code units].
     * @returns r and theta coordinates [physical units].
     */
    virtual Inline auto coord_CU_to_Sph(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> {
      return {-1.0, -1.0};
    }

    /**
     * Convert 3d coordinate from code units to global spherical frame.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @param x3 [code units].
     * @returns r, theta and phi coordinates [physical units].
     */
    virtual Inline auto coord_CU_to_Sph(const real_t&, const real_t&, const real_t&) const
      -> std::tuple<real_t, real_t, real_t> {
      return {-1.0, -1.0, -1.0};
    }

    /**
     * Compute metric components.
     *
     * @param x1 [code units].
     * @returns h_11 (covariant) metric component.
     */
    virtual Inline auto h11(const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute metric components.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @returns h_11 (covariant) metric component.
     */
    virtual Inline auto h11(const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute metric components.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @param x3 [code units].
     * @returns h_11 (covariant) metric component.
     */
    virtual Inline auto h11(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute metric components.
     *
     * @overload
     * @param x1 [code units].
     * @returns h_22 (covariant) metric component.
     */
    virtual Inline auto h22(const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute metric components.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @returns h_22 (covariant) metric component.
     */
    virtual Inline auto h22(const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute metric components.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @param x3 [code units].
     * @returns h_22 (covariant) metric component.
     */
    virtual Inline auto h22(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute metric components.
     *
     * @overload
     * @param x1 [code units].
     * @returns h_33 (covariant) metric component.
     */
    virtual Inline auto h33(const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute metric components.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @returns h_33 (covariant) metric component.
     */
    virtual Inline auto h33(const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute metric components.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @param x3 [code units].
     * @returns h_33 (covariant) metric component.
     */
    virtual Inline auto h33(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x1 [code units].
     * @returns sqrt(det(h_ij)).
     */
    virtual Inline auto sqrt_det_h(const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @returns sqrt(det(h_ij)).
     */
    virtual Inline auto sqrt_det_h(const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @param x3 [code units].
     * @returns sqrt(det(h_ij)).
     */
    virtual Inline auto sqrt_det_h(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute the area at the pole (used for axisymmetric solvers).
     *
     * @param x1 [code units].
     * @param x2 [code units].
     * @returns Area at the pole.
     */
    virtual Inline auto polar_area(const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Compute the area at the pole (used for axisymmetric solvers).
     *
     * @overload
     * @param x1 [code units].
     * @param x2 [code units].
     * @param x3 [code units].
     * @returns Area at the pole.
     */
    virtual Inline auto polar_area(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

    /**
     * Comment on notations.
     *
     * CNT -> contravariant (upper index)
     * CVR -> covariant (lower index)
     * HAT -> local orthonormal (hatted index)
     * LOC -> global cartesian
     *
     */

    /**
     * Convert x1-component of the vector from contravariant (CNT) to covariant (CVR).
     *
     * @param ax1 x1-component of the contravariant vector.
     * @param x1 [code units].
     * @returns x1-component of the covariant vector.
     */
    Inline auto vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1) -> real_t;

    // /**
    //  * Convert x1-component of the vector from contravariant (CNT) to covariant (CVR).
    //  *
    //  * @overload
    //  * @param ax1 x1-component of the contravariant vector.
    //  * @param x1 [code units].
    //  * @param x2 [code units].
    //  * @returns x1-component of the covariant vector.
    //  */
    // Inline auto vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t;

    // /**
    //  * Convert x1-component of the vector from contravariant (CNT) to covariant (CVR).
    //  *
    //  * @overload
    //  * @param ax1 x1-component of the contravariant vector.
    //  * @param x1 [code units].
    //  * @param x2 [code units].
    //  * @param x3 [code units].
    //  * @returns x1-component of the covariant vector.
    //  */
    // Inline auto vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    /**
     * Convert x1-component of the vector from contravariant (CNT) to covariant (CVR).
     *
     * @overload
     * @param ax1 x1-component of the contravariant vector.
     * @param x1 [code units].
     * @param x2 [code units].
     * @param x3 [code units].
     * @returns x1-component of the covariant vector.
     */
    Inline auto vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1) -> real_t;
    Inline auto vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    Inline auto vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1) -> real_t;
    Inline auto vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    // CVR -> CNT:
    Inline auto vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t;
    Inline auto vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    Inline auto vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t;
    Inline auto vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    Inline auto vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t;
    Inline auto vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    // CNT -> HAT:
    Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1) -> real_t;
    Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1) -> real_t;
    Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1) -> real_t;
    Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    // HAT -> CNT:
    Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t;
    Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t;
    Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t;
    Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t;
    Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

    // CNT -> LOC:
    Inline auto vec_CNT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1)
      -> std::tuple<real_t, real_t, real_t>;
    Inline auto
    vec_CNT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2)
      -> std::tuple<real_t, real_t, real_t>;
    Inline auto vec_CNT_to_LOC(
      const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
      -> std::tuple<real_t, real_t, real_t>;

    // LOC -> CNT:
    Inline auto vec_LOC_to_CNT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1)
      -> std::tuple<real_t, real_t, real_t>;
    Inline auto vec_LOC_to_CNT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2)
      -> std::tuple<real_t, real_t, real_t>;
    Inline auto vec_LOC_to_CNT(
      const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2, const real_t& x3)
      -> std::tuple<real_t, real_t, real_t>;

    // grid-specific
    // HAT -> LOC (virtual):
    virtual Inline auto vec_HAT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1)
      -> std::tuple<real_t, real_t, real_t> {
      return {-1.0, -1.0, -1.0};
    }
    virtual Inline auto
    vec_HAT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2)
      -> std::tuple<real_t, real_t, real_t> {
      return {-1.0, -1.0, -1.0};
    }
    virtual Inline auto vec_HAT_to_LOC(
      const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
      -> std::tuple<real_t, real_t, real_t> {
      return {-1.0, -1.0, -1.0};
    }

    // LOC -> HAT (virtual):
    virtual Inline auto vec_LOC_to_HAT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1)
      -> std::tuple<real_t, real_t, real_t> {
      return {-1.0, -1.0, -1.0};
    }
    virtual Inline auto
    vec_LOC_to_HAT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2)
      -> std::tuple<real_t, real_t, real_t> {
      return {-1.0, -1.0, -1.0};
    }
    virtual Inline auto vec_LOC_to_HAT(
      const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2, const real_t& x3)
      -> std::tuple<real_t, real_t, real_t> {
      return {-1.0, -1.0, -1.0};
    }
  };

  // CNT -> HAT:
  template <>
  Inline auto Grid<ONE_D>::vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1) -> real_t {
    return std::sqrt(h11(x1)) * ax1;
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
    return std::sqrt(h11(x1, x2)) * ax1;
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return std::sqrt(h11(x1, x2, x3)) * ax1;
  }

  template <>
  Inline auto Grid<ONE_D>::vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1) -> real_t {
    return std::sqrt(h22(x1)) * ax2;
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
    return std::sqrt(h22(x1, x2)) * ax2;
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return std::sqrt(h22(x1, x2, x3)) * ax2;
  }

  template <>
  Inline auto Grid<ONE_D>::vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1) -> real_t {
    return std::sqrt(h33(x1)) * ax3;
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
    return std::sqrt(h33(x1, x2)) * ax3;
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return std::sqrt(h33(x1, x2, x3)) * ax3;
  }

  // CNT -> CVR:
  template <>
  Inline auto Grid<ONE_D>::vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1) -> real_t {
    return ax1 * h11(x1);
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
    return ax1 * h11(x1, x2);
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax1 * h11(x1, x2, x3);
  }

  template <>
  Inline auto Grid<ONE_D>::vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1) -> real_t {
    return ax2 * h22(x1);
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
    return ax2 * h22(x1, x2);
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax2 * h22(x1, x2, x3);
  }

  template <>
  Inline auto Grid<ONE_D>::vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1) -> real_t {
    return ax3 * h33(x1);
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
    return ax3 * h33(x1, x2);
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax3 * h33(x1, x2, x3);
  }

  // CVR -> CNT:
  template <>
  Inline auto Grid<ONE_D>::vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t {
    return ax1 / h11(x1);
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
    return ax1 / h11(x1, x2);
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax1 / h11(x1, x2, x3);
  }

  template <>
  Inline auto Grid<ONE_D>::vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t {
    return ax2 / h22(x1);
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
    return ax2 / h22(x1, x2);
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax2 / h22(x1, x2, x3);
  }

  template <>
  Inline auto Grid<ONE_D>::vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t {
    return ax3 / h33(x1);
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
    return ax3 / h33(x1, x2);
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax3 / h33(x1, x2, x3);
  }

  // LOC -> CNT:
  template <>
  Inline auto Grid<ONE_D>::vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t {
    return ax1 / std::sqrt(h11(x1));
  }
  template <>
  Inline auto Grid<TWO_D>::vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
    return ax1 / std::sqrt(h11(x1, x2));
  }
  template <>
  Inline auto Grid<THREE_D>::vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax1 / std::sqrt(h11(x1, x2, x3));
  }

  template <>
  Inline auto Grid<ONE_D>::vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t {
    return ax2 / std::sqrt(h22(x1));
  }
  template <>
  Inline auto Grid<TWO_D>::vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
    return ax2 / std::sqrt(h22(x1, x2));
  }
  template <>
  Inline auto Grid<THREE_D>::vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax2 / std::sqrt(h22(x1, x2, x3));
  }

  template <>
  Inline auto Grid<ONE_D>::vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t {
    return ax3 / std::sqrt(h33(x1));
  }
  template <>
  Inline auto Grid<TWO_D>::vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
    return ax3 / std::sqrt(h33(x1, x2));
  }
  template <>
  Inline auto Grid<THREE_D>::vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
    -> real_t {
    return ax3 / std::sqrt(h33(x1, x2, x3));
  }

  // CNT -> LOC:
  template <>
  Inline auto Grid<ONE_D>::vec_CNT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1)
    -> std::tuple<real_t, real_t, real_t> {
    auto ax1_hat {vec_CNT_to_HAT_x1(ax1, x1)};
    auto ax2_hat {vec_CNT_to_HAT_x2(ax2, x1)};
    auto ax3_hat {vec_CNT_to_HAT_x3(ax3, x1)};
    return vec_HAT_to_LOC(ax1_hat, ax2_hat, ax3_hat, x1);
  }
  template <>
  Inline auto Grid<TWO_D>::vec_CNT_to_LOC(
    const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2)
    -> std::tuple<real_t, real_t, real_t> {
    auto ax1_hat {vec_CNT_to_HAT_x1(ax1, x1, x2)};
    auto ax2_hat {vec_CNT_to_HAT_x2(ax2, x1, x2)};
    auto ax3_hat {vec_CNT_to_HAT_x3(ax3, x1, x2)};
    return vec_HAT_to_LOC(ax1_hat, ax2_hat, ax3_hat, x1, x2);
  }
  template <>
  Inline auto Grid<THREE_D>::vec_CNT_to_LOC(
    const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
    -> std::tuple<real_t, real_t, real_t> {
    auto ax1_hat {vec_CNT_to_HAT_x1(ax1, x1, x2, x3)};
    auto ax2_hat {vec_CNT_to_HAT_x2(ax2, x1, x2, x3)};
    auto ax3_hat {vec_CNT_to_HAT_x3(ax3, x1, x2, x3)};
    return vec_HAT_to_LOC(ax1_hat, ax2_hat, ax3_hat, x1, x2, x3);
  }

  // LOC -> CNT:
  template <>
  Inline auto Grid<ONE_D>::vec_LOC_to_CNT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1)
    -> std::tuple<real_t, real_t, real_t> {
    auto [ax1_hat, ax2_hat, ax3_hat] = vec_LOC_to_HAT(ax, ay, az, x1);
    auto ax1 {vec_HAT_to_CNT_x1(ax1_hat, x1)};
    auto ax2 {vec_HAT_to_CNT_x2(ax2_hat, x1)};
    auto ax3 {vec_HAT_to_CNT_x3(ax3_hat, x1)};
    return {ax1, ax2, ax3};
  }
  template <>
  Inline auto
  Grid<TWO_D>::vec_LOC_to_CNT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2)
    -> std::tuple<real_t, real_t, real_t> {
    auto [ax1_hat, ax2_hat, ax3_hat] = vec_LOC_to_HAT(ax, ay, az, x1, x2);
    auto ax1 {vec_HAT_to_CNT_x1(ax1_hat, x1, x2)};
    auto ax2 {vec_HAT_to_CNT_x2(ax2_hat, x1, x2)};
    auto ax3 {vec_HAT_to_CNT_x3(ax3_hat, x1, x2)};
    return {ax1, ax2, ax3};
  }
  template <>
  Inline auto Grid<THREE_D>::vec_LOC_to_CNT(
    const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2, const real_t& x3)
    -> std::tuple<real_t, real_t, real_t> {
    auto [ax1_hat, ax2_hat, ax3_hat] = vec_LOC_to_HAT(ax, ay, az, x1, x2, x3);
    auto ax1 {vec_HAT_to_CNT_x1(ax1_hat, x1, x2, x3)};
    auto ax2 {vec_HAT_to_CNT_x2(ax2_hat, x1, x2, x3)};
    auto ax3 {vec_HAT_to_CNT_x3(ax3_hat, x1, x2, x3)};
    return {ax1, ax2, ax3};
  }
} // namespace ntt

#endif

// #ifndef OBJECTS_GEOMETRY_GRID_H
// #define OBJECTS_GEOMETRY_GRID_H

// #include "global.h"

// #include <tuple>
// #include <string>

// namespace ntt {
//   /**
//    * Arbitrary coordinate grid with a diagonal h_ij.
//    *
//    * @tparam D dimension.
//    */
//   template <Dimension D>
//   struct Grid {
//     const std::string label;
//     const std::size_t Nx1, Nx2, Nx3;
//     const real_t x1_min, x1_max;
//     const real_t x2_min, x2_max;
//     const real_t x3_min, x3_max;

//     Grid(const std::string& label_, std::vector<std::size_t> resolution, std::vector<real_t> extent)
//       : label {label_},
//         Nx1 {resolution.size() > 0 ? resolution[0] : 1},
//         Nx2 {resolution.size() > 1 ? resolution[1] : 1},
//         Nx3 {resolution.size() > 2 ? resolution[2] : 1},
//         x1_min {resolution.size() > 0 ? extent[0] : ZERO},
//         x1_max {resolution.size() > 0 ? extent[1] : ZERO},
//         x2_min {resolution.size() > 1 ? extent[2] : ZERO},
//         x2_max {resolution.size() > 1 ? extent[3] : ZERO},
//         x3_min {resolution.size() > 2 ? extent[4] : ZERO},
//         x3_max {resolution.size() > 2 ? extent[5] : ZERO} {}
//     virtual ~Grid() = default;

//     /**
//      * Compute minimum effective cell size for a given coordinate grid (in physical units).
//      *
//      * @returns Minimum cell size of the grid [physical units].
//      */
//     virtual auto findSmallestCell() const -> real_t { return -1.0; }

//     /**
//      * Convert `real_t` type code unit coordinate to cell index + displacement.
//      *
//      * @returns A pair of `long int` and `float`: cell index + displacement.
//      */
//     Inline auto CU_to_Idi(const real_t& xi) const -> std::pair<long int, float> {
//       // TODO: this is a hack
//       auto i {static_cast<long int>(xi + N_GHOSTS)};
//       float di {static_cast<float>(xi) - static_cast<float>(i)};
//       return {i, di};
//     }

//     /**
//      * Convert 1d coordinate from code units to global Cartesian frame.
//      *
//      * @param x1 [code units].
//      * @returns x coordinate [physical units].
//      */
//     virtual Inline auto coord_CU_to_Cart(const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Convert 2d coordinate from code units to global Cartesian frame.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @returns x and y coordinates [physical units].
//      */
//     virtual Inline auto coord_CU_to_Cart(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> {
//       return {-1.0, -1.0};
//     }

//     /**
//      * Convert 3d coordinate from code units to global Cartesian frame.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns x, y and z coordinates [physical units].
//      */
//     virtual Inline auto coord_CU_to_Cart(const real_t&, const real_t&, const real_t&) const
//       -> std::tuple<real_t, real_t, real_t> {
//       return {-1.0, -1.0, -1.0};
//     }

//     /**
//      * Convert 2d coordinate from code units to global spherical frame.
//      *
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @returns r and theta coordinates [physical units].
//      */
//     virtual Inline auto coord_CU_to_Sph(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> {
//       return {-1.0, -1.0};
//     }

//     /**
//      * Convert 3d coordinate from code units to global spherical frame.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns r, theta and phi coordinates [physical units].
//      */
//     virtual Inline auto coord_CU_to_Sph(const real_t&, const real_t&, const real_t&) const
//       -> std::tuple<real_t, real_t, real_t> {
//       return {-1.0, -1.0, -1.0};
//     }

//     /**
//      * Compute metric components.
//      *
//      * @param x1 [code units].
//      * @returns h_11 (covariant) metric component.
//      */
//     virtual Inline auto h11(const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute metric components.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @returns h_11 (covariant) metric component.
//      */
//     virtual Inline auto h11(const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute metric components.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns h_11 (covariant) metric component.
//      */
//     virtual Inline auto h11(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute metric components.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @returns h_22 (covariant) metric component.
//      */
//     virtual Inline auto h22(const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute metric components.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @returns h_22 (covariant) metric component.
//      */
//     virtual Inline auto h22(const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute metric components.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns h_22 (covariant) metric component.
//      */
//     virtual Inline auto h22(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute metric components.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @returns h_33 (covariant) metric component.
//      */
//     virtual Inline auto h33(const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute metric components.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @returns h_33 (covariant) metric component.
//      */
//     virtual Inline auto h33(const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute metric components.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns h_33 (covariant) metric component.
//      */
//     virtual Inline auto h33(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute the square root of the determinant of h-matrix.
//      *
//      * @param x1 [code units].
//      * @returns sqrt(det(h_ij)).
//      */
//     virtual Inline auto sqrt_det_h(const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute the square root of the determinant of h-matrix.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @returns sqrt(det(h_ij)).
//      */
//     virtual Inline auto sqrt_det_h(const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute the square root of the determinant of h-matrix.
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns sqrt(det(h_ij)).
//      */
//     virtual Inline auto sqrt_det_h(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute the area at the pole (used for axisymmetric solvers).
//      *
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @returns Area at the pole.
//      */
//     virtual Inline auto polar_area(const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Compute the area at the pole (used for axisymmetric solvers).
//      *
//      * @overload
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns Area at the pole.
//      */
//     virtual Inline auto polar_area(const real_t&, const real_t&, const real_t&) const -> real_t { return -1.0; }

//     /**
//      * Comment on notations.
//      *
//      * CNT -> contravariant (upper index)
//      * CVR -> covariant (lower index)
//      * HAT -> local orthonormal (hatted index)
//      * LOC -> global cartesian
//      * 
//      */

//     /**
//      * Convert x1-component of the vector from contravariant (CNT) to covariant (CVR).
//      *
//      * @param ax1 x1-component of the contravariant vector.
//      * @param x1 [code units].
//      * @returns x1-component of the covariant vector.
//      */
//     Inline auto vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1) -> real_t;

//     /**
//      * Convert x1-component of the vector from contravariant (CNT) to covariant (CVR).
//      *
//      * @overload
//      * @param ax1 x1-component of the contravariant vector.
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @returns x1-component of the covariant vector.
//      */
//     Inline auto vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t;

//     /**
//      * Convert x1-component of the vector from contravariant (CNT) to covariant (CVR).
//      *
//      * @overload
//      * @param ax1 x1-component of the contravariant vector.
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns x1-component of the covariant vector.
//      */
//     Inline auto vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     /**
//      * Convert x1-component of the vector from contravariant (CNT) to covariant (CVR).
//      *
//      * @overload
//      * @param ax1 x1-component of the contravariant vector.
//      * @param x1 [code units].
//      * @param x2 [code units].
//      * @param x3 [code units].
//      * @returns x1-component of the covariant vector.
//      */
//     Inline auto vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1) -> real_t;
//     Inline auto vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     Inline auto vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1) -> real_t;
//     Inline auto vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     // CVR -> CNT:
//     Inline auto vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t;
//     Inline auto vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     Inline auto vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t;
//     Inline auto vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     Inline auto vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t;
//     Inline auto vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     // CNT -> HAT:
//     Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1) -> real_t;
//     Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1) -> real_t;
//     Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1) -> real_t;
//     Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     // HAT -> CNT:
//     Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t;
//     Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t;
//     Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t;
//     Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t;
//     Inline auto vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t;

//     // CNT -> LOC:
//     Inline auto vec_CNT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1)
//       -> std::tuple<real_t, real_t, real_t>;
//     Inline auto vec_CNT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2)
//       -> std::tuple<real_t, real_t, real_t>;
//     Inline auto vec_CNT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
//       -> std::tuple<real_t, real_t, real_t>;

//     // LOC -> CNT:
//     Inline auto vec_LOC_to_CNT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1)
//       -> std::tuple<real_t, real_t, real_t>;
//     Inline auto vec_LOC_to_CNT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2)
//       -> std::tuple<real_t, real_t, real_t>;
//     Inline auto vec_LOC_to_CNT(
//       const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2, const real_t& x3)
//       -> std::tuple<real_t, real_t, real_t>;

//     // grid-specific
//     // HAT -> LOC (virtual):
//     virtual Inline auto vec_HAT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1)
//       -> std::tuple<real_t, real_t, real_t> {
//       return {-1.0, -1.0, -1.0};
//     }
//     virtual Inline auto
//     vec_HAT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2)
//       -> std::tuple<real_t, real_t, real_t> {
//       return {-1.0, -1.0, -1.0};
//     }
//     virtual Inline auto vec_HAT_to_LOC(
//       const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
//       -> std::tuple<real_t, real_t, real_t> {
//       return {-1.0, -1.0, -1.0};
//     }

//     // LOC -> HAT (virtual):
//     virtual Inline auto vec_LOC_to_HAT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1)
//       -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }
//     virtual Inline auto
//     vec_LOC_to_HAT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2)
//       -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }
//     virtual Inline auto vec_LOC_to_HAT(
//       const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2, const real_t& x3)
//       -> std::tuple<real_t, real_t, real_t> {
//       return {-1.0, -1.0, -1.0};
//     }
//   };

//   // CNT -> HAT:
//   template <>
//   Inline auto Grid<ONE_D>::vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1) -> real_t {
//     return std::sqrt(h11(x1)) * ax1;
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
//     return std::sqrt(h11(x1, x2)) * ax1;
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CNT_to_HAT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return std::sqrt(h11(x1, x2, x3)) * ax1;
//   }

//   template <>
//   Inline auto Grid<ONE_D>::vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1) -> real_t {
//     return std::sqrt(h22(x1)) * ax2;
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
//     return std::sqrt(h22(x1, x2)) * ax2;
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CNT_to_HAT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return std::sqrt(h22(x1, x2, x3)) * ax2;
//   }

//   template <>
//   Inline auto Grid<ONE_D>::vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1) -> real_t {
//     return std::sqrt(h33(x1)) * ax3;
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
//     return std::sqrt(h33(x1, x2)) * ax3;
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CNT_to_HAT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return std::sqrt(h33(x1, x2, x3)) * ax3;
//   }

//   // CNT -> CVR:
//   template <>
//   Inline auto Grid<ONE_D>::vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1) -> real_t {
//     return ax1 * h11(x1);
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
//     return ax1 * h11(x1, x2);
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CNT_to_CVR_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return ax1 * h11(x1, x2, x3);
//   }

//   template <>
//   Inline auto Grid<ONE_D>::vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1) -> real_t {
//     return ax2 * h22(x1);
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
//     return ax2 * h22(x1, x2);
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CNT_to_CVR_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return ax2 * h22(x1, x2, x3);
//   }

//   template <>
//   Inline auto Grid<ONE_D>::vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1) -> real_t {
//     return ax3 * h33(x1);
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
//     return ax3 * h33(x1, x2);
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CNT_to_CVR_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3) -> real_t {
//     return ax3 * h33(x1, x2, x3);
//   }

//   // CVR -> CNT:
//   template <>
//   Inline auto Grid<ONE_D>::vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t {
//     return ax1 / h11(x1);
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
//     return ax1 / h11(x1, x2);
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CVR_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return ax1 / h11(x1, x2, x3);
//   }

//   template <>
//   Inline auto Grid<ONE_D>::vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t {
//     return ax2 / h22(x1);
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
//     return ax2 / h22(x1, x2);
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CVR_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return ax2 / h22(x1, x2, x3);
//   }

//   template <>
//   Inline auto Grid<ONE_D>::vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t {
//     return ax3 / h33(x1);
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
//     return ax3 / h33(x1, x2);
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CVR_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return ax3 / h33(x1, x2, x3);
//   }

//   // LOC -> CNT:
//   template <>
//   Inline auto Grid<ONE_D>::vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1) -> real_t {
//     return ax1 / std::sqrt(h11(x1));
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2) -> real_t {
//     return ax1 / std::sqrt(h11(x1, x2));
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_HAT_to_CNT_x1(const real_t& ax1, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return ax1 / std::sqrt(h11(x1, x2, x3));
//   }

//   template <>
//   Inline auto Grid<ONE_D>::vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1) -> real_t {
//     return ax2 / std::sqrt(h22(x1));
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2) -> real_t {
//     return ax2 / std::sqrt(h22(x1, x2));
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_HAT_to_CNT_x2(const real_t& ax2, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return ax2 / std::sqrt(h22(x1, x2, x3));
//   }

//   template <>
//   Inline auto Grid<ONE_D>::vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1) -> real_t {
//     return ax3 / std::sqrt(h33(x1));
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2) -> real_t {
//     return ax3 / std::sqrt(h33(x1, x2));
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_HAT_to_CNT_x3(const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> real_t {
//     return ax3 / std::sqrt(h33(x1, x2, x3));
//   }

//   // CNT -> LOC:
//   template <>
//   Inline auto Grid<ONE_D>::vec_CNT_to_LOC(const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1)
//     -> std::tuple<real_t, real_t, real_t> {
//     auto ax1_hat {vec_CNT_to_HAT_x1(ax1, x1)};
//     auto ax2_hat {vec_CNT_to_HAT_x2(ax2, x1)};
//     auto ax3_hat {vec_CNT_to_HAT_x3(ax3, x1)};
//     return vec_HAT_to_LOC(ax1_hat, ax2_hat, ax3_hat, x1);
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_CNT_to_LOC(
//     const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2)
//     -> std::tuple<real_t, real_t, real_t> {
//     auto ax1_hat {vec_CNT_to_HAT_x1(ax1, x1, x2)};
//     auto ax2_hat {vec_CNT_to_HAT_x2(ax2, x1, x2)};
//     auto ax3_hat {vec_CNT_to_HAT_x3(ax3, x1, x2)};
//     return vec_HAT_to_LOC(ax1_hat, ax2_hat, ax3_hat, x1, x2);
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_CNT_to_LOC(
//     const real_t& ax1, const real_t& ax2, const real_t& ax3, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> std::tuple<real_t, real_t, real_t> {
//     auto ax1_hat {vec_CNT_to_HAT_x1(ax1, x1, x2, x3)};
//     auto ax2_hat {vec_CNT_to_HAT_x2(ax2, x1, x2, x3)};
//     auto ax3_hat {vec_CNT_to_HAT_x3(ax3, x1, x2, x3)};
//     return vec_HAT_to_LOC(ax1_hat, ax2_hat, ax3_hat, x1, x2, x3);
//   }

//   // LOC -> CNT:
//   template <>
//   Inline auto Grid<ONE_D>::vec_LOC_to_CNT(const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1)
//     -> std::tuple<real_t, real_t, real_t> {
//     auto [ax1_hat, ax2_hat, ax3_hat] = vec_LOC_to_HAT(ax, ay, az, x1);
//     auto ax1 {vec_HAT_to_CNT_x1(ax1_hat, x1)};
//     auto ax2 {vec_HAT_to_CNT_x2(ax2_hat, x1)};
//     auto ax3 {vec_HAT_to_CNT_x3(ax3_hat, x1)};
//     return {ax1, ax2, ax3};
//   }
//   template <>
//   Inline auto Grid<TWO_D>::vec_LOC_to_CNT(
//     const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2)
//     -> std::tuple<real_t, real_t, real_t> {
//     auto [ax1_hat, ax2_hat, ax3_hat] = vec_LOC_to_HAT(ax, ay, az, x1, x2);
//     auto ax1 {vec_HAT_to_CNT_x1(ax1_hat, x1, x2)};
//     auto ax2 {vec_HAT_to_CNT_x2(ax2_hat, x1, x2)};
//     auto ax3 {vec_HAT_to_CNT_x3(ax3_hat, x1, x2)};
//     return {ax1, ax2, ax3};
//   }
//   template <>
//   Inline auto Grid<THREE_D>::vec_LOC_to_CNT(
//     const real_t& ax, const real_t& ay, const real_t& az, const real_t& x1, const real_t& x2, const real_t& x3)
//     -> std::tuple<real_t, real_t, real_t> {
//     auto [ax1_hat, ax2_hat, ax3_hat] = vec_LOC_to_HAT(ax, ay, az, x1, x2, x3);
//     auto ax1 {vec_HAT_to_CNT_x1(ax1_hat, x1, x2, x3)};
//     auto ax2 {vec_HAT_to_CNT_x2(ax2_hat, x1, x2, x3)};
//     auto ax3 {vec_HAT_to_CNT_x3(ax3_hat, x1, x2, x3)};
//     return {ax1, ax2, ax3};
//   }
// } // namespace ntt

// #endif
