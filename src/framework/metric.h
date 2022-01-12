#ifndef FRAMEWORK_METRIC_H
#define FRAMEWORK_METRIC_H

#include "global.h"

#include <stdexcept>

namespace ntt {
  /**
   * Arbitrary metric: h_ij. Coordinates vary from `0` to `nx1` ... (code units).
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  struct Metric {
    // text label of the metric
    const std::string label;
    // max of coordinates in code units
    const real_t nx1, nx2, nx3;
    // extent in `x1` in physical units
    const real_t x1_min, x1_max;
    // extent in `x2` in physical units
    const real_t x2_min, x2_max;
    // extent in `x3` in physical units
    const real_t x3_min, x3_max;

    Metric(const std::string& label_, std::vector<std::size_t> resolution, std::vector<real_t> extent)
      : label {label_},
        nx1 {resolution.size() > 0 ? (real_t)(resolution[0]) : ONE},
        nx2 {resolution.size() > 1 ? (real_t)(resolution[1]) : ONE},
        nx3 {resolution.size() > 2 ? (real_t)(resolution[2]) : ONE},
        x1_min {resolution.size() > 0 ? extent[0] : ZERO},
        x1_max {resolution.size() > 0 ? extent[1] : ZERO},
        x2_min {resolution.size() > 1 ? extent[2] : ZERO},
        x2_max {resolution.size() > 1 ? extent[3] : ZERO},
        x3_min {resolution.size() > 2 ? extent[4] : ZERO},
        x3_max {resolution.size() > 2 ? extent[5] : ZERO} {}
    virtual ~Metric() = default;

    /**
     * Convert `real_t` type code unit coordinate to cell index + displacement.
     *
     * @todo `xi + N_GHOSTS` is a bit of a hack.
     * @returns A pair of `int` and `float`: cell index + displacement.
     */
    Inline auto CU_to_Idi(const real_t& xi) const -> std::pair<int, float> {
      auto i {static_cast<int>(xi + N_GHOSTS)};
      float di {static_cast<float>(xi) - static_cast<float>(i)};
      return {i, di};
    }

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    virtual auto findSmallestCell() const -> real_t { 
      NTTError("not implemented"); 
      return ZERO;
    }

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_11 (covariant, lower index) metric component.
     */
    virtual Inline auto h_11(const coord_t<D>&) const -> real_t { 
      return ZERO;
    }
    /**
     * Compute metric component 12.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_12 (covariant, lower index) metric component.
     */
    virtual Inline auto h_12(const coord_t<D>&) const -> real_t { 
      return ZERO;
    }
    /**
     * Compute metric component 13.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_13 (covariant, lower index) metric component.
     */
    virtual Inline auto h_13(const coord_t<D>&) const -> real_t { 
      return ZERO;
    }
    /**
     * Compute metric component 21.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_21 (covariant, lower index) metric component.
     */
    virtual Inline auto h_21(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_22 (covariant, lower index) metric component.
     */
    virtual Inline auto h_22(const coord_t<D>&) const -> real_t { 
      return ZERO;
    }
    /**
     * Compute metric component 23.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_23 (covariant, lower index) metric component.
     */
    virtual Inline auto h_23(const coord_t<D>&) const -> real_t { 
      return ZERO;
    }
    /**
     * Compute metric component 31.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_31 (covariant, lower index) metric component.
     */
    virtual Inline auto h_31(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    /**
     * Compute metric component 32.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_32 (covariant, lower index) metric component.
     */
    virtual Inline auto h_32(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_33 (covariant, lower index) metric component.
     */
    virtual Inline auto h_33(const coord_t<D>&) const -> real_t { 
      return ZERO;
    }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns sqrt(det(h_ij)).
     */
    virtual Inline auto sqrt_det_h(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns Area at the pole.
     */
    virtual Inline auto polar_area(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cartesian coordinates in physical units (size of the array is D).
     */
    virtual Inline void x_Code2Cart(const coord_t<D>&, coord_t<D>&) const { }

    /**
     * Coordinate conversion from code units to Spherical physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cpherical coordinates in physical units (size of the array is D).
     */
    virtual Inline void x_Code2Sph(const coord_t<D>&, coord_t<D>&) const { }

    /**
     * Vector conversion from hatted to contravariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     * @param vi vector in contravariant basis (size of the array is 3).
     */
    virtual Inline void
    v_Hat2Cntrv(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const { }

    /**
     * Vector conversion from contravariant to hatted basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi vector in contravariant basis (size of the array is 3).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     */
    virtual Inline void v_Cntrv2Hat(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const { }
  };

} // namespace ntt

#endif
