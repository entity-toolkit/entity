#ifndef FRAMEWORK_METRICS_UTILS_X_CODE_CART_FORGSPH_H
#define FRAMEWORK_METRICS_UTILS_X_CODE_CART_FORGSPH_H

#ifdef __INTELLISENSE__
#  pragma diag_suppress 1670
#  pragma diag_suppress 864
#  pragma diag_suppress 258
#  pragma diag_suppress 77
#  pragma diag_suppress 65
#  pragma diag_suppress 20
#endif

/**
 * @brief Coordinate transformations for generalized spherical coordinates.
 * @implements x: Code -> Cart
 * @implements x: Cart -> Code
 */

/**
 * Coordinate conversion from code units to Cartesian physical units.
 *
 * @param xi coordinate array in code units
 * @param x coordinate array in Cartesian physical units
 */
Inline void x_Code2Cart(const coord_t<D>& xi, coord_t<D>& x) const {
  if constexpr (D == Dim1) {
    NTTError("x_Code2Cart not implemented for 1D");
  } else if constexpr (D == Dim2) {
    coord_t<D> x_sph { ZERO };
    x_Code2Sph(xi, x_sph);
    x[0] = x_sph[0] * math::sin(x_sph[1]);
    x[1] = x_sph[0] * math::cos(x_sph[1]);
  } else if constexpr (D == Dim3) {
    coord_t<D> x_sph { ZERO };
    x_Code2Sph(xi, x_sph);
    x[0] = x_sph[0] * math::sin(x_sph[1]) * math::cos(x_sph[2]);
    x[1] = x_sph[0] * math::sin(x_sph[1]) * math::sin(x_sph[2]);
    x[2] = x_sph[0] * math::cos(x_sph[1]);
  }
}
/**
 * Coordinate conversion from Cartesian physical units to code units.
 *
 * @param x coordinate array in Cartesian coordinates in
 * physical units
 * @param xi coordinate array in code units
 */
Inline void x_Cart2Code(const coord_t<D>& x, coord_t<D>& xi) const {
  if constexpr (D == Dim1) {
    NTTError("x_Cart2Code not implemented for 1D");
  } else if constexpr (D == Dim2) {
    coord_t<D> x_sph { ZERO };
    x_sph[0] = math::sqrt(x[0] * x[0] + x[1] * x[1]);
    x_sph[1] = static_cast<real_t>(constant::HALF_PI) - math::atan2(x[1], x[0]);
    x_Sph2Code(x_sph, xi);
  } else if constexpr (D == Dim3) {
    coord_t<D> x_sph { ZERO };
    x_sph[0] = math::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    x_sph[1] = static_cast<real_t>(constant::HALF_PI) - math::atan2(x[1], x[0]);
    x_sph[2] = math::acos(x[2] / x_sph[0]);
    x_Sph2Code(x_sph, xi);
  }
}

#ifdef __INTELLISENSE__
#  pragma diag_default 20
#  pragma diag_default 65
#  pragma diag_default 77
#  pragma diag_default 258
#  pragma diag_default 864
#  pragma diag_default 1670
#endif

#endif    // FRAMEWORK_METRICS_UTILS_X_CODE_CART_FORGSPH_H