#ifndef FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORSPH_H
#define FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORSPH_H

#ifdef __INTELLISENSE__
#  pragma diag_suppress 1670
#  pragma diag_suppress 864
#  pragma diag_suppress 258
#  pragma diag_suppress 77
#  pragma diag_suppress 65
#  pragma diag_suppress 20
#endif

/**
 * @brief Coordinate transformations for normal spherical.
 * @implements x: Code -> Sph
 * @implements x: Sph -> Code
 */

/**
 * Coordinate conversion from code units to Spherical physical units.
 *
 * @param xi coordinate array in code units
 * @param x coordinate array in Spherical coordinates in physical units
 */
Inline void x_Code2Sph(const coord_t<D>& xi, coord_t<D>& x) const {
  if constexpr (D == Dim1) {
    NTTError("x_Code2Sph not implemented for 1D");
  } else if constexpr (D == Dim2) {
    x[0] = xi[0] * dr + this->x1_min;
    x[1] = xi[1] * dtheta;
  } else if constexpr (D == Dim3) {
    x[0] = xi[0] * dr + this->x1_min;
    x[1] = xi[1] * dtheta;
    x[2] = xi[2] * dphi;
  }
}
/**
 * Coordinate conversion from Spherical physical units to code units.
 *
 * @param x coordinate array in Spherical coordinates in physical units
 * @param xi coordinate array in code units
 */
Inline void x_Sph2Code(const coord_t<D>& x, coord_t<D>& xi) const {
  if constexpr (D == Dim1) {
    NTTError("x_Code2Sph not implemented for 1D");
  } else if constexpr (D == Dim2) {
    xi[0] = (x[0] - this->x1_min) / dr;
    xi[1] = x[1] / dtheta;
  } else if constexpr (D == Dim3) {
    x[0] = (xi[0] - this->x1_min) / dr;
    x[1] = xi[1] / dtheta;
    x[2] = xi[2] / dphi;
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

#endif    // FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORSPH_H