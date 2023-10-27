#ifndef FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORSPH_H
#define FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORSPH_H

/**
 * @brief Coordinate transformations for normal spherical.
 * @implements x: Code -> Sph
 * @implements x: Sph -> Code
 */

/**
 * Coordinate conversion from code units to Spherical physical units.
 * @param xi coordinate array in code units
 * @param x coordinate array in Spherical coordinates in physical units
 */
Inline void x_Code2Sph(const coord_t<Dim3>& xi, coord_t<Dim3>& x) const {
  x[0] = xi[0] * dr + this->x1_min;
  x[1] = xi[1] * dtheta + this->x2_min;
  if constexpr (D == Dim2) {
    x[2] = xi[2];
  } else if constexpr (D == Dim3) {
    x[2] = xi[2] * dphi + this->x3_min;
  }
}

Inline auto x1_Code2Sph(const real_t& x1) const -> real_t {
  return x1 * dr + this->x1_min;
}

Inline auto x2_Code2Sph(const real_t& x2) const -> real_t {
  return x2 * dtheta + this->x2_min;
}

Inline auto x3_Code2Sph(const real_t& x3) const -> real_t {
  if constexpr (D == Dim2) {
    return x3;
  } else if constexpr (D == Dim3) {
    return x3 * dphi + this->x3_min;
  }
}

/**
 * Coordinate conversion from Spherical physical units to code units.
 * @param x coordinate array in Spherical coordinates in physical units
 * @param xi coordinate array in code units
 */
Inline void x_Sph2Code(const coord_t<Dim3>& x, coord_t<Dim3>& xi) const {
  xi[0] = (x[0] - this->x1_min) * dr_inv;
  xi[1] = (x[1] - this->x2_min) * dtheta_inv;
  if constexpr (D == Dim2) {
    xi[2] = x[2];
  } else if constexpr (D == Dim3) {
    xi[2] = (x[2] - this->x3_min) * dphi_inv;
  }
}

Inline auto x1_Sph2Code(const real_t& r) const -> real_t {
  return (r - this->x1_min) * dr_inv;
}

Inline auto x2_Sph2Code(const real_t& th) const -> real_t {
  return (th - this->x2_min) * dtheta_inv;
}

Inline auto x3_Sph2Code(const real_t& phi) const -> real_t {
  if constexpr (D == Dim2) {
    return phi;
  } else if constexpr (D == Dim3) {
    return (phi - this->x3_min) * dphi_inv;
  }
}

#endif // FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORSPH_H