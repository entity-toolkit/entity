#ifndef FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORQSPH_H
#define FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORQSPH_H

/**
 * @brief Coordinate transformations for stretched spherical.
 * @implements x: Code -> Sph
 * @implements x: Sph -> Code
 */

/**
 * Coordinate conversion from code units to Spherical physical units.
 * @param xi coordinate array in code units
 * @param x coordinate array in Spherical coordinates in physical units
 */
Inline void x_Code2Sph(const coord_t<D>& xi, coord_t<D>& x) const {
  x[0] = x1_Code2Sph(xi[0]);
  if constexpr (D != Dim1) {
    x[1] = x2_Code2Sph(xi[1]);
    if constexpr (D == Dim3) {
      x[2] = x3_Code2Sph(xi[2]);
    }
  }
}

Inline auto x1_Code2Sph(const real_t& x1) const -> real_t {
  return r0 + math::exp(x1 * dchi + chi_min);
}

Inline auto x2_Code2Sph(const real_t& x2) const -> real_t {
  return eta2theta(x2 * deta + eta_min);
}

Inline auto x3_Code2Sph(const real_t& x3) const -> real_t {
  if constexpr (D != Dim3) {
    return x3;
  } else {
    return x3 * dphi + phi_min;
  }
}

/**
 * Coordinate conversion from Spherical physical units to code units.
 * @param x coordinate array in Spherical coordinates in physical units
 * @param xi coordinate array in code units
 */
Inline void x_Sph2Code(const coord_t<D>& x, coord_t<D>& xi) const {
  xi[0] = x1_Sph2Code(x[0]);
  if constexpr (D != Dim1) {
    xi[1] = x2_Sph2Code(x[1]);
    if constexpr (D == Dim3) {
      xi[2] = x3_Sph2Code(x[2]);
    }
  }
}

Inline auto x1_Sph2Code(const real_t& r) const -> real_t {
  return (math::log(r - r0) - chi_min) * dchi_inv;
}

Inline auto x2_Sph2Code(const real_t& th) const -> real_t {
  return (theta2eta(th) - eta_min) * deta_inv;
}

Inline auto x3_Sph2Code(const real_t& phi) const -> real_t {
  if constexpr (D != Dim3) {
    return phi;
  } else {
    return (phi - phi_min) * dphi_inv;
  }
}

#endif // FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORQSPH_H
