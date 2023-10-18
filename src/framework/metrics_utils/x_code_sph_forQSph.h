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
Inline void x_Code2Sph(const coord_t<Dim3>& xi, coord_t<Dim3>& x) const {
  x[0] = r0 + math::exp(xi[0] * dchi + chi_min);
  x[1] = eta2theta(xi[1] * deta + eta_min);
  if constexpr (D == Dim2) {
    x[2] = xi[2];
  } else if constexpr (D == Dim3) {
    x[2] = xi[2] * dphi + phi_min;
  }
}

Inline auto x1_Code2Sph(const real_t& x1) const -> real_t {
  return r0 + math::exp(x1 * dchi + chi_min);
}

Inline auto x2_Code2Sph(const real_t& x2) const -> real_t {
  return eta2theta(x2 * deta + eta_min);
}

Inline auto x3_Code2Sph(const real_t& x3) const -> real_t {
  if constexpr (D == Dim2) {
    return x3;
  } else if constexpr (D == Dim3) {
    return x3 * dphi + phi_min;
  }
}

/**
 * Coordinate conversion from Spherical physical units to code units.
 * @param x coordinate array in Spherical coordinates in physical units
 * @param xi coordinate array in code units
 */
Inline void x_Sph2Code(const coord_t<Dim3>& x, coord_t<Dim3>& xi) const {
  xi[0] = (math::log(x[0] - r0) - chi_min) * dchi_inv;
  xi[1] = (theta2eta(x[1]) - eta_min) * deta_inv;
  if constexpr (D == Dim2) {
    xi[2] = x[2];
  } else if constexpr (D == Dim3) {
    xi[2] = (x[2] - phi_min) * dphi_inv;
  }
}

Inline auto x1_Sph2Code(const real_t& r) const -> real_t {
  return (math::log(r - r0) - chi_min) * dchi_inv;
}

Inline auto x2_Sph2Code(const real_t& th) const -> real_t {
  return (theta2eta(th) - eta_min) * deta_inv;
}

Inline auto x3_Sph2Code(const real_t& phi) const -> real_t {
  if constexpr (D == Dim2) {
    return phi;
  } else if constexpr (D == Dim3) {
    return (phi - phi_min) * dphi_inv;
  }
}

#endif // FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORQSPH_H