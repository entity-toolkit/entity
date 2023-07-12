#ifndef FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORQSPH_H
#define FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORQSPH_H

/**
 * @brief Coordinate transformations for stretched spherical.
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
  if constexpr (D == Dim2) {
    real_t chi { xi[0] * dchi + chi_min };
    real_t eta { xi[1] * deta + eta_min };
    x[0] = r0 + math::exp(chi);
    x[1] = eta2theta(eta);
  } else if constexpr (D == Dim3) {
    real_t chi { xi[0] * dchi + chi_min };
    real_t eta { xi[1] * deta + eta_min };
    real_t phi { xi[2] * dphi + phi_min };
    x[0] = r0 + math::exp(chi);
    x[1] = eta2theta(eta);
    x[2] = phi;
  }
}

/**
 * Coordinate conversion from Spherical physical units to code units.
 *
 * @param x coordinate array in Spherical coordinates in physical units
 * @param xi coordinate array in code units
 */
Inline void x_Sph2Code(const coord_t<D>& x, coord_t<D>& xi) const {
  if constexpr (D == Dim2) {
    real_t chi { math::log(x[0] - r0) };
    real_t eta { theta2eta(x[1]) };
    xi[0] = (chi - chi_min) * dchi_inv;
    xi[1] = (eta - eta_min) * deta_inv;
  } else if constexpr (D == Dim3) {
    real_t chi { math::log(x[0] - r0) };
    real_t eta { theta2eta(x[1]) };
    real_t phi { x[2] };
    xi[0] = (chi - chi_min) * dchi_inv;
    xi[1] = (eta - eta_min) * deta_inv;
    xi[2] = (phi - phi_min) * dphi_inv;
  }
}

#endif    // FRAMEWORK_METRICS_UTILS_X_CODE_SPH_FORQSPH_H