/**
 * @brief Vector transformations for normal spherical.
 * @implements v3: Cntrv -> PhysCntrv
 * @implements v3: PhysCntrv -> Cntrv
 * @implements v3: Cov -> PhysCov
 * @implements v3: PhysCov -> Cov
 */

/**
 * Vector conversion from contravariant to spherical contravariant.
 *
 * @param xi coordinate array in code units
 * @param vi_cntrv vector in contravariant basis
 * @param vsph_cntrv vector in spherical contravariant basis
 */
Inline void v3_Cntrv2PhysCntrv(const coord_t<D>&  xi,
                               const vec_t<Dim::_3D>& vi_cntrv,
                               vec_t<Dim::_3D>&       vsph_cntrv) const {
  vsph_cntrv[0] = vi_cntrv[0] * math::exp(xi[0] * dchi + chi_min) * dchi;
  vsph_cntrv[1] = vi_cntrv[1] * dtheta_deta(xi[1] * deta + eta_min) * deta;
  if constexpr (D == Dim::_2D) {
    vsph_cntrv[2] = vi_cntrv[2];
  } else {
    vsph_cntrv[2] = vi_cntrv[2] * dphi;
  }
}

/**
 * Vector conversion from spherical contravariant to contravariant.
 *
 * @param xi coordinate array in code units
 * @param vsph_cntrv vector in spherical contravariant basis
 * @param vi_cntrv vector in contravariant basis
 */
Inline void v3_PhysCntrv2Cntrv(const coord_t<D>&  xi,
                               const vec_t<Dim::_3D>& vsph_cntrv,
                               vec_t<Dim::_3D>&       vi_cntrv) const {
  vi_cntrv[0] = vsph_cntrv[0] * dchi_inv / (math::exp(xi[0] * dchi + chi_min));
  vi_cntrv[1] = vsph_cntrv[1] * deta_inv / (dtheta_deta(xi[1] * deta + eta_min));
  if constexpr (D == Dim::_2D) {
    vi_cntrv[2] = vsph_cntrv[2];
  } else {
    vi_cntrv[2] = vsph_cntrv[2] * dphi_inv;
  }
}

/**
 * Vector conversion from covariant to spherical covariant.
 *
 * @param xi coordinate array in code units
 * @param vi_cov vector in covariant basis
 * @param vsph_cov vector in spherical covariant basis
 */
Inline void v3_Cov2PhysCov(const coord_t<D>&  xi,
                           const vec_t<Dim::_3D>& vi_cov,
                           vec_t<Dim::_3D>&       vsph_cov) const {
  vsph_cov[0] = vi_cov[0] * dchi_inv / (math::exp(xi[0] * dchi + chi_min));
  vsph_cov[1] = vi_cov[1] * deta_inv / (dtheta_deta(xi[1] * deta + eta_min));
  if constexpr (D == Dim::_2D) {
    vsph_cov[2] = vi_cov[2];
  } else {
    vsph_cov[2] = vi_cov[2] * dphi_inv;
  }
}

/**
 * Vector conversion from spherical covariant to covariant.
 *
 * @param xi coordinate array in code units
 * @param vsph_cov vector in spherical covariant basis
 * @param vi_cov vector in covariant basis
 */
Inline void v3_PhysCov2Cov(const coord_t<D>&  xi,
                           const vec_t<Dim::_3D>& vsph_cov,
                           vec_t<Dim::_3D>&       vi_cov) const {
  vi_cov[0] = vsph_cov[0] * (math::exp(xi[0] * dchi + chi_min) * dchi);
  vi_cov[1] = vsph_cov[1] * (dtheta_deta(xi[1] * deta + eta_min) * deta);
  if constexpr (D == Dim::_2D) {
    vi_cov[2] = vsph_cov[2];
  } else {
    vi_cov[2] = vsph_cov[2] * dphi;
  }
}