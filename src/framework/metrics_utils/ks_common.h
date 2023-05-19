#ifndef FRAMEWORK_METRICS_KS_COMMON_H
#define FRAMEWORK_METRICS_KS_COMMON_H

#ifdef __INTELLISENSE__
#  pragma diag_suppress 1670
#  pragma diag_suppress 864
#  pragma diag_suppress 20
#endif

/**
 * Compute the square root of the determinant of h-matrix.
 *
 * @param x coordinate array in code units
 * @returns sqrt(det(h)).
 */
Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
  return math::sqrt(h_22(x) * (h_11(x) * h_33(x) - h_13(x) * h_13(x)));
}

/**
 * Compute inverse metric component 11 from h_ij.
 *
 * @param x coordinate array in code units
 * @returns h^11 (contravariant, upper index) metric component.
 */
Inline auto h11(const coord_t<D>& x) const -> real_t {
  return h_33(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
}

/**
 * Compute inverse metric component 22 from h_ij.
 *
 * @param x coordinate array in code units
 * @returns h^22 (contravariant, upper index) metric component.
 */
Inline auto h22(const coord_t<D>& x) const -> real_t {
  return ONE / h_22(x);
}

/**
 * Compute inverse metric component 33 from h_ij.
 *
 * @param x coordinate array in code units
 * @returns h^33 (contravariant, upper index) metric component.
 */
Inline auto h33(const coord_t<D>& x) const -> real_t {
  return h_11(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
}

/**
 * Compute inverse metric component 13 from h_ij.
 *
 * @param x coordinate array in code units
 * @returns h^13 (contravariant, upper index) metric component.
 */
Inline auto h13(const coord_t<D>& x) const -> real_t {
  return -h_13(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
}

/**
 * Vector conversion from hatted to contravariant basis.
 *
 * @param xi coordinate array in code units
 * @param vi_hat vector in hatted basis
 * @param vi_cntrv vector in contravariant basis
 */
Inline void v3_Hat2Cntrv(const coord_t<D>&  xi,
                         const vec_t<Dim3>& vi_hat,
                         vec_t<Dim3>&       vi_cntrv) const {
  real_t A0 { math::sqrt(h11(xi)) };
  vi_cntrv[0] = vi_hat[0] * A0;
  vi_cntrv[1] = vi_hat[1] / math::sqrt(h_22(xi));
  vi_cntrv[2] = vi_hat[2] / math::sqrt(h_33(xi)) - vi_hat[0] * A0 * h_13(xi) / h_33(xi);
}

/**
 * Vector conversion from contravariant to hatted basis.
 *
 * @param xi coordinate array in code units
 * @param vi_cntrv vector in contravariant basis
 * @param vi_hat vector in hatted basis
 */
Inline void v3_Cntrv2Hat(const coord_t<D>&  xi,
                         const vec_t<Dim3>& vi_cntrv,
                         vec_t<Dim3>&       vi_hat) const {
  vi_hat[0] = vi_cntrv[0] / math::sqrt(h11(xi));
  vi_hat[1] = vi_cntrv[1] * math::sqrt(h_22(xi));
  vi_hat[2]
    = vi_cntrv[2] * math::sqrt(h_33(xi)) + vi_cntrv[0] * h_13(xi) / math::sqrt(h_33(xi));
}

/**
 * Vector conversion from hatted to covariant basis.
 *
 * @param xi coordinate array in code units
 * @param vi_hat vector in hatted basis
 * @param vi_cov vector in covariant basis
 */
Inline void v3_Hat2Cov(const coord_t<D>&  xi,
                       const vec_t<Dim3>& vi_hat,
                       vec_t<Dim3>&       vi_cov) const {
  vi_cov[0] = vi_hat[0] / math::sqrt(h11(xi)) + vi_hat[2] * h_13(xi) / math::sqrt(h_33(xi));
  vi_cov[1] = vi_hat[1] * math::sqrt(h_22(xi));
  vi_cov[2] = vi_hat[2] * math::sqrt(h_33(xi));
}

/**
 * Vector conversion from covariant to hatted basis.
 *
 * @param xi coordinate array in code units
 * @param vi_cov vector in covariant basis
 * @param vi_hat vector in hatted basis
 */
Inline void v3_Cov2Hat(const coord_t<D>&  xi,
                       const vec_t<Dim3>& vi_cov,
                       vec_t<Dim3>&       v_hat) const {
  real_t A0 { math::sqrt(h11(xi)) };
  v_hat[0] = vi_cov[0] * A0 - vi_cov[2] * A0 * h_13(xi) / h_33(xi);
  v_hat[1] = vi_cov[1] / math::sqrt(h_22(xi));
  v_hat[2] = vi_cov[2] / math::sqrt(h_33(xi));
}

/**
 * Vector conversion from covariant to contravariant basis.
 *
 * @param xi coordinate array in code units
 * @param vi_cov vector in covariant basis
 * @param vi_cntrv vector in contravariant basis
 */
Inline void v3_Cov2Cntrv(const coord_t<D>&  xi,
                         const vec_t<Dim3>& vi_cov,
                         vec_t<Dim3>&       vi_cntrv) const {
  vi_cntrv[0] = vi_cov[0] * h11(xi) + vi_cov[2] * h13(xi);
  vi_cntrv[1] = vi_cov[1] * h22(xi);
  vi_cntrv[2] = vi_cov[0] * h13(xi) + vi_cov[2] * h33(xi);
}

/**
 * Vector conversion from contravariant to covariant basis.
 *
 * @param xi coordinate array in code units
 * @param vi_cntrv vector in contravariant basis
 * @param vi_cov vector in covaraint basis
 */
Inline void v3_Cntrv2Cov(const coord_t<D>&  xi,
                         const vec_t<Dim3>& vi_cntrv,
                         vec_t<Dim3>&       vi_cov) const {
  vi_cov[0] = vi_cntrv[0] * h_11(xi) + vi_cntrv[2] * h_13(xi);
  vi_cov[1] = vi_cntrv[1] * h_22(xi);
  vi_cov[2] = vi_cntrv[0] * h_13(xi) + vi_cntrv[2] * h_33(xi);
}

/**
 * Compute the squared norm of a covariant vector.
 *
 * @param xi coordinate array in code units
 * @param vi_cov vector in covariant basis
 * @return Norm of the covariant vector.
 */
Inline auto v3_CovNorm(const coord_t<D>& xi, const vec_t<Dim3>& vi_cov) const -> real_t {
  return vi_cov[0] * vi_cov[0] * h11(xi) + vi_cov[1] * vi_cov[1] * h22(xi)
         + vi_cov[2] * vi_cov[2] * h33(xi) + TWO * vi_cov[0] * vi_cov[2] * h13(xi);
}

/**
 * Compute the squared norm of a contravariant vector.
 *
 * @param xi coordinate array in code units
 * @param vi_cntrv vector in contravariant basis
 * @return Norm of the contravariant vector.
 */
Inline auto v3_CntrvNorm(const coord_t<D>& xi, const vec_t<Dim3>& vi_cntrv) const -> real_t {
  return vi_cntrv[0] * vi_cntrv[0] * h_11(xi) + vi_cntrv[1] * vi_cntrv[1] * h_22(xi)
         + vi_cntrv[2] * vi_cntrv[2] * h_33(xi) + TWO * vi_cntrv[0] * vi_cntrv[2] * h_13(xi);
}

#ifdef __INTELLISENSE__
#  pragma diag_default 20
#  pragma diag_default 864
#  pragma diag_default 1670
#endif

#endif