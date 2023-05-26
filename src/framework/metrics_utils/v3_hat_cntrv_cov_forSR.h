#ifndef FRAMEWORK_METRICS_UTILS_V3_HAT_CNTRV_COV_FORSR_H
#define FRAMEWORK_METRICS_UTILS_V3_HAT_CNTRV_COV_FORSR_H

#ifdef __INTELLISENSE__
#  pragma diag_suppress 1670
#  pragma diag_suppress 864
#  pragma diag_suppress 258
#  pragma diag_suppress 77
#  pragma diag_suppress 65
#  pragma diag_suppress 20
#endif

/**
 * @brief Vector transformations for SR.
 * @implements v3: Hat -> Cntrv
 * @implements v3: Hat -> Cov
 * @implements v3: Cntrv -> Hat
 * @implements v3: Cov -> Hat
 * @implements v3: Cntrv -> Cov
 * @implements v3: Cov -> Cntrv
 */

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
  vi_cntrv[0] = vi_hat[0] / math::sqrt(h_11(xi));
  vi_cntrv[1] = vi_hat[1] / math::sqrt(h_22(xi));
  vi_cntrv[2] = vi_hat[2] / math::sqrt(h_33(xi));
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
  vi_hat[0] = vi_cntrv[0] * math::sqrt(h_11(xi));
  vi_hat[1] = vi_cntrv[1] * math::sqrt(h_22(xi));
  vi_hat[2] = vi_cntrv[2] * math::sqrt(h_33(xi));
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
  vi_cov[0] = vi_hat[0] * math::sqrt(h_11(xi));
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
                       vec_t<Dim3>&       vi_hat) const {
  vi_hat[0] = vi_cov[0] / math::sqrt(h_11(xi));
  vi_hat[1] = vi_cov[1] / math::sqrt(h_22(xi));
  vi_hat[2] = vi_cov[2] / math::sqrt(h_33(xi));
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
  vi_cntrv[0] = vi_cov[0] / h_11(xi);
  vi_cntrv[1] = vi_cov[1] / h_22(xi);
  vi_cntrv[2] = vi_cov[2] / h_33(xi);
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
  vi_cov[0] = vi_cntrv[0] * h_11(xi);
  vi_cov[1] = vi_cntrv[1] * h_22(xi);
  vi_cov[2] = vi_cntrv[2] * h_33(xi);
}

#ifdef __INTELLISENSE__
#  pragma diag_default 20
#  pragma diag_default 65
#  pragma diag_default 77
#  pragma diag_default 258
#  pragma diag_default 864
#  pragma diag_default 1670
#endif

#endif    // FRAMEWORK_METRICS_UTILS_V3_HAT_CNTRV_COV_FORSR_H