#ifndef FRAMEWORK_AUX_DIAG_VTRANS_H
#define FRAMEWORK_AUX_DIAG_VTRANS_H

#include "global.h"

/**
 * Vector conversion from hatted to contravariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */
Inline void v_Hat2Cntrv(const coord_t<D>&                xi,
                        const vec_t<Dimension::THREE_D>& vi_hat,
                        vec_t<Dimension::THREE_D>&       vi_cntrv) const {
  vi_cntrv[0] = vi_hat[0] / math::sqrt(h_11(xi));
  vi_cntrv[1] = vi_hat[1] / math::sqrt(h_22(xi));
  vi_cntrv[2] = vi_hat[2] / math::sqrt(h_33(xi));
}

/**
 * Vector conversion from contravariant to hatted basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 */
Inline void v_Cntrv2Hat(const coord_t<D>&                xi,
                        const vec_t<Dimension::THREE_D>& vi_cntrv,
                        vec_t<Dimension::THREE_D>&       vi_hat) const {
  vi_hat[0] = vi_cntrv[0] * math::sqrt(h_11(xi));
  vi_hat[1] = vi_cntrv[1] * math::sqrt(h_22(xi));
  vi_hat[2] = vi_cntrv[2] * math::sqrt(h_33(xi));
}

/**
 * Vector conversion from hatted to covariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 */
Inline void v_Hat2Cov(const coord_t<D>&                xi,
                      const vec_t<Dimension::THREE_D>& vi_hat,
                      vec_t<Dimension::THREE_D>&       vi_cov) const {
  vi_cov[0] = vi_hat[0] * math::sqrt(h_11(xi));
  vi_cov[1] = vi_hat[1] * math::sqrt(h_22(xi));
  vi_cov[2] = vi_hat[2] * math::sqrt(h_33(xi));
}

/**
 * Vector conversion from covariant to hatted basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 */
Inline void v_Cov2Hat(const coord_t<D>&                xi,
                      const vec_t<Dimension::THREE_D>& vi_cov,
                      vec_t<Dimension::THREE_D>&       vi_hat) const {
  vi_hat[0] = vi_cov[0] / math::sqrt(h_11(xi));
  vi_hat[1] = vi_cov[1] / math::sqrt(h_22(xi));
  vi_hat[2] = vi_cov[2] / math::sqrt(h_33(xi));
}

/**
 * Vector conversion from covariant to contravariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */
Inline void v_Cov2Cntrv(const coord_t<D>&                xi,
                        const vec_t<Dimension::THREE_D>& vi_cov,
                        vec_t<Dimension::THREE_D>&       vi_cntrv) const {
  vi_cntrv[0] = vi_cov[0] / h_11(xi);
  vi_cntrv[1] = vi_cov[1] / h_22(xi);
  vi_cntrv[2] = vi_cov[2] / h_33(xi);
}

/**
 * Vector conversion from contravariant to covariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_cov vector in covaraint basis (size of the array is 3).
 */
Inline void v_Cntrv2Cov(const coord_t<D>&                xi,
                        const vec_t<Dimension::THREE_D>& vi_cntrv,
                        vec_t<Dimension::THREE_D>&       vi_cov) const {
  vi_cov[0] = vi_cntrv[0] * h_11(xi);
  vi_cov[1] = vi_cntrv[1] * h_22(xi);
  vi_cov[2] = vi_cntrv[2] * h_33(xi);
}

/**
 * Compute the norm of a covariant vector.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @return Norm of the covariant vector.
 */
Inline auto v_CovNorm(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cov) const
  -> real_t {
  return vi_cov[0] * vi_cov[0] / h_11(xi) + vi_cov[1] * vi_cov[1] / h_22(xi)
         + vi_cov[2] * vi_cov[2] / h_33(xi);
}

#endif