#ifndef FRAMEWORK_AUX_GR_VTRANS_H
#define FRAMEWORK_AUX_GR_VTRANS_H

#include "wrapper.h"

/**
 * Compute the square root of the determinant of h-matrix.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns sqrt(det(h)).
 */
Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
  return math::sqrt(h_22(x) * (h_11(x) * h_33(x) - h_13(x) * h_13(x)));
}

/**
 * Compute inverse metric component 11 from h_ij.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h^11 (contravariant, upper index) metric component.
 * @details Finite limit at theta=0. Here, defined only in terms of the direct metric.
 * But this introduces a singularity at theta=0, so take the limit as theta->0.
 */
Inline auto h11(const coord_t<D>& x) const -> real_t {
  return h_33(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
}

/**
 * Compute inverse metric component 22 from h_ij.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h^22 (contravariant, upper index) metric component.
 */
Inline auto h22(const coord_t<D>& x) const -> real_t {
  return ONE / h_22(x);
}

/**
 * Compute inverse metric component 33 from h_ij.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h^33 (contravariant, upper index) metric component.
 * @details Singular at theta=0.
 */
Inline auto h33(const coord_t<D>& x) const -> real_t {
  return h_11(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
}

/**
 * Compute inverse metric component 13 from h_ij.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h^13 (contravariant, upper index) metric component.
 * @details Finite limit at theta=0. Here, defined only in terms of the direct metric.
 * But this introduces a singularity at theta=0, so take the limit as theta->0.
 */
Inline auto h13(const coord_t<D>& x) const -> real_t {
  return -h_13(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
}

/**
 * Vector conversion from hatted to contravariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */
Inline void v_Hat2Cntrv(const coord_t<D>&  xi,
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
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 */
Inline void v_Cntrv2Hat(const coord_t<D>&  xi,
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
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 */
Inline void v_Hat2Cov(const coord_t<D>&  xi,
                      const vec_t<Dim3>& vi_hat,
                      vec_t<Dim3>&       vi_cov) const {
  vi_cov[0] = vi_hat[0] / math::sqrt(h11(xi)) + vi_hat[2] * h_13(xi) / math::sqrt(h_33(xi));
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
Inline void v_Cov2Hat(const coord_t<D>&  xi,
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
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */
Inline void v_Cov2Cntrv(const coord_t<D>&  xi,
                        const vec_t<Dim3>& vi_cov,
                        vec_t<Dim3>&       vi_cntrv) const {
  vi_cntrv[0] = vi_cov[0] * h11(xi) + vi_cov[2] * h13(xi);
  vi_cntrv[1] = vi_cov[1] * h22(xi);
  vi_cntrv[2] = vi_cov[0] * h13(xi) + vi_cov[2] * h33(xi);
}

/**
 * Vector conversion from contravariant to covariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_cov vector in covaraint basis (size of the array is 3).
 */
Inline void v_Cntrv2Cov(const coord_t<D>&  xi,
                        const vec_t<Dim3>& vi_cntrv,
                        vec_t<Dim3>&       vi_cov) const {
  vi_cov[0] = vi_cntrv[0] * h_11(xi) + vi_cntrv[2] * h_13(xi);
  vi_cov[1] = vi_cntrv[1] * h_22(xi);
  vi_cov[2] = vi_cntrv[0] * h_13(xi) + vi_cntrv[2] * h_33(xi);
}

/**
 * Compute the squared norm of a covariant vector.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @return Norm of the covariant vector.
 */
Inline auto v_CovNorm(const coord_t<D>& xi, const vec_t<Dim3>& vi_cov) const -> real_t {
  return vi_cov[0] * vi_cov[0] * h11(xi) + vi_cov[1] * vi_cov[1] * h22(xi)
         + vi_cov[2] * vi_cov[2] * h33(xi) + TWO * vi_cov[0] * vi_cov[2] * h13(xi);
}

/**
 * Compute the squared norm of a contravariant vector.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @return Norm of the contravariant vector.
 */
Inline auto v_CntrvNorm(const coord_t<D>& xi, const vec_t<Dim3>& vi_cntrv) const -> real_t {
  return vi_cntrv[0] * vi_cntrv[0] * h_11(xi) + vi_cntrv[1] * vi_cntrv[1] * h_22(xi)
         + vi_cntrv[2] * vi_cntrv[2] * h_33(xi) + TWO * vi_cntrv[0] * vi_cntrv[2] * h_13(xi);
}

#endif