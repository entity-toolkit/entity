#ifndef FRAMEWORK_METRICS_NON_DIAGONAL_METRIC_TRANSFORM_H
#define FRAMEWORK_METRICS_NON_DIAGONAL_METRIC_TRANSFORM_H

#include "global.h"

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      return std::sqrt(h_22(x) * (h_11(x) * h_33(x) - h_13(x) * h_13(x)));
    }

    /**
     * Compute inverse metric component 11 from h_ij.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h^11 (contravariant, upper index) metric component.
     * @details Finite limit at theta=0. Here, defined only in terms of the direct metric.
     * But this introduces a singularity at theta=0, so take the limit as theta->0.
     */
    Inline auto h_11_inv(const coord_t<D>& x) const -> real_t {
      coord_t<D> y;
      coord_t<D> rth_;
      real_t h_33_cov, h_13_cov, inv1, inv2;
      std::copy(std::begin(x), std::end(x), std::begin(y));
      x_Code2Sph(x, rth_);
      if (std::sin(rth_[1]) == ZERO) {
      y[1] = x[1] + 1e-1;
      h_33_cov = h_33(y);
      h_13_cov = h_13(y);
      inv1 = h_33_cov / (h_11(y) * h_33_cov - h_13_cov * h_13_cov);
      y[1] = x[1] - 1e-1;
      h_33_cov = h_33(y);
      h_13_cov = h_13(y);
      inv2 = h_33_cov / (h_11(y) * h_33_cov - h_13_cov * h_13_cov);
      return HALF * (inv1 + inv2);
      } else {
      h_33_cov = h_33(x);
      h_13_cov = h_13(x);
      return h_33_cov / (h_11(x) * h_33_cov - h_13_cov * h_13_cov);
      }
    }  

    /**
     * Compute inverse metric component 22 from h_ij.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h^22 (contravariant, upper index) metric component.
     */
    Inline auto h_22_inv(const coord_t<D>& x) const -> real_t {
      return ONE / h_22(x);
    }  

    /**
     * Compute inverse metric component 33 from h_ij.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h^33 (contravariant, upper index) metric component.
     * @details Singular at theta=0. No need to take a limit.
     */
    Inline auto h_33_inv(const coord_t<D>& x) const -> real_t {
      real_t h_11_cov, h_13_cov;
      h_11_cov = h_11(x);
      h_13_cov = h_13(x);
      return h_11_cov / (h_11_cov * h_33(x) - h_13_cov * h_13_cov);
    }  

    /**
     * Compute inverse metric component 13 from h_ij.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h^13 (contravariant, upper index) metric component.
     * @details Finite limit at theta=0. Here, defined only in terms of the direct metric.
     * But this introduces a singularity at theta=0, so take the limit as theta->0.
     */
    Inline auto h_13_inv(const coord_t<D>& x) const -> real_t {
      coord_t<D> y;
      coord_t<D> rth_;
      real_t h_33_cov, h_13_cov, inv1, inv2;
      std::copy(std::begin(x), std::end(x), std::begin(y));
      x_Code2Sph(x, rth_);
      if (std::sin(rth_[1]) == ZERO) {
      y[1] = x[1] + 1e-1;
      h_13_cov = h_13(y);
      inv1 = h_13_cov / (h_11(y) * h_33(y) - h_13_cov * h_13_cov);
      y[1] = x[1] - 1e-1;
      h_13_cov = h_13(y);
      inv2 = h_13_cov / (h_11(y) * h_33(y) - h_13_cov * h_13_cov);
      return HALF * (inv1 + inv2);
      } else {
      h_13_cov = h_13(x);
      return h_13_cov / (h_11(x) * h_33(x) - h_13_cov * h_13_cov);
      }
    }  

/**
 * Vector conversion from hatted to contravariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */
  Inline void  v_Hat2Cntrv(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_hat, vec_t<Dimension::THREE_D>& vi_cntrv) const {
    real_t A0 {std::sqrt(h_11_inv(xi))};
    vi_cntrv[0] = vi_hat[0] * A0;
    vi_cntrv[1] = vi_hat[1] / std::sqrt(h_22(xi));
    vi_cntrv[2] = vi_hat[2] / std::sqrt(h_33(xi)) - vi_hat[0] * A0 * h_13(xi) / h_33(xi);
  }

/**
 * Vector conversion from contravariant to hatted basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 */
Inline void v_Cntrv2Hat(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cntrv, vec_t<Dimension::THREE_D>& vi_hat) const {
    vi_hat[0] = vi_cntrv[0] / std::sqrt(h_11_inv(xi));
    vi_hat[1] = vi_cntrv[1] * std::sqrt(h_22(xi));
    vi_hat[2] = vi_cntrv[2] * std::sqrt(h_33(xi)) + vi_cntrv[0] * h_13(xi) / std::sqrt(h_33(xi));
  }

/**
 * Vector conversion from hatted to covariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 */
Inline void v_Hat2Cov(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_hat, vec_t<Dimension::THREE_D>& vi_cov) const {
    vi_cov[0] = vi_hat[0] / std::sqrt(h_11_inv(xi)) + vi_hat[2] *  h_13(xi) / std::sqrt(h_33(xi));
    vi_cov[1] = vi_hat[1] * std::sqrt(h_22(xi));
    vi_cov[2] = vi_hat[2] * std::sqrt(h_33(xi)) ;
}

/**
 * Vector conversion from covariant to hatted basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @param vi_hat vector in hatted basis (size of the array is 3).
 */
Inline void v_Cov2Hat(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cov, vec_t<Dimension::THREE_D>& v_hat) const {
    real_t A0 {std::sqrt(h_11_inv(xi))};        
    v_hat[0] = vi_cov[0] * A0 - vi_cov[2] * A0 * h_13(xi) / h_33(xi);
    v_hat[1] = vi_cov[1] / std::sqrt(h_22(xi));
    v_hat[2] = vi_cov[2] / std::sqrt(h_33(xi)) ;
  }

/**
 * Vector conversion from contravariant to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */

Inline void v_Cntrv2Cart(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cntrv, vec_t<Dimension::THREE_D>& vi_cart) const {
  this->v_Cntrv2Hat(xi, vi_cntrv, vi_cart);
}
/**
 * Vector conversion from global Cartesian to contravariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */

Inline void v_Cart2Cntrv(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cart, vec_t<Dimension::THREE_D>& vi_cntrv) const {
  this->v_Hat2Cntrv(xi, vi_cart, vi_cntrv);
}
/**
 * Vector conversion from covariant to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */

Inline void v_Cov2Cart(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cov, vec_t<Dimension::THREE_D>& vi_cart) const {
  this->v_Cov2Hat(xi, vi_cov, vi_cart);
}

/**
 * Vector conversion from global Cartesian to covariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 */
Inline void
v_Cart2Cov(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cart, vec_t<Dimension::THREE_D>& vi_cov) const {
  this->v_Hat2Cov(xi, vi_cart, vi_cov);
}

/**
 * Vector conversion from covariant to contravariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */
Inline void
v_Cov2Cntrv(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cov, vec_t<Dimension::THREE_D>& vi_cntrv) const {
  vi_cntrv[0] = vi_cov[0] * h_11_inv(xi) + vi_cov[2] * h_13_inv(xi);
  vi_cntrv[1] = vi_cov[1] * h_22_inv(xi);
  vi_cntrv[2] = vi_cov[2] * h_33_inv(xi) + vi_cov[0] * h_13_inv(xi);
}

/**
 * Vector conversion from contravariant to covariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_cov vector in covaraint basis (size of the array is 3).
 */
Inline void
v_Cntrv2Cov(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cntrv, vec_t<Dimension::THREE_D>& vi_cov) const {
  vi_cov[0] = vi_cntrv[0] * h_11(xi) + vi_cntrv[2] * h_13(xi);
  vi_cov[1] = vi_cntrv[1] * h_22(xi);
  vi_cov[2] = vi_cntrv[2] * h_33(xi) + vi_cntrv[0] * h_13(xi);
}

/**
 * Compute the squared norm of a covariant vector.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @return Norm of the covariant vector.
 */
Inline auto v_CovNorm(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cov) const -> real_t {
  return vi_cov[0] * vi_cov[0] * h_11_inv(xi) + vi_cov[1] * vi_cov[1] * h_22_inv(xi) + vi_cov[2] * vi_cov[2] * h_33_inv(xi) + TWO * vi_cov[0] * vi_cov[2] * h_13_inv(xi);
}

/**
 * Compute the squared norm of a contravariant vector.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @return Norm of the contravariant vector.
 */
Inline auto v_CntrvNorm(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cntrv) const -> real_t {
  return vi_cntrv[0] * vi_cntrv[0] * h_11(xi) + vi_cntrv[1] * vi_cntrv[1] * h_22(xi) + vi_cntrv[2] * vi_cntrv[2] * h_33(xi) + TWO * vi_cntrv[0] * vi_cntrv[2] * h_13(xi);
}

#endif