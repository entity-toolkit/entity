#ifndef FRAMEWORK_AUX_METRIC_CART_H
#define FRAMEWORK_AUX_METRIC_CART_H

#include "global.h"

/**
 * @brief Vector transformations from and to global Cartesian basis.
 * 
 * 
 * @note Functions `v_Hat2Cart` and `v_Cart2Hat` should be implemented per each metric.
 * 
 */

/**
 * Vector conversion from contravariant to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */
Inline void v_Cntrv2Cart(const coord_t<D>&                xi,
                         const vec_t<Dimension::THREE_D>& vi_cntrv,
                         vec_t<Dimension::THREE_D>&       vi_cart) const {
  vec_t<Dimension::THREE_D> vi_hat {ZERO};
  this->v_Cntrv2Hat(xi, vi_cntrv, vi_hat);
  this->v_Hat2Cart(xi, vi_hat, vi_cart);
}

/**
 * Vector conversion from global Cartesian to contravariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */
Inline void v_Cart2Cntrv(const coord_t<D>&                xi,
                         const vec_t<Dimension::THREE_D>& vi_cart,
                         vec_t<Dimension::THREE_D>&       vi_cntrv) const {
  vec_t<Dimension::THREE_D> vi_hat {ZERO};
  this->v_Cart2Hat(xi, vi_cart, vi_hat);
  this->v_Hat2Cntrv(xi, vi_hat, vi_cntrv);
}

/**
 * Vector conversion from covariant to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */
Inline void v_Cov2Cart(const coord_t<D>&                xi,
                       const vec_t<Dimension::THREE_D>& vi_cov,
                       vec_t<Dimension::THREE_D>&       vi_cart) const {
  vec_t<Dimension::THREE_D> vi_hat {ZERO};
  this->v_Cov2Hat(xi, vi_cov, vi_hat);
  this->v_Hat2Cart(xi, vi_hat, vi_cart);
}

/**
 * Vector conversion from global Cartesian to covariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 */
Inline void v_Cart2Cov(const coord_t<D>&                xi,
                       const vec_t<Dimension::THREE_D>& vi_cart,
                       vec_t<Dimension::THREE_D>&       vi_cov) const {
  vec_t<Dimension::THREE_D> vi_hat {ZERO};
  this->v_Cart2Hat(xi, vi_cart, vi_hat);
  this->v_Hat2Cov(xi, vi_hat, vi_cov);
}

#endif