#ifndef FRAMEWORK_AUX_METRIC_CART_CART_H
#define FRAMEWORK_AUX_METRIC_CART_CART_H

#include "global.h"

/**
 * @brief Vector transformations from global Cartesian (hatted) to global Cartesian basis and
 * back.
 *
 * @note These functions do nothing, and are added purely for consistency.
 *
 */

/**
 * Vector conversion from hatted (cartesian) to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_hat vector in hatted (cartesian) basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */
Inline void v_Hat2Cart(const coord_t<D>&,
                       const vec_t<Dimension::THREE_D>& vi_hat,
                       vec_t<Dimension::THREE_D>& vi_cart) const {
  vi_cart[0] = vi_hat[0];
  vi_cart[1] = vi_hat[1];
  vi_cart[2] = vi_hat[2];
}

/**
 * Vector conversion from hatted (cartesian) to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_hat vector in hatted (cartesian) basis (size of the array is 3).
 */
Inline void v_Cart2Hat(const coord_t<D>&,
                       const vec_t<Dimension::THREE_D>& vi_cart,
                       vec_t<Dimension::THREE_D>& vi_hat) const {
  vi_hat[0] = vi_cart[0];
  vi_hat[1] = vi_cart[1];
  vi_hat[2] = vi_cart[2];
}

#endif