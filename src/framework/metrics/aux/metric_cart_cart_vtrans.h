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
                       const vec_t<Dimension::THREE_D>&,
                       vec_t<Dimension::THREE_D>&) const {}

/**
 * Vector conversion from hatted (cartesian) to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_hat vector in hatted (cartesian) basis (size of the array is 3).
 */
Inline void v_Cart2Hat(const coord_t<D>&,
                       const vec_t<Dimension::THREE_D>&,
                       vec_t<Dimension::THREE_D>&) const {}

#endif