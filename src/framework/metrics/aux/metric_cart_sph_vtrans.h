#ifndef FRAMEWORK_AUX_METRIC_CART_SPH_H
#define FRAMEWORK_AUX_METRIC_CART_SPH_H

#include "global.h"

/**
 * @brief Vector transformations from global Spherical (hatted) to global Cartesian basis and
 * back.
 *
 */

/**
 * Vector conversion from hatted (spherical) to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_hat vector in hatted (spherical) basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */
Inline void v_Hat2Cart(const coord_t<D>&                xi,
                       const vec_t<Dimension::THREE_D>& vi_hat,
                       vec_t<Dimension::THREE_D>&       vi_cart) const {
  if constexpr (D == Dimension::ONE_D) {
    NTTError("spherical v_Hat2Cart not implemented for 1D");
  } else if constexpr (D == Dimension::TWO_D) {
    coord_t<D> x_sph;
    this->x_Code2Sph(xi, x_sph);
    vi_cart[0] = vi_hat[0] * math::sin(x_sph[1]) + vi_hat[1] * math::cos(x_sph[1]);
    vi_cart[1] = vi_hat[0] * math::cos(x_sph[1]) - vi_hat[1] * math::sin(x_sph[1]);
    vi_cart[2] = vi_hat[2];
  } else if constexpr (D == Dimension::THREE_D) {
    coord_t<D> x_sph;
    this->x_Code2Sph(xi, x_sph);
    vi_cart[0] = vi_hat[0] * math::sin(x_sph[1]) * math::cos(x_sph[2])
                 + vi_hat[1] * math::cos(x_sph[1]) * math::cos(x_sph[2])
                 - vi_hat[2] * math::sin(x_sph[2]);
    vi_cart[1] = vi_hat[0] * math::sin(x_sph[1]) * math::sin(x_sph[2])
                 + vi_hat[1] * math::cos(x_sph[1]) * math::sin(x_sph[2])
                 + vi_hat[2] * math::cos(x_sph[2]);
    vi_cart[2] = vi_hat[0] * math::cos(x_sph[1]) - vi_hat[1] * math::sin(x_sph[1]);
  }
}

/**
 * Vector conversion from hatted (spherical) to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_hat vector in hatted (spherical) basis (size of the array is 3).
 */
Inline void v_Cart2Hat(const coord_t<D>&                xi,
                       const vec_t<Dimension::THREE_D>& vi_cart,
                       vec_t<Dimension::THREE_D>&       vi_hat) const {
  if constexpr (D == Dimension::ONE_D) {
    NTTError("spherical v_Cart2Hat not implemented for 1D");
  } else if constexpr (D == Dimension::TWO_D) {
    coord_t<D> x_sph;
    this->x_Code2Sph(xi, x_sph);
    vi_hat[0] = vi_cart[0] * math::sin(x_sph[1]) + vi_cart[1] * math::cos(x_sph[1]);
    vi_hat[1] = vi_cart[0] * math::cos(x_sph[1]) - vi_cart[1] * math::sin(x_sph[1]);
    vi_hat[2] = vi_cart[2];
  } else if constexpr (D == Dimension::THREE_D) {
    vi_hat[0] = vi_cart[0] * math::sin(xi[1]) * math::cos(xi[2])
                + vi_cart[1] * math::sin(xi[1]) * math::sin(xi[2])
                + vi_cart[2] * math::cos(xi[1]);
    vi_hat[1] = vi_cart[0] * math::cos(xi[1]) * math::cos(xi[2])
                + vi_cart[1] * math::cos(xi[1]) * math::sin(xi[2])
                - vi_cart[2] * math::sin(xi[1]);
    vi_hat[2] = -vi_cart[0] * math::sin(xi[2]) + vi_cart[1] * math::cos(xi[2]);
  }
}

#endif