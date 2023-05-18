#ifndef FRAMEWORK_METRICS_SPH_COMMON_H
#define FRAMEWORK_METRICS_SPH_COMMON_H

#ifdef __INTELLISENSE__
#  pragma diag_suppress 1670
#  pragma diag_suppress 864
#  pragma diag_suppress 258
#  pragma diag_suppress 65
#  pragma diag_suppress 20
#endif

/**
 * @brief Vector transformations from and to global Cartesian basis.
 *
 *
 * @note Functions `v3_Hat2Cart` and `v3_Cart2Hat` should be implemented per each metric.
 *
 */

/**
 * Vector conversion from contravariant to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */
Inline void v3_Cntrv2Cart(const coord_t<Dim3>& xi,
                          const vec_t<Dim3>&   vi_cntrv,
                          vec_t<Dim3>&         vi_cart) const {
  if constexpr (D == Dim2) {
    vec_t<Dim3> vi_hat { ZERO };
    this->v3_Cntrv2Hat({ xi[0], xi[1] }, vi_cntrv, vi_hat);
    this->v3_Hat2Cart(xi, vi_hat, vi_cart);
  } else if constexpr (D == Dim3) {
    vec_t<Dim3> vi_hat { ZERO };
    this->v3_Cntrv2Hat(xi, vi_cntrv, vi_hat);
    this->v3_Hat2Cart(xi, vi_hat, vi_cart);
  }
}

/**
 * Vector conversion from global Cartesian to contravariant basis.
 *
 * @param xi coordinate array in code units (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_cntrv vector in contravariant basis (size of the array is 3).
 */
Inline void v3_Cart2Cntrv(const coord_t<Dim3>& xi,
                          const vec_t<Dim3>&   vi_cart,
                          vec_t<Dim3>&         vi_cntrv) const {
  if constexpr (D == Dim2) {
    vec_t<Dim3> vi_hat { ZERO };
    this->v3_Cart2Hat(xi, vi_cart, vi_hat);
    this->v3_Hat2Cntrv({ xi[0], xi[1] }, vi_hat, vi_cntrv);
  } else if constexpr (D == Dim3) {
    vec_t<Dim3> vi_hat { ZERO };
    this->v3_Cart2Hat(xi, vi_cart, vi_hat);
    this->v3_Hat2Cntrv(xi, vi_hat, vi_cntrv);
  }
}

/**
 * Vector conversion from covariant to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */
Inline void v3_Cov2Cart(const coord_t<Dim3>& xi,
                        const vec_t<Dim3>&   vi_cov,
                        vec_t<Dim3>&         vi_cart) const {
  vec_t<Dim3> vi_hat { ZERO };
  this->v3_Cov2Hat(xi, vi_cov, vi_hat);
  this->v3_Hat2Cart(xi, vi_hat, vi_cart);
}

/**
 * Vector conversion from global Cartesian to covariant basis.
 *
 * @param xi coordinate array in code units (size of the array is D).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_cov vector in covariant basis (size of the array is 3).
 */
Inline void v3_Cart2Cov(const coord_t<Dim3>& xi,
                        const vec_t<Dim3>&   vi_cart,
                        vec_t<Dim3>&         vi_cov) const {
  vec_t<Dim3> vi_hat { ZERO };
  this->v3_Cart2Hat(xi, vi_cart, vi_hat);
  this->v3_Hat2Cov(xi, vi_hat, vi_cov);
}

/**
 * @brief Vector transformations from global Spherical (hatted) to global Cartesian basis and
 * back.
 *
 */

/**
 * Vector conversion from hatted (spherical) to global Cartesian basis.
 *
 * @param xi coordinate array in code units (size of the array is 3).
 * @param vi_hat vector in hatted (spherical) basis (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 */
Inline void v3_Hat2Cart(const coord_t<Dim3>& xi,
                        const vec_t<Dim3>&   vi_hat,
                        vec_t<Dim3>&         vi_cart) const {
  coord_t<Dim3> x_sph;
  if constexpr (D == Dim2) {
    coord_t<Dim2> x_sph_2d;
    this->x_Code2Sph({ xi[0], xi[1] }, x_sph_2d);
    x_sph[0] = x_sph_2d[0];
    x_sph[1] = x_sph_2d[1];
    x_sph[2] = xi[2];
  } else if constexpr (D == Dim3) {
    this->x_Code2Sph(xi, x_sph);
  }
  if constexpr (D != Dim1) {
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
 * Vector conversion global Cartesian to hatted (spherical) basis.
 *
 * @param xi coordinate array in code units (size of the array is 3).
 * @param vi_cart vector in global Cartesian basis (size of the array is 3).
 * @param vi_hat vector in hatted (spherical) basis (size of the array is 3).
 */
Inline void v3_Cart2Hat(const coord_t<Dim3>& xi,
                        const vec_t<Dim3>&   vi_cart,
                        vec_t<Dim3>&         vi_hat) const {
  coord_t<Dim3> x_sph;
  if constexpr (D == Dim2) {
    coord_t<Dim2> x_sph_2d;
    this->x_Code2Sph({ xi[0], xi[1] }, x_sph_2d);
    x_sph[0] = x_sph_2d[0];
    x_sph[1] = x_sph_2d[1];
    x_sph[2] = xi[2];
  } else if constexpr (D == Dim3) {
    this->x_Code2Sph(xi, x_sph);
  }
  if constexpr (D != Dim1) {
    vi_hat[0] = vi_cart[0] * math::sin(x_sph[1]) * math::cos(x_sph[2])
                + vi_cart[1] * math::sin(x_sph[1]) * math::sin(x_sph[2])
                + vi_cart[2] * math::cos(x_sph[1]);
    vi_hat[1] = vi_cart[0] * math::cos(x_sph[1]) * math::cos(x_sph[2])
                + vi_cart[1] * math::cos(x_sph[1]) * math::sin(x_sph[2])
                - vi_cart[2] * math::sin(x_sph[1]);
    vi_hat[2] = -vi_cart[0] * math::sin(x_sph[2]) + vi_cart[1] * math::cos(x_sph[2]);
  }
}

#ifdef __INTELLISENSE__
#  pragma diag_default 20
#  pragma diag_default 65
#  pragma diag_default 258
#  pragma diag_default 864
#  pragma diag_default 1670
#endif

#endif