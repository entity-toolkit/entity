#ifndef FRAMEWORK_METRICS_UTILS_V3_CART_HAT_CNTRV_COV_FORGSPH_H
#define FRAMEWORK_METRICS_UTILS_V3_CART_HAT_CNTRV_COV_FORGSPH_H

/**
 * @brief Vector transformations for generalized spherical.
 * @implements v3: Cntrv -> Cart
 * @implements v3: Cart -> Cntrv
 * @implements v3: Cov -> Cart
 * @implements v3: Cart -> Cov
 * @implements v3: Hat -> Cart
 * @implements v3: Cart -> Hat
 */

/**
 * Vector conversion from contravariant to global Cartesian basis.
 *
 * @param xi coordinate array in code units
 * @param vi_cntrv vector in contravariant basis
 * @param vi_cart vector in global Cartesian basis
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
 * @param xi coordinate array in code units
 * @param vi_cart vector in global Cartesian basis
 * @param vi_cntrv vector in contravariant basis
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
 * @param xi coordinate array in code units
 * @param vi_cov vector in covariant basis
 * @param vi_cart vector in global Cartesian basis
 */
Inline void v3_Cov2Cart(const coord_t<Dim3>& xi,
                        const vec_t<Dim3>&   vi_cov,
                        vec_t<Dim3>&         vi_cart) const {
  vec_t<Dim3> vi_hat { ZERO };
  if constexpr (D == Dim2) {
    this->v3_Cov2Hat({ xi[0], xi[1] }, vi_cov, vi_hat);
  } else {
    this->v3_Cov2Hat(xi, vi_cov, vi_hat);
  }
  this->v3_Hat2Cart(xi, vi_hat, vi_cart);
}

/**
 * Vector conversion from global Cartesian to covariant basis.
 *
 * @param xi coordinate array in code units
 * @param vi_cart vector in global Cartesian basis
 * @param vi_cov vector in covariant basis
 */
Inline void v3_Cart2Cov(const coord_t<Dim3>& xi,
                        const vec_t<Dim3>&   vi_cart,
                        vec_t<Dim3>&         vi_cov) const {
  vec_t<Dim3> vi_hat { ZERO };
  this->v3_Cart2Hat(xi, vi_cart, vi_hat);
  if constexpr (D == Dim2) {
    this->v3_Hat2Cov({ xi[0], xi[1] }, vi_hat, vi_cov);
  } else {
    this->v3_Hat2Cov(xi, vi_hat, vi_cov);
  }
}

/**
 * @brief Vector transformations from global Spherical (hatted) to global
 * Cartesian basis and back.
 *
 */

/**
 * Vector conversion from hatted (spherical) to global Cartesian basis.
 *
 * @param xi coordinate array in code units
 * @param vi_hat vector in hatted (spherical) basis
 * @param vi_cart vector in global Cartesian basis
 */
Inline void v3_Hat2Cart(const coord_t<Dim3>& xi,
                        const vec_t<Dim3>&   vi_hat,
                        vec_t<Dim3>&         vi_cart) const {
  coord_t<Dim3> x_sph;
  this->x_Code2Sph(xi, x_sph);
  if constexpr (D != Dim1) {
    vi_cart[0] = vi_hat[0] * math::sin(x_sph[1]) * math::cos(x_sph[2]) +
                 vi_hat[1] * math::cos(x_sph[1]) * math::cos(x_sph[2]) -
                 vi_hat[2] * math::sin(x_sph[2]);
    vi_cart[1] = vi_hat[0] * math::sin(x_sph[1]) * math::sin(x_sph[2]) +
                 vi_hat[1] * math::cos(x_sph[1]) * math::sin(x_sph[2]) +
                 vi_hat[2] * math::cos(x_sph[2]);
    vi_cart[2] = vi_hat[0] * math::cos(x_sph[1]) - vi_hat[1] * math::sin(x_sph[1]);
  }
}

/**
 * Vector conversion global Cartesian to hatted (spherical) basis.
 *
 * @param xi coordinate array in code units
 * @param vi_cart vector in global Cartesian basis
 * @param vi_hat vector in hatted (spherical) basis
 */
Inline void v3_Cart2Hat(const coord_t<Dim3>& xi,
                        const vec_t<Dim3>&   vi_cart,
                        vec_t<Dim3>&         vi_hat) const {
  coord_t<Dim3> x_sph;
  this->x_Code2Sph(xi, x_sph);
  if constexpr (D != Dim1) {
    vi_hat[0] = vi_cart[0] * math::sin(x_sph[1]) * math::cos(x_sph[2]) +
                vi_cart[1] * math::sin(x_sph[1]) * math::sin(x_sph[2]) +
                vi_cart[2] * math::cos(x_sph[1]);
    vi_hat[1] = vi_cart[0] * math::cos(x_sph[1]) * math::cos(x_sph[2]) +
                vi_cart[1] * math::cos(x_sph[1]) * math::sin(x_sph[2]) -
                vi_cart[2] * math::sin(x_sph[1]);
    vi_hat[2] = -vi_cart[0] * math::sin(x_sph[2]) +
                vi_cart[1] * math::cos(x_sph[2]);
  }
}

#endif // FRAMEWORK_METRICS_UTILS_V3_CART_HAT_CNTRV_COV_FORGSPH_H