#ifndef FRAMEWORK_METRICS_KS_PHYS_UNITS_H
#define FRAMEWORK_METRICS_KS_PHYS_UNITS_H

#ifdef __INTELLISENSE__
#  pragma diag_suppress 77
#  pragma diag_suppress 65
#endif

/**
 * Compute metric component 11 in physical coordinate basis.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h_11 (covariant, lower index) metric component.
 */
Inline auto h_11_phys(const coord_t<D>& x) const -> real_t {
  real_t r { x[0] * dr + this->x1_min };
  real_t theta { x[1] * dtheta };
  real_t cth { math::cos(theta) };
  return (ONE + TWO * r / (SQR(r) + a_sqr * SQR(cth)));
}

/**
 * Compute metric component 22 in physical coordinate basis.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h_22 (covariant, lower index) metric component.
 */
Inline auto h_22_phys(const coord_t<D>& x) const -> real_t {
  real_t r { x[0] * dr + this->x1_min };
  real_t theta { x[1] * dtheta };
  real_t cth { math::cos(theta) };
  return (SQR(r) + a_sqr * SQR(cth));
}

/**
 * Compute metric component 33 in physical coordinate basis.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h_33 (covariant, lower index) metric component.
 */
Inline auto h_33_phys(const coord_t<D>& x) const -> real_t {
  real_t r { x[0] * dr + this->x1_min };
  real_t theta { x[1] * dtheta };
  real_t cth { math::cos(theta) };
  real_t sth { math::sin(theta) };

  real_t delta { SQR(r) - TWO * r + a_sqr };
  real_t As { (SQR(r) + a_sqr) * (SQR(r) + a_sqr) - a_sqr * delta * SQR(sth) };
  return As * SQR(sth) / (SQR(r) + a_sqr * SQR(cth));
}

/**
 * Compute metric component 13 in physical coordinate basis.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h_13 (covariant, lower index) metric component.
 */
Inline auto h_13_phys(const coord_t<D>& x) const -> real_t {
  real_t r { x[0] * dr + this->x1_min };
  real_t theta { x[1] * dtheta };
  real_t sth { math::sin(theta) };
  return -a * SQR(sth) * (ONE + TWO * r / (SQR(r) + a_sqr * SQR(cth)));
}

/**
 * Compute inverse metric component 11 from h_ij in physical coordinate basis.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h^11 (contravariant, upper index) metric component.
 */
Inline auto h11_phys(const coord_t<D>& x) const -> real_t {
  return h_33_phys(x) / (h_11_phys(x) * h_33_phys(x) - SQR(h_13_phys(x)));
}

/**
 * Compute inverse metric component 22 from h_ij in physical coordinate basis.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h^22 (contravariant, upper index) metric component.
 */
Inline auto h22_phys(const coord_t<D>& x) const -> real_t {
  return ONE / h_22_phys(x);
}

/**
 * Compute inverse metric component 33 from h_ij in physical coordinate basis.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h^33 (contravariant, upper index) metric component.
 */
Inline auto h33_phys(const coord_t<D>& x) const -> real_t {
  return h_11_phys(x) / (h_11_phys(x) * h_33_phys(x) - SQR(h_13_phys(x)));
}

/**
 * Compute inverse metric component 13 from h_ij in physical coordinate basis.
 *
 * @param x coordinate array in code units (size of the array is D).
 * @returns h^13 (contravariant, upper index) metric component.
 */
Inline auto h13_phys(const coord_t<D>& x) const -> real_t {
  return -h_13_phys(x) / (h_11_phys(x) * h_33_phys(x) - SQR(h_13_phys(x)));
}

#ifdef __INTELLISENSE__
#  pragma diag_default 65
#  pragma diag_default 77
#endif

#endif    // FRAMEWORK_METRICS_KS_PHYS_UNITS_H