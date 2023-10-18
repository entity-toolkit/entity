#ifndef FRAMEWORK_METRICS_UTILS_X_CODE_CART_FORGSPH_H
#define FRAMEWORK_METRICS_UTILS_X_CODE_CART_FORGSPH_H

/**
 * @brief Coordinate transformations for generalized spherical coordinates.
 * @implements x: Code -> Cart
 * @implements x: Cart -> Code
 */

/**
 * Coordinate conversion from code units to Cartesian physical units.
 * @param xi coordinate array in code units
 * @param x coordinate array in Cartesian physical units
 */
Inline void x_Code2Cart(const coord_t<Dim3>& xi, coord_t<Dim3>& x) const {
  coord_t<Dim3> x_sph { ZERO };
  x_Code2Sph(xi, x_sph);
  x[0] = x_sph[0] * math::sin(x_sph[1]) * math::cos(x_sph[2]);
  x[1] = x_sph[0] * math::sin(x_sph[1]) * math::sin(x_sph[2]);
  x[2] = x_sph[0] * math::cos(x_sph[1]);
}

/**
 * Coordinate conversion from Cartesian physical units to code units.
 * @param x coordinate array in Cartesian coordinates in physical units
 * @param xi coordinate array in code units
 */
Inline void x_Cart2Code(const coord_t<Dim3>& x, coord_t<Dim3>& xi) const {
  coord_t<Dim3> x_sph { ZERO };
  const real_t  rxy2 = SQR(x[0]) + SQR(x[1]);
  x_sph[0]           = math::sqrt(rxy2 + SQR(x[2]));
  x_sph[1]           = static_cast<real_t>(constant::HALF_PI) -
             math::atan2(x[2], math::sqrt(rxy2));
  x_sph[2] = static_cast<real_t>(constant::PI) - math::atan2(x[1], -x[0]);
  x_Sph2Code(x_sph, xi);
}

#endif // FRAMEWORK_METRICS_UTILS_X_CODE_CART_FORGSPH_H