#ifndef FRAMEWORK_METRICS_UTILS_X_CODE_PHYS_FORGSPH_H
#define FRAMEWORK_METRICS_UTILS_X_CODE_PHYS_FORGSPH_H

/**
 * @brief Coordinate transformations for general spherical.
 * @implements x: Code -> Phys
 * @implements x: Phys -> Code
 */

/**
 * Coordinate conversion from code units to Spherical physical units.
 *
 * @param xi coordinate array in code units
 * @param x coordinate array in Spherical coordinates in physical units
 */
Inline void x_Code2Phys(const coord_t<FullD>& xi, coord_t<FullD>& x) const {
  this->x_Code2Sph(xi, x);
}

Inline auto x1_Code2Phys(const real_t& x1) const -> real_t {
  return this->x1_Code2Sph(x1);
}

Inline auto x2_Code2Phys(const real_t& x2) const -> real_t {
  return this->x2_Code2Sph(x2);
}

Inline auto x3_Code2Phys(const real_t& x3) const -> real_t {
  return this->x3_Code2Sph(x3);
}

/**
 * Coordinate conversion from Spherical physical units to code units.
 *
 * @param x coordinate array in Spherical coordinates in physical units
 * @param xi coordinate array in code units
 */
Inline void x_Phys2Code(const coord_t<FullD>& x, coord_t<FullD>& xi) const {
  this->x_Sph2Code(x, xi);
}

Inline auto x1_Phys2Code(const real_t& r) const -> real_t {
  return this->x1_Sph2Code(r);
}

Inline auto x2_Phys2Code(const real_t& th) const -> real_t {
  return this->x2_Sph2Code(th);
}

Inline auto x3_Phys2Code(const real_t& phi) const -> real_t {
  return this->x3_Sph2Code(phi);
}

#endif // FRAMEWORK_METRICS_UTILS_X_CODE_PHYS_FORGSPH_H
