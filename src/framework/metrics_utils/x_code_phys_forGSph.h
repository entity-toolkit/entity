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
Inline void x_Code2Phys(const coord_t<D>& xi, coord_t<D>& x) const {
  this->x_Code2Sph(xi, x);
}

/**
 * Coordinate conversion from Spherical physical units to code units.
 *
 * @param x coordinate array in Spherical coordinates in physical units
 * @param xi coordinate array in code units
 */
Inline void x_Phys2Code(const coord_t<D>& x, coord_t<D>& xi) const {
  this->x_Sph2Code(x, xi);
}

#endif    // FRAMEWORK_METRICS_UTILS_X_CODE_PHYS_FORGSPH_H