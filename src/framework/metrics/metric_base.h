#ifndef FRAMEWORK_METRICS_BASE_H
#define FRAMEWORK_METRICS_BASE_H

#include "global.h"

#include <stdexcept>

/*
 *
 * Vector transformations
 *
 * Cntrv (A^mu)
 *   ^  ^
 *   |  |
 *   |  v
 *   | Hat <---> Cart (A_xyz)
 *   |  ^
 *   |  |
 *   v  v
 *   Cov (A_mu)
 *
 * Cntrv: contravariant vector
 * Cov: covariant vector
 * Hat: hatted (orthonormal) basis vector
 * Cart: global Cartesian basis vector (defined for diagonal only)
 *
 */

/*
 *
 * Coordinate transformations
 *
 *   +---> Cart
 *   |
 *   v
 * Code
 *   ^
 *   |
 *   +---> Sph
 * 
 * Code: coordinates in code units
 * Cart: coordinates in global Cartesian basis
 * Sph: coordinates in spherical basis
 * 
 */

namespace ntt {
  /**
   * Parent metric class: h_ij. Coordinates vary from `0` to `nx1` ... (code units).
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  struct MetricBase {
    // text label of the metric
    const std::string label;
    // max of coordinates in code units
    const real_t nx1, nx2, nx3;
    // extent in `x1` in physical units
    const real_t x1_min, x1_max;
    // extent in `x2` in physical units
    const real_t x2_min, x2_max;
    // extent in `x3` in physical units
    const real_t x3_min, x3_max;

    MetricBase(const std::string& label_, std::vector<unsigned int> resolution, std::vector<real_t> extent)
      : label {label_},
        nx1 {resolution.size() > 0 ? (real_t)(resolution[0]) : ONE},
        nx2 {resolution.size() > 1 ? (real_t)(resolution[1]) : ONE},
        nx3 {resolution.size() > 2 ? (real_t)(resolution[2]) : ONE},
        x1_min {resolution.size() > 0 ? extent[0] : ZERO},
        x1_max {resolution.size() > 0 ? extent[1] : ZERO},
        x2_min {resolution.size() > 1 ? extent[2] : ZERO},
        x2_max {resolution.size() > 1 ? extent[3] : ZERO},
        x3_min {resolution.size() > 2 ? extent[4] : ZERO},
        x3_max {resolution.size() > 2 ? extent[5] : ZERO} {}
    ~MetricBase() = default;

    /**
     * Convert `real_t` type code unit coordinate to cell index + displacement.
     *
     * TODO: `xi + N_GHOSTS` is a bit of a hack.
     * @returns A pair of `int` and `float`: cell index + displacement.
     */
    Inline auto CU_to_Idi(const real_t& xi) const -> std::pair<int, float> {
      auto  i {static_cast<int>(xi + N_GHOSTS) - N_GHOSTS};
      float di {static_cast<float>(xi) - static_cast<float>(i)};
      return {i, di};
    }
  };

} // namespace ntt

#endif
