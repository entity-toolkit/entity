#ifndef FRAMEWORK_METRICS_MINKOWSKI_H
#define FRAMEWORK_METRICS_MINKOWSKI_H

#include "global.h"
#include "metric.h"

#include <tuple>

namespace ntt {
  /**
   * Minkowski metric (cartesian system): diag(-1, 1, 1, 1).
   * Cell sizes in each direction dx1 = dx2 = dx3 are equal.
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Minkowski : public Metric<D> {
  private:
    const real_t dx, dx_sqr, inv_dx;

  public:
    Minkowski(std::vector<std::size_t> resolution, std::vector<real_t> extent)
      : Metric<D> {"minkowski", resolution, extent},
        dx((this->x1_max - this->x1_min) / this->nx1),
        dx_sqr(dx * dx),
        inv_dx(ONE / dx) {}
    ~Minkowski() = default;

    /**
     * Compute minimum effective cell size for Minkowski metric: `dx / sqrt(D)` (in physical units).
     *
     * @returns Minimum cell size [physical units].
     */
    auto findSmallestCell() const -> real_t { return dx / std::sqrt(static_cast<real_t>(D)); }

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>&) const -> real_t { return dx_sqr; }
    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>&) const -> real_t { return dx_sqr; }
    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>&) const -> real_t { return dx_sqr; }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns sqrt(det(h_ij)).
     */
    virtual Inline auto sqrt_det_h(const coord_t<D>&) const -> real_t { return dx_sqr * dx; }
  };

} // namespace ntt

#endif
