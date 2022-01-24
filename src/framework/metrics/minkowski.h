#ifndef FRAMEWORK_METRICS_MINKOWSKI_H
#define FRAMEWORK_METRICS_MINKOWSKI_H

#include "global.h"
#include "metric_base.h"

#include <cmath>

namespace ntt {
  /**
   * Metric metric (cartesian system): diag(-1, 1, 1, 1).
   * Cell sizes in each direction dx1 = dx2 = dx3 are equal.
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Metric : public MetricBase<D> {
  private:
    const real_t dx, dx_sqr, inv_dx;

  public:
    Metric(std::vector<unsigned int> resolution, std::vector<real_t> extent, const real_t*)
      : MetricBase<D> {"minkowski", resolution, extent},
        dx((this->x1_max - this->x1_min) / this->nx1),
        dx_sqr(dx * dx),
        inv_dx(ONE / dx) {}
    ~Metric() = default;

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
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
    Inline auto sqrt_det_h(const coord_t<D>&) const -> real_t { return dx_sqr * dx; }

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cartesian coordinates in physical units (size of the array is D).
     */
    Inline void x_Code2Cart(const coord_t<D>&, coord_t<D>&) const;
    /**
     * Vector conversion from hatted to contravariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     * @param vi vector in contravariant basis (size of the array is 3).
     */
    Inline void v_Hat2Cntrv(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from contravariant to hatted basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi vector in contravariant basis (size of the array is 3).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     */
    Inline void v_Cntrv2Hat(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;

    /**
     * Coordinate conversion from code units to Spherical physical units.
     * @todo Actually implement.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cpherical coordinates in physical units (size of the array is D).
     */
    Inline void x_Code2Sph(const coord_t<D>&, coord_t<D>&) const {};
  };

  // * * * * * * * * * * * * * * *
  // vector transformations
  // * * * * * * * * * * * * * * *
  template <Dimension D>
  Inline void Metric<D>::v_Hat2Cntrv(const coord_t<D>& xi,
                                     const vec_t<Dimension::THREE_D>& vi_hat,
                                     vec_t<Dimension::THREE_D>& vi) const {
    vi[0] = vi_hat[0] / std::sqrt(h_11(xi));
    vi[1] = vi_hat[1] / std::sqrt(h_22(xi));
    vi[2] = vi_hat[2] / std::sqrt(h_33(xi));
  }

  template <Dimension D>
  Inline void Metric<D>::v_Cntrv2Hat(const coord_t<D>& xi,
                                     const vec_t<Dimension::THREE_D>& vi,
                                     vec_t<Dimension::THREE_D>& vi_hat) const {
    vi_hat[0] = vi[0] * std::sqrt(h_11(xi));
    vi_hat[1] = vi[1] * std::sqrt(h_22(xi));
    vi_hat[2] = vi[2] * std::sqrt(h_33(xi));
  }

  // * * * * * * * * * * * * * * *
  // 1D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::ONE_D>::x_Code2Cart(const coord_t<Dimension::ONE_D>& xi,
                                                    coord_t<Dimension::ONE_D>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
  }

  // * * * * * * * * * * * * * * *
  // 2D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::TWO_D>::x_Code2Cart(const coord_t<Dimension::TWO_D>& xi,
                                                    coord_t<Dimension::TWO_D>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
  }

  // * * * * * * * * * * * * * * *
  // 3D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::THREE_D>::x_Code2Cart(const coord_t<Dimension::THREE_D>& xi,
                                                      coord_t<Dimension::THREE_D>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
    x[2] = xi[2] * dx + this->x3_min;
  }

} // namespace ntt

#endif
