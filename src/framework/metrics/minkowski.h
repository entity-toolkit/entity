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
    auto findSmallestCell() const -> real_t { return dx / math::sqrt(static_cast<real_t>(D)); }

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
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "diag_vector_transform.h"

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cartesian physical units (size of the array is D).
     */
    Inline void x_Code2Cart(const coord_t<D>&, coord_t<D>&) const;
    /**
     * Coordinate conversion from Cartesian physical units to code units.
     *
     * @param x coordinate array in Cartesian coordinates in
     * physical units (size of the array is D).
     * @param xi coordinate array in code units (size of the array is D).
     */
    Inline void x_Cart2Code(const coord_t<D>&, coord_t<D>&) const;
    /**
     * Coordinate conversion from code units to Spherical physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Spherical coordinates in physical units (size of the array is D).
     */
    Inline void x_Code2Sph(const coord_t<D>&, coord_t<D>&) const;
    /**
     * Coordinate conversion from Spherical physical units to code units.
     *
     * @param xi coordinate array in Spherical coordinates in physical units (size of the array is D).
     * @param x coordinate array in code units (size of the array is D).
     */
    Inline void x_Sph2Code(const coord_t<D>&, coord_t<D>&) const;
  };

  // * * * * * * * * * * * * * * *
  // 1D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::ONE_D>::x_Code2Cart(const coord_t<Dimension::ONE_D>& xi,
                                                    coord_t<Dimension::ONE_D>&       x) const {
    x[0] = xi[0] * dx + this->x1_min;
  }
  template <>
  Inline void Metric<Dimension::ONE_D>::x_Cart2Code(const coord_t<Dimension::ONE_D>& x,
                                                    coord_t<Dimension::ONE_D>&       xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
  }
  template <>
  Inline void Metric<Dimension::ONE_D>::x_Code2Sph(const coord_t<Dimension::ONE_D>&, coord_t<Dimension::ONE_D>&) const {
  }
  template <>
  Inline void Metric<Dimension::ONE_D>::x_Sph2Code(const coord_t<Dimension::ONE_D>&, coord_t<Dimension::ONE_D>&) const {
  }

  // * * * * * * * * * * * * * * *
  // 2D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::TWO_D>::x_Code2Cart(const coord_t<Dimension::TWO_D>& xi,
                                                    coord_t<Dimension::TWO_D>&       x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
  }
  template <>
  Inline void Metric<Dimension::TWO_D>::x_Cart2Code(const coord_t<Dimension::TWO_D>& x,
                                                    coord_t<Dimension::TWO_D>&       xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
    xi[1] = (x[1] - this->x2_min) * inv_dx;
  }
  template <>
  Inline void Metric<Dimension::TWO_D>::x_Code2Sph(const coord_t<Dimension::TWO_D>& xi,
                                                   coord_t<Dimension::TWO_D>&       x) const {
    x_Code2Cart(xi, x);                          // convert to Cartesian coordinates
    x[0] = math::sqrt(x[0] * x[0] + x[1] * x[1]); // r = sqrt(x^2 + y^2)
    x[1] = math::atan2(x[1], x[0]);               // theta = atan(y/x)
  }
  template <>
  Inline void Metric<Dimension::TWO_D>::x_Sph2Code(const coord_t<Dimension::TWO_D>& x,
                                                   coord_t<Dimension::TWO_D>&       xi) const {
    xi[0] = x[0] * math::cos(x[1]); // x = r * cos(theta)
    xi[1] = x[0] * math::sin(x[1]); // y = r * sin(theta)
    x_Cart2Code(xi, xi);           // convert to code units
  }

  // * * * * * * * * * * * * * * *
  // 3D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::THREE_D>::x_Code2Cart(const coord_t<Dimension::THREE_D>& xi,
                                                      coord_t<Dimension::THREE_D>&       x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
    x[2] = xi[2] * dx + this->x3_min;
  }
  template <>
  Inline void Metric<Dimension::THREE_D>::x_Cart2Code(const coord_t<Dimension::THREE_D>& x,
                                                      coord_t<Dimension::THREE_D>&       xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
    xi[1] = (x[1] - this->x2_min) * inv_dx;
    xi[2] = (x[2] - this->x3_min) * inv_dx;
  }
  template <>
  Inline void Metric<Dimension::THREE_D>::x_Code2Sph(const coord_t<Dimension::THREE_D>& xi,
                                                     coord_t<Dimension::THREE_D>&       x) const {
    x_Code2Cart(xi, x);                                        // convert to Cartesian coordinates
    x[0] = math::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]); // r = sqrt(x^2 + y^2 + z^2)
    x[1] = math::atan2(x[1], x[0]);                             // theta = atan(y/x)
    x[2] = math::acos(x[2] / x[0]);                             // phi = acos(z/r)
  }
  template <>
  Inline void Metric<Dimension::THREE_D>::x_Sph2Code(const coord_t<Dimension::THREE_D>& x,
                                                     coord_t<Dimension::THREE_D>&       xi) const {
    xi[0] = x[0] * math::sin(x[1]) * math::cos(x[2]); // x = r * sin(theta) * cos(phi)
    xi[1] = x[0] * math::sin(x[1]) * math::sin(x[2]); // y = r * sin(theta) * sin(phi)
    xi[2] = x[0] * math::cos(x[1]);                  // z = r * cos(theta)
    x_Cart2Code(xi, xi);                            // convert to code units
  }

} // namespace ntt

#endif