#ifndef FRAMEWORK_METRICS_MINKOWSKI_H
#define FRAMEWORK_METRICS_MINKOWSKI_H

#include "wrapper.h"

#include "metric_base.h"

#include <cmath>

namespace ntt {
  /**
   * Flat metric (cartesian system): diag(-1, 1, 1, 1).
   * Cell sizes in each direction dx1 = dx2 = dx3 are equal.
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Metric : public MetricBase<D> {
  private:
    const real_t dx, dx_sqr, inv_dx;

  public:
    const real_t dx_min;

    Metric(std::vector<unsigned int> resolution, std::vector<real_t> extent, const real_t*)
      : MetricBase<D> { "minkowski", resolution, extent },
        dx((this->x1_max - this->x1_min) / this->nx1),
        dx_sqr(dx * dx),
        inv_dx(ONE / dx),
        dx_min { findSmallestCell() } {}
    ~Metric() = default;

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    auto findSmallestCell() const -> real_t {
      return dx / math::sqrt(static_cast<real_t>(D));
    }

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>&) const -> real_t {
      return dx_sqr;
    }
    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>&) const -> real_t {
      return dx_sqr;
    }
    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>&) const -> real_t {
      return dx_sqr;
    }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h(const coord_t<D>&) const -> real_t {
      return math::pow(dx, static_cast<short>(D));
    }

    /**
     * Compute the fiducial minimum cell volume.
     *
     * @returns Minimum cell volume of the grid [code units].
     */
    Inline auto min_cell_volume() const -> real_t {
      return math::pow(dx, static_cast<short>(D));
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/sr_common.h"

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
     * @param x coordinate array in Spherical coordinates in physical units (size of the array
     * is D).
     */
    Inline void x_Code2Sph(const coord_t<D>&, coord_t<D>&) const;
    /**
     * Coordinate conversion from Spherical physical units to code units.
     *
     * @param xi coordinate array in Spherical coordinates in physical units (size of the array
     * is D).
     * @param x coordinate array in code units (size of the array is D).
     */
    Inline void x_Sph2Code(const coord_t<D>&, coord_t<D>&) const;

    /**
     * Vector conversion from contravariant to global Cartesian basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     * @param vi_cart vector in global Cartesian basis (size of the array is 3).
     */
    Inline void v3_Cntrv2Cart(const coord_t<D>&  xi,
                             const vec_t<Dim3>& vi_cntrv,
                             vec_t<Dim3>&       vi_cart) const {
      this->v3_Cntrv2Hat(xi, vi_cntrv, vi_cart);
    }

    /**
     * Vector conversion from global Cartesian to contravariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cart vector in global Cartesian basis (size of the array is 3).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     */
    Inline void v3_Cart2Cntrv(const coord_t<D>&  xi,
                             const vec_t<Dim3>& vi_cart,
                             vec_t<Dim3>&       vi_cntrv) const {
      this->v3_Hat2Cntrv(xi, vi_cart, vi_cntrv);
    }

    /**
     * Vector conversion from covariant to global Cartesian basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cov vector in covariant basis (size of the array is 3).
     * @param vi_cart vector in global Cartesian basis (size of the array is 3).
     */
    Inline void v3_Cov2Cart(const coord_t<D>&  xi,
                           const vec_t<Dim3>& vi_cov,
                           vec_t<Dim3>&       vi_cart) const {
      this->v3_Cov2Hat(xi, vi_cov, vi_cart);
    }

    /**
     * Vector conversion from global Cartesian to covariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cart vector in global Cartesian basis (size of the array is 3).
     * @param vi_cov vector in covariant basis (size of the array is 3).
     */
    Inline void v3_Cart2Cov(const coord_t<D>&  xi,
                           const vec_t<Dim3>& vi_cart,
                           vec_t<Dim3>&       vi_cov) const {
      this->v3_Hat2Cov(xi, vi_cart, vi_cov);
    }
  };

  // * * * * * * * * * * * * * * *
  // 1D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dim1>::x_Code2Cart(const coord_t<Dim1>& xi, coord_t<Dim1>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
  }
  template <>
  Inline void Metric<Dim1>::x_Cart2Code(const coord_t<Dim1>& x, coord_t<Dim1>& xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
  }
  template <>
  Inline void Metric<Dim1>::x_Code2Sph(const coord_t<Dim1>&, coord_t<Dim1>&) const {}
  template <>
  Inline void Metric<Dim1>::x_Sph2Code(const coord_t<Dim1>&, coord_t<Dim1>&) const {}

  // * * * * * * * * * * * * * * *
  // 2D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dim2>::x_Code2Cart(const coord_t<Dim2>& xi, coord_t<Dim2>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
  }
  template <>
  Inline void Metric<Dim2>::x_Cart2Code(const coord_t<Dim2>& x, coord_t<Dim2>& xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
    xi[1] = (x[1] - this->x2_min) * inv_dx;
  }
  template <>
  Inline void Metric<Dim2>::x_Code2Sph(const coord_t<Dim2>& xi, coord_t<Dim2>& x) const {
    x_Code2Cart(xi, x);                              // convert to Cartesian coordinates
    x[0] = math::sqrt(x[0] * x[0] + x[1] * x[1]);    // r = sqrt(x^2 + y^2)
    x[1] = math::atan2(x[1], x[0]);                  // theta = atan(y/x)
  }
  template <>
  Inline void Metric<Dim2>::x_Sph2Code(const coord_t<Dim2>& x, coord_t<Dim2>& xi) const {
    xi[0] = x[0] * math::cos(x[1]);    // x = r * cos(theta)
    xi[1] = x[0] * math::sin(x[1]);    // y = r * sin(theta)
    x_Cart2Code(xi, xi);               // convert to code units
  }

  // * * * * * * * * * * * * * * *
  // 3D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dim3>::x_Code2Cart(const coord_t<Dim3>& xi, coord_t<Dim3>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
    x[2] = xi[2] * dx + this->x3_min;
  }
  template <>
  Inline void Metric<Dim3>::x_Cart2Code(const coord_t<Dim3>& x, coord_t<Dim3>& xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
    xi[1] = (x[1] - this->x2_min) * inv_dx;
    xi[2] = (x[2] - this->x3_min) * inv_dx;
  }
  template <>
  Inline void Metric<Dim3>::x_Code2Sph(const coord_t<Dim3>& xi, coord_t<Dim3>& x) const {
    x_Code2Cart(xi, x);    // convert to Cartesian coordinates
    x[0] = math::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);    // r = sqrt(x^2 + y^2 + z^2)
    x[1] = math::atan2(x[1], x[0]);                                // theta = atan(y/x)
    x[2] = math::acos(x[2] / x[0]);                                // phi = acos(z/r)
  }
  template <>
  Inline void Metric<Dim3>::x_Sph2Code(const coord_t<Dim3>& x, coord_t<Dim3>& xi) const {
    xi[0] = x[0] * math::sin(x[1]) * math::cos(x[2]);    // x = r * sin(theta) * cos(phi)
    xi[1] = x[0] * math::sin(x[1]) * math::sin(x[2]);    // y = r * sin(theta) * sin(phi)
    xi[2] = x[0] * math::cos(x[1]);                      // z = r * cos(theta)
    x_Cart2Code(xi, xi);                                 // convert to code units
  }

}    // namespace ntt

#endif