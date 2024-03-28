#ifndef FRAMEWORK_METRICS_MINKOWSKI_H
#define FRAMEWORK_METRICS_MINKOWSKI_H

#include "wrapper.h"

#include "metric_base.h"

#include "utilities/qmath.h"

#include <cmath>
#include <map>

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
    const real_t dx, dx_sqr, dx_inv;

  public:
    constexpr static Dimension PrtlD = D;

    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const std::map<std::string, real_t>&) :
      MetricBase<D> { "minkowski", CoordinateSystem::Cart, resolution, extent },
      dx((this->x1_max - this->x1_min) / this->nx1),
      dx_sqr(dx * dx),
      dx_inv(ONE / dx) {
      this->set_dxMin(find_dxMin());
      if constexpr (D == Dim2) {
        NTTHostErrorIf(
          !AlmostEqual((this->x2_max - this->x2_min) / (real_t)(this->nx2), dx),
          "dx2 must be equal to dx1 in 2D");
      } else if constexpr (D == Dim3) {
        NTTHostErrorIf(
          !AlmostEqual((this->x2_max - this->x2_min) / (real_t)(this->nx2), dx),
          "dx2 must be equal to dx1 in 3D");
        NTTHostErrorIf(
          !AlmostEqual((this->x3_max - this->x3_min) / (real_t)(this->nx3), dx),
          "dx3 must be equal to dx1 in 3D");
      }
    }

    ~Metric() = default;

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      return dx / math::sqrt(static_cast<real_t>(D));
    }

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>&) const -> real_t {
      return dx_sqr;
    }

    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>&) const -> real_t {
      return dx_sqr;
    }

    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>&) const -> real_t {
      return dx_sqr;
    }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h(const coord_t<D>&) const -> real_t {
      return math::pow(dx, static_cast<short>(D));
    }

    /**
     * Compute the square root of the determinant of h-matrix.
     * @note for compatibility purposes
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>&) const -> real_t {
      return math::pow(dx, static_cast<short>(D));
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/param_forSR.h"

#include "metrics_utils/v3_hat_cntrv_cov_forSR.h"

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units
     * @param x coordinate array in Cartesian physical units
     */
    Inline void x_Code2Cart(const coord_t<D>&, coord_t<D>&) const;
    /**
     * Coordinate conversion from Cartesian physical units to code units.
     *
     * @param x coordinate array in Cartesian coordinates in
     * physical units
     * @param xi coordinate array in code units
     */
    Inline void x_Cart2Code(const coord_t<D>&, coord_t<D>&) const;

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units
     * @param x coordinate array in Cartesian physical units
     */
    Inline void x_Code2Phys(const coord_t<D>& xi, coord_t<D>& x) const {
      this->x_Code2Cart(xi, x);
    }

    /**
     * Coordinate conversion from Cartesian physical units to code units.
     *
     * @param x coordinate array in Cartesian coordinates in physical units
     * @param xi coordinate array in code units
     */
    Inline void x_Phys2Code(const coord_t<D>& x, coord_t<D>& xi) const {
      this->x_Cart2Code(x, xi);
    }

    /**
     * Coordinate conversion from code units to Spherical physical units.
     *
     * @param xi coordinate array in code units
     * @param x coordinate array in Spherical coordinates in physical units
     */
    Inline void x_Code2Sph(const coord_t<D>&, coord_t<D>&) const;
    /**
     * Coordinate conversion from Spherical physical units to code units.
     *
     * @param xi coordinate array in Spherical coordinates in physical units
     * @param x coordinate array in code units
     */
    Inline void x_Sph2Code(const coord_t<D>&, coord_t<D>&) const;

    /**
     * Vector conversion from contravariant to global Cartesian basis.
     *
     * @param xi coordinate array in code units
     * @param vi_cntrv vector in contravariant basis
     * @param vi_cart vector in global Cartesian basis
     */
    Inline void v3_Cntrv2Cart(const coord_t<D>&  xi,
                              const vec_t<Dim3>& vi_cntrv,
                              vec_t<Dim3>&       vi_cart) const {
      this->v3_Cntrv2Hat(xi, vi_cntrv, vi_cart);
    }

    /**
     * Vector conversion from global Cartesian to contravariant basis.
     *
     * @param xi coordinate array in code units
     * @param vi_cart vector in global Cartesian basis
     * @param vi_cntrv vector in contravariant basis
     */
    Inline void v3_Cart2Cntrv(const coord_t<D>&  xi,
                              const vec_t<Dim3>& vi_cart,
                              vec_t<Dim3>&       vi_cntrv) const {
      this->v3_Hat2Cntrv(xi, vi_cart, vi_cntrv);
    }

    /**
     * Vector conversion from covariant to global Cartesian basis.
     *
     * @param xi coordinate array in code units
     * @param vi_cov vector in covariant basis
     * @param vi_cart vector in global Cartesian basis
     */
    Inline void v3_Cov2Cart(const coord_t<D>&  xi,
                            const vec_t<Dim3>& vi_cov,
                            vec_t<Dim3>&       vi_cart) const {
      this->v3_Cov2Hat(xi, vi_cov, vi_cart);
    }

    /**
     * Vector conversion from global Cartesian to covariant basis.
     *
     * @param xi coordinate array in code units
     * @param vi_cart vector in global Cartesian basis
     * @param vi_cov vector in covariant basis
     */
    Inline void v3_Cart2Cov(const coord_t<D>&  xi,
                            const vec_t<Dim3>& vi_cart,
                            vec_t<Dim3>&       vi_cov) const {
      this->v3_Hat2Cov(xi, vi_cart, vi_cov);
    }

    /**
     * Vector conversion from contravariant to physical contravariant.
     *
     * @param xi coordinate array in code units
     * @param vi_cntrv vector in contravariant basis
     * @param v_cntrv vector in physical contravariant basis
     */
    Inline void v3_Cntrv2PhysCntrv(const coord_t<D>&,
                                   const vec_t<Dim3>& vi_cntrv,
                                   vec_t<Dim3>&       v_cntrv) const {
      if constexpr (D == Dim1) {
        v_cntrv[0] = vi_cntrv[0] * dx;
        v_cntrv[1] = vi_cntrv[1];
        v_cntrv[2] = vi_cntrv[2];
      } else if constexpr (D == Dim2) {
        v_cntrv[0] = vi_cntrv[0] * dx;
        v_cntrv[1] = vi_cntrv[1] * dx;
        v_cntrv[2] = vi_cntrv[2];
      } else {
        v_cntrv[0] = vi_cntrv[0] * dx;
        v_cntrv[1] = vi_cntrv[1] * dx;
        v_cntrv[2] = vi_cntrv[2] * dx;
      }
    }

    /**
     * Vector conversion from physical contravariant to contravariant.
     *
     * @param xi coordinate array in code units
     * @param v_cntrv vector in physical contravariant basis
     * @param vi_cntrv vector in contravariant basis
     */
    Inline void v3_PhysCntrv2Cntrv(const coord_t<D>&,
                                   const vec_t<Dim3>& v_cntrv,
                                   vec_t<Dim3>&       vi_cntrv) const {
      if constexpr (D == Dim1) {
        vi_cntrv[0] = v_cntrv[0] * dx_inv;
        vi_cntrv[1] = v_cntrv[1];
        vi_cntrv[2] = v_cntrv[2];
      } else if constexpr (D == Dim2) {
        vi_cntrv[0] = v_cntrv[0] * dx_inv;
        vi_cntrv[1] = v_cntrv[1] * dx_inv;
        vi_cntrv[2] = v_cntrv[2];
      } else {
        vi_cntrv[0] = v_cntrv[0] * dx_inv;
        vi_cntrv[1] = v_cntrv[1] * dx_inv;
        vi_cntrv[2] = v_cntrv[2] * dx_inv;
      }
    }

    /**
     * Vector conversion from covariant to physical covariant.
     *
     * @param xi coordinate array in code units
     * @param vi_cov vector in covariant basis
     * @param v_cov vector in physical covariant basis
     */
    Inline void v3_Cov2PhysCov(const coord_t<D>&,
                               const vec_t<Dim3>& vi_cov,
                               vec_t<Dim3>&       v_cov) const {
      if constexpr (D == Dim1) {
        v_cov[0] = vi_cov[0] * dx_inv;
        v_cov[1] = vi_cov[1];
        v_cov[2] = vi_cov[2];
      } else if constexpr (D == Dim2) {
        v_cov[0] = vi_cov[0] * dx_inv;
        v_cov[1] = vi_cov[1] * dx_inv;
        v_cov[2] = vi_cov[2];
      } else {
        v_cov[0] = vi_cov[0] * dx_inv;
        v_cov[1] = vi_cov[1] * dx_inv;
        v_cov[2] = vi_cov[2] * dx_inv;
      }
    }

    /**
     * Vector conversion from covariant to physical covariant.
     *
     * @param xi coordinate array in code units
     * @param v_cov vector in physical covariant basis
     * @param vi_cov vector in covariant basis
     */
    Inline void v3_PhysCov2Cov(const coord_t<D>&,
                               const vec_t<Dim3>& v_cov,
                               vec_t<Dim3>&       vi_cov) const {
      if constexpr (D == Dim1) {
        vi_cov[0] = v_cov[0] * dx;
        vi_cov[1] = v_cov[1];
        vi_cov[2] = v_cov[2];
      } else if constexpr (D == Dim2) {
        vi_cov[0] = v_cov[0] * dx;
        vi_cov[1] = v_cov[1] * dx;
        vi_cov[2] = v_cov[2];
      } else {
        vi_cov[0] = v_cov[0] * dx;
        vi_cov[1] = v_cov[1] * dx;
        vi_cov[2] = v_cov[2] * dx;
      }
    }

    /**
     * Vector conversion from hatted to Cartesian basis.
     *
     * @param xi coordinate array in code units
     * @param v_hat vector in hatted basis
     * @param v_cart vector in Cartesian basis
     */
    Inline void v3_Hat2Cart(const coord_t<D>&,
                            const vec_t<Dim3>& v_hat,
                            vec_t<Dim3>&       v_cart) const {
      v_cart[0] = v_hat[0];
      v_cart[1] = v_hat[1];
      v_cart[2] = v_hat[2];
    }

    /**
     * Vector conversion from Cartesian to hatted basis.
     *
     * @param xi coordinate array in code units
     * @param v_cart vector in Cartesian basis
     * @param v_hat vector in hatted basis
     */
    Inline void v3_Cart2Hat(const coord_t<D>&,
                            const vec_t<Dim3>& v_cart,
                            vec_t<Dim3>&       v_hat) const {
      v_hat[0] = v_cart[0];
      v_hat[1] = v_cart[1];
      v_hat[2] = v_cart[2];
    }

    Inline auto x1_Code2Cart(const real_t& x1) const -> real_t {
      return x1 * dx + this->x1_min;
    }

    Inline auto x2_Code2Cart(const real_t& x2) const -> real_t {
      return x2 * dx + this->x2_min;
    }

    Inline auto x3_Code2Cart(const real_t& x3) const -> real_t {
      return x3 * dx + this->x3_min;
    }

    Inline auto x1_Cart2Code(const real_t& x) const -> real_t {
      return (x - this->x1_min) * dx_inv;
    }

    Inline auto x2_Cart2Code(const real_t& y) const -> real_t {
      return (y - this->x2_min) * dx_inv;
    }

    Inline auto x3_Cart2Code(const real_t& z) const -> real_t {
      return (z - this->x3_min) * dx_inv;
    }

    Inline auto x1_Code2Phys(const real_t& x1) const -> real_t {
      return this->x1_Code2Cart(x1);
    }

    Inline auto x2_Code2Phys(const real_t& x2) const -> real_t {
      return this->x2_Code2Cart(x2);
    }

    Inline auto x3_Code2Phys(const real_t& x3) const -> real_t {
      return this->x3_Code2Cart(x3);
    }

    Inline auto x1_Phys2Code(const real_t& x) const -> real_t {
      return this->x1_Cart2Code(x);
    }

    Inline auto x2_Phys2Code(const real_t& y) const -> real_t {
      return this->x2_Cart2Code(y);
    }

    Inline auto x3_Phys2Code(const real_t& z) const -> real_t {
      return this->x3_Cart2Code(z);
    }
  };

  /* -------------------------------------------------------------------------- */
  /*                                     1D */
  /* -------------------------------------------------------------------------- */
  template <>
  Inline void Metric<Dim1>::x_Code2Cart(const coord_t<Dim1>& xi,
                                        coord_t<Dim1>&       x) const {
    x[0] = xi[0] * dx + this->x1_min;
  }

  template <>
  Inline void Metric<Dim1>::x_Cart2Code(const coord_t<Dim1>& x,
                                        coord_t<Dim1>&       xi) const {
    xi[0] = (x[0] - this->x1_min) * dx_inv;
  }

  template <>
  Inline void Metric<Dim1>::x_Code2Sph(const coord_t<Dim1>&, coord_t<Dim1>&) const {
  }

  template <>
  Inline void Metric<Dim1>::x_Sph2Code(const coord_t<Dim1>&, coord_t<Dim1>&) const {
  }

  /* -------------------------------------------------------------------------- */
  /*                                     2D */
  /* -------------------------------------------------------------------------- */
  template <>
  Inline void Metric<Dim2>::x_Code2Cart(const coord_t<Dim2>& xi,
                                        coord_t<Dim2>&       x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
  }

  template <>
  Inline void Metric<Dim2>::x_Cart2Code(const coord_t<Dim2>& x,
                                        coord_t<Dim2>&       xi) const {
    xi[0] = (x[0] - this->x1_min) * dx_inv;
    xi[1] = (x[1] - this->x2_min) * dx_inv;
  }

  template <>
  Inline void Metric<Dim2>::x_Code2Sph(const coord_t<Dim2>& xi,
                                       coord_t<Dim2>&       x) const {
    coord_t<Dim2> x_cart { ZERO };
    x_Code2Cart(xi, x_cart);
    x[0] = math::sqrt(SQR(x_cart[0]) + SQR(x_cart[1]));
    x[1] = static_cast<real_t>(constant::HALF_PI) -
           math::atan2(x_cart[1], x_cart[0]);
  }

  template <>
  Inline void Metric<Dim2>::x_Sph2Code(const coord_t<Dim2>& x,
                                       coord_t<Dim2>&       xi) const {
    coord_t<Dim2> x_cart { ZERO };
    x_cart[0] = x[0] * math::sin(x[1]);
    x_cart[1] = x[0] * math::cos(x[1]);
    x_Cart2Code(x_cart, xi);
  }

  /* -------------------------------------------------------------------------- */
  /*                                     3D */
  /* -------------------------------------------------------------------------- */
  template <>
  Inline void Metric<Dim3>::x_Code2Cart(const coord_t<Dim3>& xi,
                                        coord_t<Dim3>&       x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
    x[2] = xi[2] * dx + this->x3_min;
  }

  template <>
  Inline void Metric<Dim3>::x_Cart2Code(const coord_t<Dim3>& x,
                                        coord_t<Dim3>&       xi) const {
    xi[0] = (x[0] - this->x1_min) * dx_inv;
    xi[1] = (x[1] - this->x2_min) * dx_inv;
    xi[2] = (x[2] - this->x3_min) * dx_inv;
  }

  template <>
  Inline void Metric<Dim3>::x_Code2Sph(const coord_t<Dim3>& xi,
                                       coord_t<Dim3>&       x) const {
    coord_t<Dim3> x_cart { ZERO };
    x_Code2Cart(xi, x_cart);
    x[0] = math::sqrt(SQR(x_cart[0]) + SQR(x_cart[1]) + SQR(x_cart[2]));
    x[1] = static_cast<real_t>(constant::HALF_PI) -
           math::atan2(x_cart[1], x_cart[0]);
    x[2] = math::acos(x_cart[2] / x_cart[0]);
  }

  template <>
  Inline void Metric<Dim3>::x_Sph2Code(const coord_t<Dim3>& x,
                                       coord_t<Dim3>&       xi) const {
    coord_t<Dim3> x_cart { ZERO };
    x_cart[0] = x[0] * math::sin(x[1]) * math::cos(x[2]);
    x_cart[1] = x[0] * math::sin(x[1]) * math::sin(x[2]);
    x_cart[2] = x[0] * math::cos(x[1]);
    x_Cart2Code(x_cart, xi);
  }

} // namespace ntt

#endif
