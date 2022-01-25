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
     * Vector conversion from hatted to contravariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     */
    Inline void v_Hat2Cntrv(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from contravariant to hatted basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     */
    Inline void v_Cntrv2Hat(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from hatted to covariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     * @param vi_cov vector in covariant basis (size of the array is 3).
     */
    Inline void v_Hat2Cov(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from covariant to hatted basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cov vector in covariant basis (size of the array is 3).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     */
    Inline void v_Cov2Hat(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from contravariant to global Cartesian basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     * @param vi_cart vector in global Cartesian basis (size of the array is 3).
     */
    Inline void v_Cntrv2Cart(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from covariant to global Cartesian basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cov vector in covariant basis (size of the array is 3).
     * @param vi_cart vector in global Cartesian basis (size of the array is 3).
     */
    Inline void v_Cov2Cart(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from global Cartesian to contravariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cart vector in global Cartesian basis (size of the array is 3).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     */
    Inline void v_Cart2Cntrv(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from global Cartesian to covariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cart vector in global Cartesian basis (size of the array is 3).
     * @param vi_cov vector in covariant basis (size of the array is 3).
     */
    Inline void v_Cart2Cov(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from covariant to contravariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cov vector in covariant basis (size of the array is 3).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     */
    Inline void v_Cov2Cntrv(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Vector conversion from contravariant to covariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     * @param vi_cov vector in covaraint basis (size of the array is 3).
     */
    Inline void v_Cntrv2Cov(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Compute the norm of a covariant vector.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cov vector in covariant basis (size of the array is 3).
     * @return Norm of the covariant vector.
     */
    Inline auto v_CovNorm(const coord_t<D>&, const vec_t<Dimension::THREE_D>&) const -> real_t;
    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cartesian coordinates in physical units (size of the array is D).
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
                                     vec_t<Dimension::THREE_D>& vi_cntrv) const {
    vi_cntrv[0] = vi_hat[0] / std::sqrt(h_11(xi));
    vi_cntrv[1] = vi_hat[1] / std::sqrt(h_22(xi));
    vi_cntrv[2] = vi_hat[2] / std::sqrt(h_33(xi));
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cntrv2Hat(const coord_t<D>& xi,
                                     const vec_t<Dimension::THREE_D>& vi_cntrv,
                                     vec_t<Dimension::THREE_D>& vi_hat) const {
    vi_hat[0] = vi_cntrv[0] * std::sqrt(h_11(xi));
    vi_hat[1] = vi_cntrv[1] * std::sqrt(h_22(xi));
    vi_hat[2] = vi_cntrv[2] * std::sqrt(h_33(xi));
  }
  template <Dimension D>
  Inline void Metric<D>::v_Hat2Cov(const coord_t<D>& xi,
                                   const vec_t<Dimension::THREE_D>& vi_hat,
                                   vec_t<Dimension::THREE_D>& vi_cov) const {
    vi_cov[0] = vi_hat[0] * std::sqrt(h_11(xi));
    vi_cov[1] = vi_hat[1] * std::sqrt(h_22(xi));
    vi_cov[2] = vi_hat[2] * std::sqrt(h_33(xi));
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cov2Hat(const coord_t<D>& xi,
                                   const vec_t<Dimension::THREE_D>& vi_cov,
                                   vec_t<Dimension::THREE_D>& vi_hat) const {
    vi_hat[0] = vi_cov[0] / std::sqrt(h_11(xi));
    vi_hat[1] = vi_cov[1] / std::sqrt(h_22(xi));
    vi_hat[2] = vi_cov[2] / std::sqrt(h_33(xi));
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cntrv2Cart(const coord_t<D>& xi,
                                      const vec_t<Dimension::THREE_D>& vi_cntrv,
                                      vec_t<Dimension::THREE_D>& vi_cart) const {
    this->v_Cntrv2Hat(xi, vi_cntrv, vi_cart);
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cart2Cntrv(const coord_t<D>& xi,
                                      const vec_t<Dimension::THREE_D>& vi_cart,
                                      vec_t<Dimension::THREE_D>& vi_cntrv) const {
    this->v_Hat2Cntrv(xi, vi_cart, vi_cntrv);
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cov2Cart(const coord_t<D>& xi,
                                    const vec_t<Dimension::THREE_D>& vi_cov,
                                    vec_t<Dimension::THREE_D>& vi_cart) const {
    vi_cart[0] = vi_cov[0] / std::sqrt(h_11(xi));
    vi_cart[1] = vi_cov[1] / std::sqrt(h_22(xi));
    vi_cart[2] = vi_cov[2] / std::sqrt(h_33(xi));
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cart2Cov(const coord_t<D>& xi,
                                    const vec_t<Dimension::THREE_D>& vi_cart,
                                    vec_t<Dimension::THREE_D>& vi_cov) const {
    vi_cov[0] = vi_cart[0] * std::sqrt(h_11(xi));
    vi_cov[1] = vi_cart[1] * std::sqrt(h_22(xi));
    vi_cov[2] = vi_cart[2] * std::sqrt(h_33(xi));
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cov2Cntrv(const coord_t<D>& xi,
                                     const vec_t<Dimension::THREE_D>& vi_cov,
                                     vec_t<Dimension::THREE_D>& vi_cntrv) const {
    vi_cntrv[0] = vi_cov[0] / h_11(xi);
    vi_cntrv[1] = vi_cov[1] / h_22(xi);
    vi_cntrv[2] = vi_cov[2] / h_33(xi);
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cntrv2Cov(const coord_t<D>& xi,
                                     const vec_t<Dimension::THREE_D>& vi_cntrv,
                                     vec_t<Dimension::THREE_D>& vi_cov) const {
    vi_cov[0] = vi_cntrv[0] * h_11(xi);
    vi_cov[1] = vi_cntrv[1] * h_22(xi);
    vi_cov[2] = vi_cntrv[2] * h_33(xi);
  }
  template <Dimension D>
  Inline auto Metric<D>::v_CovNorm(const coord_t<D>& xi, const vec_t<Dimension::THREE_D>& vi_cov) const -> real_t {
    return vi_cov[0] * vi_cov[0] / h_11(xi) + vi_cov[1] * vi_cov[1] / h_22(xi) + vi_cov[2] * vi_cov[2] / h_33(xi);
  }

  // * * * * * * * * * * * * * * *
  // 1D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::ONE_D>::x_Code2Cart(const coord_t<Dimension::ONE_D>& xi,
                                                    coord_t<Dimension::ONE_D>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
  }
  template <>
  Inline void Metric<Dimension::ONE_D>::x_Cart2Code(const coord_t<Dimension::ONE_D>& x,
                                                    coord_t<Dimension::ONE_D>& xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
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
  template <>
  Inline void Metric<Dimension::TWO_D>::x_Cart2Code(const coord_t<Dimension::TWO_D>& x,
                                                    coord_t<Dimension::TWO_D>& xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
    xi[1] = (x[1] - this->x2_min) * inv_dx;
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
  template <>
  Inline void Metric<Dimension::THREE_D>::x_Cart2Code(const coord_t<Dimension::THREE_D>& x,
                                                      coord_t<Dimension::THREE_D>& xi) const {
    xi[0] = (x[0] - this->x1_min) * inv_dx;
    xi[1] = (x[1] - this->x2_min) * inv_dx;
    xi[2] = (x[2] - this->x3_min) * inv_dx;
  }

} // namespace ntt

#endif
