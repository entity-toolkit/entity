#ifndef FRAMEWORK_METRICS_KERR_SCHILD_H
#define FRAMEWORK_METRICS_KERR_SCHILD_H

#include "global.h"
#include "metric_base.h"

#include <cmath>
#include <cassert>

namespace ntt {
  /**
   * Kerr metric in Kerr-Schild coordinates
   * Units: c = rg = 1
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Metric : public MetricBase<D> {
  private:
    const real_t dr, dtheta, dphi;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;
    // Spin parameter, in [0,1[
    const real_t a;

  public:
    Metric(std::vector<unsigned int> resolution, std::vector<real_t> extent, const real_t* params)
      : MetricBase<D> {"spherical", resolution, extent},
        dr((this->x1_max - this->x1_min) / this->nx1),
        dtheta(constant::PI / this->nx2),
        dphi(constant::TWO_PI / this->nx3),
        dr_sqr(dr * dr),
        dtheta_sqr(dtheta * dtheta),
        dphi_sqr(dphi * dphi),
        a(params[3]) {}
    ~Metric() = default;

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     * @todo Implement real CFL condition; this is approximate
     * @returns Minimum cell size of the grid [physical units].
     */
    auto findSmallestCell() const -> real_t {
      if constexpr (D == Dimension::TWO_D) {
        auto dx1 {dr};
        auto dx2 {this->x1_min * dtheta};
        
        return ONE / std::sqrt(ONE / (dx1 * dx1) + ONE / (dx2 * dx2));
      } else {
        NTTError("min cell finding not implemented for 3D spherical");
      }
      return ZERO;
    }

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};
      return dr_sqr * ( ONE + TWO * r / (r * r + a * a * cth * cth) );
    }    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};
      return dtheta_sqr / (r * r + a * a * cth * cth);
    }
    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};
      real_t sth {std::sin(theta)};

      real_t delta {r * r - TWO * r + a * a};
      real_t As {(r * r + a * a) * (r * r + a * a) - a * a * delta * sth * sth};
      return As * sth * sth / (r * r + a * a * cth * cth);
    }

    Inline auto h_13(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};
      real_t sth {std::sin(theta)};
      return - dr * a * sth * sth *( ONE + TWO * r / (r * r + a * a * cth * cth));
    }
    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};
      real_t sth {std::sin(theta)};

      real_t z {TWO * r / (r * r + a * a * cth * cth)};
      real_t alpha {ONE / std::sqrt(ONE + z)};
      return (r * r + a * a * cth * cth)* sth / alpha;
    }

    Inline auto alpha(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};

      real_t z {TWO * r / (r * r + a * a * cth * cth)};
      return ONE / std::sqrt(ONE + z);
    }

    Inline auto betar(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};

      real_t z {TWO * r / (r * r + a * a * cth * cth)};
      return z / (ONE + z);
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     * Approximate solution for the polar area.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns Area at the pole.
     */
    Inline auto polar_area(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t del_theta {x[1] * dtheta};
      return dtheta * dphi * std::sqrt((r * r + a * a ) * (ONE + TWO * r / ( r * r + a * a))) * (ONE - std::cos(del_theta));
    }
    /**
     * Coordinate conversion from code units to Metric physical units.
     * @todo Actually implement.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cpherical coordinates in physical units (size of the array is D).
     */
    Inline void x_Code2Sph(const coord_t<D>& xi, coord_t<D>& x) const;
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
     * Linear form conversion from hatted to covariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param omega_hat form in hatted basis (size of the array is 3).
     * @param omega form in covariant basis (size of the array is 3).
     */
    Inline void omega_Hat2Cov(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
    /**
     * Linear form conversion from covariant to hatted basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param omega vector in covariant basis (size of the array is 3).
     * @param omega_hat vector in hatted basis (size of the array is 3).
     */
    Inline void omega_Cov2Hat(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     * @todo Actually implement.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cartesian coordinates in physical units (size of the array is D).
     */
    Inline void x_Code2Cart(const coord_t<D>&, coord_t<D>&) const {};
  };

  // * * * * * * * * * * * * * * *
  // vector transformations
  // * * * * * * * * * * * * * * *
  template <Dimension D>
  Inline void Metric<D>::v_Hat2Cntrv(const coord_t<D>& xi,
                                        const vec_t<Dimension::THREE_D>& vi_hat,
                                        vec_t<Dimension::THREE_D>& vi) const {
    real_t A0 {std::sqrt(h_33(xi) / ( h_11(xi) * h_33(xi) - h_13(xi) * h_13(xi)))};
  
    vi[0] = vi_hat[0] * A0;
    vi[1] = vi_hat[1] / std::sqrt(h_22(xi));
    vi[2] = vi_hat[2] / std::sqrt(h_33(xi)) - vi_hat[0] * A0 * h_13(xi) / h_33(xi);
  }
  template <Dimension D>
  Inline void Metric<D>::v_Cntrv2Hat(const coord_t<D>& xi,
                                        const vec_t<Dimension::THREE_D>& vi,
                                        vec_t<Dimension::THREE_D>& vi_hat) const {
        
    vi_hat[0] = vi[0] / std::sqrt(h_33(xi) / ( h_11(xi) * h_33(xi) - h_13(xi) * h_13(xi)));
    vi_hat[1] = vi[1] * std::sqrt(h_22(xi));
    vi_hat[2] = vi[2] * std::sqrt(h_33(xi)) + vi[0] * (h_13(xi) / std::sqrt(h_33(xi)));
  }
  template <Dimension D>
  Inline void Metric<D>::omega_Cov2Hat(const coord_t<D>& xi,
                                        const vec_t<Dimension::THREE_D>& omega,
                                        vec_t<Dimension::THREE_D>& omega_hat) const {
    real_t A0 {std::sqrt(h_33(xi) / ( h_11(xi) * h_33(xi) - h_13(xi) * h_13(xi)))};
        
    omega_hat[0] = omega[0] * A0 - omega[2] * A0 * h_13(xi) / h_33(xi);
    omega_hat[1] = omega[1] / std::sqrt(h_22(xi));
    omega_hat[2] = omega[2] / std::sqrt(h_33(xi)) ;
  }
  template <Dimension D>
  Inline void Metric<D>::omega_Hat2Cov(const coord_t<D>& xi,
                                        const vec_t<Dimension::THREE_D>& omega_hat,
                                        vec_t<Dimension::THREE_D>& omega) const {
        
    omega[0] = omega_hat[0] / std::sqrt(h_33(xi) / ( h_11(xi) * h_33(xi) - h_13(xi) * h_13(xi))) + omega_hat[2] *  h_13(xi) / std::sqrt(h_33(xi));
    omega[1] = omega_hat[1] * std::sqrt(h_22(xi));
    omega[2] = omega_hat[2] * std::sqrt(h_33(xi)) ;
  }

  // * * * * * * * * * * * * * * *
  // 1D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::ONE_D>::x_Code2Sph(const coord_t<Dimension::ONE_D>&,
                                                      coord_t<Dimension::ONE_D>&) const { }

  // * * * * * * * * * * * * * * *
  // 2D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::TWO_D>::x_Code2Sph(const coord_t<Dimension::TWO_D>& xi,
                                                      coord_t<Dimension::TWO_D>& x) const {
    x[0] = xi[0] * dr + this->x1_min;
    x[1] = xi[1] * dtheta + this->x2_min;
  }

  // * * * * * * * * * * * * * * *
  // 3D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Metric<Dimension::THREE_D>::x_Code2Sph(const coord_t<Dimension::THREE_D>& xi,
                                                        coord_t<Dimension::THREE_D>& x) const {
    x[0] = xi[0] * dr + this->x1_min;
    x[1] = xi[1] * dtheta + this->x2_min;
    x[2] = xi[2] * dphi + this->x3_min;
  }

  } // namespace ntt

#endif
