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
      : MetricBase<D> {"kerr_schild", resolution, extent},
        dr((this->x1_max - this->x1_min) / this->nx1),
        dtheta(constant::PI / this->nx2),
        dphi(constant::TWO_PI / this->nx3),
        dr_sqr(dr * dr),
        dtheta_sqr(dtheta * dtheta),
        dphi_sqr(dphi * dphi),
        a(params[3]) {}
    ~Metric() = default;

    [[nodiscard]] auto spin() const -> const real_t& {return a;}

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
      return dr_sqr * (ONE + TWO * r / (r * r + a * a * cth * cth));
    }    
    
    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};
      return dtheta_sqr * (r * r + a * a * cth * cth);
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

    /**
     * Compute metric component 13.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_13 (covariant, lower index) metric component.
     */
    Inline auto h_13(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};
      real_t sth {std::sin(theta)};
      return - dr * a * sth * sth * ( ONE + TWO * r / (r * r + a * a * cth * cth));
    }

    /**
     * Compute lapse function.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns alpha.
     */
    Inline auto alpha(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};

      real_t z {TWO * r / (r * r + a * a * cth * cth)};
      return ONE / std::sqrt(ONE + z);
    }

    /**
     * Compute radial component of shift vector.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns beta^1 (contravariant).
     */
    Inline auto beta1u(const coord_t<D>& x) const -> real_t {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t cth {std::cos(theta)};

      real_t z {TWO * r / (r * r + a * a * cth * cth)};
      return z / (ONE + z) / dr;
    }

     /**
     * Compute the square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      return h_22(x) / alpha(x);
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
      return dtheta * dphi * std::sqrt((r * r + a * a) * (ONE + TWO * r / ( r * r + a * a))) * (ONE - std::cos(del_theta));
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "non_diag_vector_transform.h"

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     * @returns Minimum cell size of the grid [physical units].
     */
    auto findSmallestCell() const -> real_t {
      if constexpr (D == Dimension::TWO_D) {
        real_t min_dx {-1.0};
        for (int i {0}; i < this->nx1; ++i) {
          for (int j {0}; j < this->nx2; ++j) {
            real_t i_ {(real_t)(i) + HALF};
            real_t j_ {(real_t)(j) + HALF};
            real_t inv_dx1_ {this->h_11_inv({i_, j_})};
            real_t inv_dx2_ {this->h_22_inv({i_, j_})};
            real_t dx = 1.0 / (this->alpha({i_, j_}) * std::sqrt(inv_dx1_ + inv_dx2_) + this->beta1u({i_, j_}));
            if ((min_dx >= dx) || (min_dx < 0.0)) { min_dx = dx; }
          }
        }
        return min_dx;
      } else {
        NTTError("min cell finding not implemented for 3D qspherical");
      }
      return ZERO;
    }

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cartesian physical units (size of the array is D).
     */
    Inline void x_Code2Cart(const coord_t<D>& xi, coord_t<D>& x) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("x_Code2Cart not implemented for 1D");
      } else if constexpr (D == Dimension::TWO_D) {
        coord_t<D> x_sph;
        x_Code2Sph(xi, x_sph);
        x[0] = x_sph[0] * std::sin(x_sph[1]);
        x[1] = x_sph[0] * std::cos(x_sph[1]);
      } else if constexpr (D == Dimension::THREE_D) {
        coord_t<D> x_sph;
        x_Code2Sph(xi, x_sph);
        x[0] = x_sph[0] * std::sin(x_sph[1]) * std::cos(x_sph[2]);
        x[1] = x_sph[0] * std::sin(x_sph[1]) * std::sin(x_sph[2]);
        x[2] = x_sph[0] * std::cos(x_sph[1]);
      }
    }

    /**
     * Coordinate conversion from Cartesian physical units to code units.
     *
     * @param x coordinate array in Cartesian coordinates in
     * physical units (size of the array is D).
     * @param xi coordinate array in code units (size of the array is D).
     */
    Inline void x_Cart2Code(const coord_t<D>& x, coord_t<D>& xi) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("x_Cart2Code not implemented for 1D");
      } else if constexpr (D == Dimension::TWO_D) {
        coord_t<D> x_sph;
        x_sph[0] = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        x_sph[1] = std::atan2(x[1], x[0]);
        x_Sph2Code(x_sph, xi);
      } else if constexpr (D == Dimension::THREE_D) {
        coord_t<D> x_sph;
        x_sph[0] = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        x_sph[1] = std::atan2(x[1], x[0]);
        x_sph[2] = std::acos(x[2] / x_sph[0]);
        x_Sph2Code(x_sph, xi);
      }
    }

    /**
     * Coordinate conversion from code units to Spherical physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Spherical coordinates in physical units (size of the array is D).
     */
    Inline void x_Code2Sph(const coord_t<D>& xi, coord_t<D>& x) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("x_Code2Sph not implemented for 1D");
      } else if constexpr (D == Dimension::TWO_D) {
        x[0] = xi[0] * dr + this->x1_min;
        x[1] = xi[1] * dtheta;
      } else if constexpr (D == Dimension::THREE_D) {
        x[0] = xi[0] * dr + this->x1_min;
        x[1] = xi[1] * dtheta;
        x[2] = xi[2] * dphi;
      }
    }

    /**
     * Coordinate conversion from Spherical physical units to code units.
     *
     * @param xi coordinate array in Spherical coordinates in physical units (size of the array is D).
     * @param x coordinate array in code units (size of the array is D).
     */
    Inline void x_Sph2Code(const coord_t<D>& x, coord_t<D>& xi) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("x_Code2Sph not implemented for 1D");
      } else if constexpr (D == Dimension::TWO_D) {
        xi[0] = (x[0] - this->x1_min) / dr;
        xi[1] = x[1] / dtheta;
      } else if constexpr (D == Dimension::THREE_D) {
        x[0] = (xi[0] - this->x1_min) / dr;
        x[1] = xi[1] / dtheta;
        x[2] = xi[2] / dphi;
      }
    }

  };

} // namespace ntt

#endif
