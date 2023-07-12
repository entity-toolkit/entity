#ifndef FRAMEWORK_METRICS_KERR_SCHILD_H
#define FRAMEWORK_METRICS_KERR_SCHILD_H

#include "wrapper.h"

#include "metric_base.h"

#include <cassert>
#include <cmath>

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
    const real_t dr_inv, dtheta_inv, dphi_inv;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;
    // Spin parameter, in [0,1[
    // and horizon size in units of rg
    // all physical extents are in units of rg
    const real_t rh_, rg_, a, a_sqr;

    Inline auto  Delta(const real_t& r) const -> real_t {
      return SQR(r) - TWO * r + a_sqr;
    }

    Inline auto Sigma(const real_t& r, const real_t& theta) const -> real_t {
      return SQR(r) + a_sqr * SQR(math::cos(theta));
    }

    Inline auto A(const real_t& r, const real_t& theta) const -> real_t {
      return SQR(SQR(r) + a_sqr) - a_sqr * Delta(r) * SQR(math::sin(theta));
    }

    Inline auto z(const real_t& r, const real_t& theta) const -> real_t {
      return TWO * r / Sigma(r, theta);
    }

  public:
    const real_t dx_min;

    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const real_t*             params)
      : MetricBase<D> { "kerr_schild", resolution, extent },
        rh_ { params[5] },
        rg_ { ONE },
        a { params[4] },
        a_sqr { SQR(a) },
        dr { (this->x1_max - this->x1_min) / this->nx1 },
        dtheta { static_cast<real_t>(constant::PI / this->nx2) },
        dphi { static_cast<real_t>(constant::TWO_PI / this->nx3) },
        dr_inv { ONE / dr },
        dtheta_inv { ONE / dtheta },
        dphi_inv { ONE / dphi },
        dr_sqr { dr * dr },
        dtheta_sqr { dtheta * dtheta },
        dphi_sqr { dphi * dphi },
        dx_min { findSmallestCell() } {}
    ~Metric() = default;

    [[nodiscard]] Inline auto spin() const -> const real_t& {
      return a;
    }
    [[nodiscard]] Inline auto rhorizon() const -> const real_t& {
      return rh_;
    }
    [[nodiscard]] Inline auto rg() const -> const real_t& {
      return rg_;
    }

    /**
     * Metric component 11.
     *
     * @param xi coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      return dr_sqr * (ONE + z(r, theta));
    }

    /**
     * Metric component 22.
     *
     * @param xi coordinate array in code units
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      return dtheta_sqr * Sigma(r, theta);
    }

    /**
     * Metric component 33.
     *
     * @param xi coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      if constexpr (D == Dim2) {
        return A(r, theta) * SQR(math::sin(theta)) / Sigma(r, theta);
      } else {
        return dphi_sqr * A(r, theta) * SQR(math::sin(theta)) / Sigma(r, theta);
      }
    }

    /**
     * Metric component 13.
     *
     * @param xi coordinate array in code units
     * @returns h_13 (covariant, lower index) metric component.
     */
    Inline auto h_13(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      if constexpr (D == Dim2) {
        return -dr * a * (ONE + z(r, theta)) * SQR(math::sin(theta));
      } else {
        return -dr * dphi * a * (ONE + z(r, theta)) * SQR(math::sin(theta));
      }
    }

    /**
     * Inverse metric component 11 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^11 (contravariant, upper index) metric component.
     */
    Inline auto h11(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      const real_t Sigma_ { Sigma(r, theta) };
      return SQR(dr_inv) * A(r, theta) / (Sigma_ * (Sigma_ + TWO * r));
    }

    /**
     * Inverse metric component 22 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^22 (contravariant, upper index) metric component.
     */
    Inline auto h22(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      return SQR(dtheta_inv) / Sigma(r, theta);
    }

    /**
     * Inverse metric component 33 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^33 (contravariant, upper index) metric component.
     */
    Inline auto h33(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      if constexpr (D == Dim2) {
        return ONE / (Sigma(r, theta) * SQR(math::sin(theta)));
      } else {
        return SQR(dphi_inv) / (Sigma(r, theta) * SQR(math::sin(theta)));
      }
    }

    /**
     * Inverse metric component 13 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^13 (contravariant, upper index) metric component.
     */
    Inline auto h13(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      if constexpr (D == Dim2) {
        return dr_inv * a / Sigma(r, theta);
      } else {
        return dr_inv * dphi_inv * a / Sigma(r, theta);
      }
    }

    /**
     * Lapse function.
     *
     * @param xi coordinate array in code units
     * @returns alpha.
     */
    Inline auto alpha(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      return ONE / math::sqrt(ONE + z(r, theta));
    }

    /**
     * Radial component of shift vector.
     *
     * @param xi coordinate array in code units
     * @returns beta^1 (contravariant).
     */
    Inline auto beta1(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      const real_t z_ { z(r, theta) };
      return dr_inv * z_ / (ONE + z_);
    }

    /**
     * Square root of the determinant of h-matrix.
     *
     * @param xi coordinate array in code units
     * @returns sqrt(det(h)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      if constexpr (D == Dim2) {
        return dr * dtheta * Sigma(r, theta) * math::sin(theta)
               * math::sqrt(ONE + z(r, theta));
      } else {
        return dr * dtheta * dphi * Sigma(r, theta) * math::sin(theta)
               * math::sqrt(ONE + z(r, theta));
      }
    }

    /**
     * Square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param xi coordinate array in code units
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& xi) const -> real_t {
      const real_t r { xi[0] * dr + this->x1_min };
      const real_t theta { xi[1] * dtheta + this->x2_min };
      if constexpr (D == Dim2) {
        return dr * dtheta * Sigma(r, theta) * math::sqrt(ONE + z(r, theta));
      } else {
        return dr * dtheta * dphi * Sigma(r, theta) * math::sqrt(ONE + z(r, theta));
      }
    }

    /**
     * Compute the fiducial minimum cell volume.
     *
     * @returns Minimum cell volume of the grid [code units].
     */
    Inline auto min_cell_volume() const -> real_t {
      return math::pow(dx_min * math::sqrt(static_cast<real_t>(D)), static_cast<short>(D));
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     * Approximate solution for the polar area.
     *
     * @param x1 coordinate in code units
     * @returns Area at the pole.
     */
    Inline auto polar_area(const real_t& x1) const -> real_t {
      real_t r { x1 * dr + this->x1_min };
      real_t del_theta { HALF * dtheta };
      return dr * (SQR(r) + a_sqr) * math::sqrt(ONE + TWO * r / (SQR(r) + a_sqr))
             * (ONE - math::cos(del_theta));
    }
/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a non-diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/x_code_cart_forGSph.h"
#include "metrics_utils/x_code_sph_forSph.h"
#include "metrics_utils/x_code_phys_forGSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forGSph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forGR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forSph.h"

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     * @returns Minimum cell size of the grid [physical units].
     */
    auto findSmallestCell() const -> real_t {
      if constexpr (D == Dim2) {
        real_t min_dx { -ONE };
        for (int i { 0 }; i < this->nx1; ++i) {
          for (int j { 0 }; j < this->nx2; ++j) {
            real_t        i_ { static_cast<real_t>(i) + HALF };
            real_t        j_ { static_cast<real_t>(j) + HALF };
            coord_t<Dim2> ij { i_, j_ };
            real_t        dx = ONE
                        / (this->alpha(ij) * std::sqrt(this->h11(ij) + this->h22(ij))
                           + this->beta1(ij));
            if ((min_dx > dx) || (min_dx < 0.0)) {
              min_dx = dx;
            }
          }
        }
        return min_dx;
      } else {
        NTTHostError("min cell finding not implemented for 3D");
        return ZERO;
      }
    }
  };
}    // namespace ntt

#endif