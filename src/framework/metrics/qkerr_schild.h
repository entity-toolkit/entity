#ifndef FRAMEWORK_METRICS_QKERR_SCHILD_H
#define FRAMEWORK_METRICS_QKERR_SCHILD_H

#include "wrapper.h"

#include "metric_base.h"

#include "utils/qmath.h"

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
    // Spin parameter, in [0,1[
    // and horizon size in units of rg
    // all physical extents are in units of rg
    const real_t rh_, rg_, a, a_sqr;

    const real_t r0, h;
    const real_t chi_min, eta_min, phi_min;
    const real_t dchi, deta, dphi;
    const real_t dchi_inv, deta_inv, dphi_inv;
    const real_t dchi_sqr, deta_sqr, dphi_sqr;

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
    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const real_t*             params)
      : MetricBase<D> { "qkerr_schild", resolution, extent },
        rh_ { params[5] },
        rg_ { ONE },
        a(params[4]),
        a_sqr { SQR(a) },
        r0(params[0]),
        h(params[1]),
        chi_min { math::log(this->x1_min - r0) },
        eta_min { theta2eta(this->x2_min) },
        phi_min { this->x3_min },
        dchi { (math::log(this->x1_max - r0) - chi_min) / this->nx1 },
        deta { (theta2eta(this->x2_max) - eta_min) / this->nx2 },
        dphi { (this->x3_max - phi_min) / this->nx3 },
        dchi_inv { ONE / dchi },
        deta_inv { ONE / deta },
        dphi_inv { ONE / dphi },
        dchi_sqr { SQR(dchi) },
        deta_sqr { SQR(deta) },
        dphi_sqr { SQR(dphi) } {
      this->set_dxMin(find_dxMin());
    }
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
     * Minimum effective cell size for a given metric (in physical units).
     * @returns Minimum cell size of the grid [physical units].
     */
    [[nodiscard]] auto find_dxMin() const -> real_t override {
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

    /**
     * Metric component 11.
     *
     * @param xi coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>& xi) const -> real_t {
      const real_t chi { xi[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t theta { eta2theta(xi[1] * deta + eta_min) };
      return dchi_sqr * math::exp(TWO * chi) * (ONE + z(r, theta));
    }

    /**
     * Metric component 22.
     *
     * @param xi coordinate array in code units
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& xi) const -> real_t {
      const real_t r { r0 + math::exp(xi[0] * dchi + chi_min) };
      const real_t eta { xi[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      return deta_sqr * SQR(dtheta_deta(eta)) * Sigma(r, theta);
    }

    /**
     * Metric component 33.
     *
     * @param xi coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& xi) const -> real_t {
      const real_t r { r0 + math::exp(xi[0] * dchi + chi_min) };
      const real_t theta { eta2theta(xi[1] * deta + eta_min) };
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
      const real_t chi { xi[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t theta { eta2theta(xi[1] * deta + eta_min) };
      if constexpr (D == Dim2) {
        return -dchi * math::exp(chi) * a * (ONE + z(r, theta)) * SQR(math::sin(theta));
      } else {
        return -dchi * math::exp(chi) * dphi * a * (ONE + z(r, theta)) * SQR(math::sin(theta));
      }
    }

    /**
     * Inverse metric component 11 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^11 (contravariant, upper index) metric component.
     */
    Inline auto h11(const coord_t<D>& xi) const -> real_t {
      const real_t chi { xi[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t theta { eta2theta(xi[1] * deta + eta_min) };
      const real_t Sigma_ { Sigma(r, theta) };
      return (math::exp(-TWO * chi) / dchi_sqr) * A(r, theta) / (Sigma_ * (Sigma_ + TWO * r));
    }

    /**
     * Inverse metric component 22 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^22 (contravariant, upper index) metric component.
     */
    Inline auto h22(const coord_t<D>& xi) const -> real_t {
      const real_t r { r0 + math::exp(xi[0] * dchi + chi_min) };
      const real_t eta { xi[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      return ONE / (Sigma(r, theta) * SQR(dtheta_deta(eta)) * deta_sqr);
    }

    /**
     * Inverse metric component 33 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^33 (contravariant, upper index) metric component.
     */
    Inline auto h33(const coord_t<D>& xi) const -> real_t {
      const real_t r { r0 + math::exp(xi[0] * dchi + chi_min) };
      const real_t theta { eta2theta(xi[1] * deta + eta_min) };
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
      const real_t chi { xi[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t theta { eta2theta(xi[1] * deta + eta_min) };
      if constexpr (D == Dim2) {
        return (math::exp(-chi) * dchi_inv) * a / Sigma(r, theta);
      } else {
        return (math::exp(-chi) * dchi_inv) * dphi_inv * a / Sigma(r, theta);
      }
    }

    /**
     * Lapse function.
     *
     * @param xi coordinate array in code units
     * @returns alpha.
     */
    Inline auto alpha(const coord_t<D>& xi) const -> real_t {
      const real_t r { r0 + math::exp(xi[0] * dchi + chi_min) };
      const real_t theta { eta2theta(xi[1] * deta + eta_min) };
      return ONE / math::sqrt(ONE + z(r, theta));
    }

    /**
     * Radial component of shift vector.
     *
     * @param xi coordinate array in code units
     * @returns beta^1 (contravariant).
     */
    Inline auto beta1(const coord_t<D>& xi) const -> real_t {
      const real_t chi { xi[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t theta { eta2theta(xi[1] * deta + eta_min) };
      const real_t z_ { z(r, theta) };
      return math::exp(-chi) * dchi_inv * z_ / (ONE + z_);
    }

    /**
     * Square root of the determinant of h-matrix.
     *
     * @param xi coordinate array in code units
     * @returns sqrt(det(h)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& xi) const -> real_t {
      const real_t chi { xi[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t eta { xi[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      if constexpr (D == Dim2) {
        return dchi * math::exp(chi) * dtheta_deta(eta) * deta * Sigma(r, theta)
               * math::sin(theta) * math::sqrt(ONE + z(r, theta));
      } else {
        return dchi * math::exp(chi) * dtheta_deta(eta) * deta * dphi * Sigma(r, theta)
               * math::sin(theta) * math::sqrt(ONE + z(r, theta));
      }
    }

    /**
     * Square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param xi coordinate array in code units
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& xi) const -> real_t {
      const real_t chi { xi[0] * dchi + chi_min };
      const real_t r { r0 + math::exp(chi) };
      const real_t eta { xi[1] * deta + eta_min };
      const real_t theta { eta2theta(eta) };
      if constexpr (D == Dim2) {
        return dchi * math::exp(chi) * dtheta_deta(eta) * deta * Sigma(r, theta)
               * math::sqrt(ONE + z(r, theta));
      } else {
        return dchi * math::exp(chi) * dtheta_deta(eta) * deta * dphi * Sigma(r, theta)
               * math::sqrt(ONE + z(r, theta));
      }
    }

    /**
     * Area at the pole (used in axisymmetric solvers).
     * Approximate solution for the polar area.
     *
     * @param x1 coordinate in code units
     * @returns Area at the pole.
     */
    Inline auto polar_area(const real_t& x1) const -> real_t {
      real_t chi { x1 * dchi + chi_min };
      real_t r { r0 + math::exp(chi) };
      real_t del_theta { eta2theta(HALF * deta + eta_min) };
      return dchi * math::exp(chi) * (SQR(r) + a_sqr)
             * math::sqrt(ONE + TWO * r / (SQR(r) + a_sqr)) * (ONE - math::cos(del_theta));
    }
/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a non-diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/angle_stretch_forQSph.h"
#include "metrics_utils/param_forGR.h"
#include "metrics_utils/x_code_cart_forGSph.h"
#include "metrics_utils/x_code_phys_forGSph.h"
#include "metrics_utils/x_code_sph_forQSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forGSph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forGR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forQSph.h"
  };

}    // namespace ntt

#endif
