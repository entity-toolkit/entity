#ifndef FRAMEWORK_METRICS_QKERR_SCHILD_H
#define FRAMEWORK_METRICS_QKERR_SCHILD_H

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
    const real_t r0, h;
    const real_t chi_min, eta_min, phi_min;
    const real_t dchi, deta, dphi;
    const real_t dchi_inv, deta_inv, dphi_inv;
    const real_t dchi_sqr, deta_sqr, dphi_sqr;
    // Spin parameter, in [0,1[
    // and horizon size in units of rg
    // all physical extents are in units of rg
    const real_t rh, a, a_sqr;

  public:
    const real_t dx_min;

    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const real_t*             params)
      : MetricBase<D> { "qkerr_schild", resolution, extent },
        a(params[4]),
        a_sqr { a_sqr },
        rh(params[5]),
        r0(params[0]),
        h(params[1]),
        chi_min { math::log(this->x1_min - r0) },
        eta_min { ZERO },
        phi_min { ZERO },
        dchi { (math::log(this->x1_max - r0) - chi_min) / this->nx1 },
        deta { constant::PI / this->nx2 },
        dphi { constant::TWO_PI / this->nx3 },
        dchi_inv { ONE / dchi },
        deta_inv { ONE / deta },
        dphi_inv { ONE / dphi },
        dchi_sqr { SQR(dchi) },
        deta_sqr { SQR(deta) },
        dphi_sqr { SQR(dphi) },
        dx_min { findSmallestCell() } {}
    ~Metric() = default;

    [[nodiscard]] Inline auto spin() const -> const real_t& {
      return a;
    }

    [[nodiscard]] auto rhorizon() const -> const real_t& {
      return rh;
    }

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

    /**
     * Compute metric component 11.
     *
     * @param xi coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>& xi) const -> real_t {
      real_t chi { xi[0] * dchi + chi_min };
      real_t r { r0 + math::exp(chi) };
      real_t eta { xi[1] * deta + eta_min };
      real_t theta { eta2theta(eta) };
      real_t cth { math::cos(theta) };
      return dchi_sqr * math::exp(2.0 * chi) * (ONE + TWO * r / (SQR(r) + a_sqr * SQR(cth)));
    }

    /**
     * Compute metric component 22.
     *
     * @param xi coordinate array in code units
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& xi) const -> real_t {
      real_t chi { xi[0] * dchi + chi_min };
      real_t r { r0 + math::exp(chi) };
      real_t eta { xi[1] * deta + eta_min };
      real_t theta { eta2theta(eta) };
      real_t dtheta_deta_ { dtheta_deta(eta) };
      real_t cth { math::cos(theta) };
      return deta_sqr * SQR(dtheta_deta_) * (SQR(r) + a_sqr * SQR(cth));
    }

    /**
     * Compute metric component 33.
     *
     * @param xi coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& xi) const -> real_t {
      real_t chi { xi[0] * dchi + chi_min };
      real_t r { r0 + math::exp(chi) };
      real_t eta { xi[1] * deta + eta_min };
      real_t theta { eta2theta(eta) };
      real_t cth { math::cos(theta) };
      real_t sth { math::sin(theta) };
      real_t delta { SQR(r) - TWO * r + a_sqr };
      real_t As { (SQR(r) + a_sqr) * (SQR(r) + a_sqr) - a_sqr * delta * SQR(sth) };
      return As * SQR(sth) / (SQR(r) + a_sqr * SQR(cth));
    }

    /**
     * Compute metric component 13.
     *
     * @param xi coordinate array in code units
     * @returns h_13 (covariant, lower index) metric component.
     */
    Inline auto h_13(const coord_t<D>& xi) const -> real_t {
      real_t chi { xi[0] * dchi + chi_min };
      real_t r { r0 + math::exp(chi) };
      real_t eta { xi[1] * deta + eta_min };
      real_t theta { eta2theta(eta) };
      real_t cth { math::cos(theta) };
      real_t sth { math::sin(theta) };
      return -dchi * math::exp(chi) * a * SQR(sth)
             * (ONE + TWO * r / (SQR(r) + a_sqr * SQR(cth)));
    }

    /**
     * Compute lapse function.
     *
     * @param xi coordinate array in code units
     * @returns alpha.
     */
    Inline auto alpha(const coord_t<D>& xi) const -> real_t {
      real_t chi { xi[0] * dchi + chi_min };
      real_t r { r0 + math::exp(chi) };
      real_t eta { xi[1] * deta + eta_min };
      real_t theta { eta2theta(eta) };
      real_t cth { math::cos(theta) };
      real_t z { TWO * r / (SQR(r) + a_sqr * SQR(cth)) };
      return ONE / math::sqrt(ONE + z);
    }

    /**
     * Compute radial component of shift vector.
     *
     * @param xi coordinate array in code units
     * @returns beta^1 (contravariant).
     */
    Inline auto beta1(const coord_t<D>& xi) const -> real_t {
      real_t chi { xi[0] * dchi + chi_min };
      real_t r { r0 + math::exp(chi) };
      real_t eta { xi[1] * deta + eta_min };
      real_t theta { eta2theta(eta) };
      real_t cth { math::cos(theta) };
      real_t z { TWO * r / (SQR(r) + a_sqr * SQR(cth)) };
      return math::exp(-chi) * dchi_inv * (z / (ONE + z));
    }

    /**
     * Compute the square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      return h_22(x) / alpha(x);
    }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      return math::sqrt(h_22(x) * (h_11(x) * h_33(x) - h_13(x) * h_13(x)));
    }

    /**
     * Compute inverse metric component 11 from h_ij.
     *
     * @param x coordinate array in code units
     * @returns h^11 (contravariant, upper index) metric component.
     */
    Inline auto h11(const coord_t<D>& x) const -> real_t {
      return h_33(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
    }

    /**
     * Compute inverse metric component 22 from h_ij.
     *
     * @param x coordinate array in code units
     * @returns h^22 (contravariant, upper index) metric component.
     */
    Inline auto h22(const coord_t<D>& x) const -> real_t {
      return ONE / h_22(x);
    }

    /**
     * Compute inverse metric component 33 from h_ij.
     *
     * @param x coordinate array in code units
     * @returns h^33 (contravariant, upper index) metric component.
     */
    Inline auto h33(const coord_t<D>& x) const -> real_t {
      return h_11(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
    }

    /**
     * Compute inverse metric component 13 from h_ij.
     *
     * @param x coordinate array in code units
     * @returns h^13 (contravariant, upper index) metric component.
     */
    Inline auto h13(const coord_t<D>& x) const -> real_t {
      return -h_13(x) / (h_11(x) * h_33(x) - SQR(h_13(x)));
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
     * @param x coordinate array in code units
     * @returns Area at the pole.
     */
    Inline auto polar_area(const coord_t<D>& x) const -> real_t {
      real_t chi { x[0] * dchi + chi_min };
      real_t r { r0 + math::exp(chi) };
      real_t del_eta { x[1] * deta + eta_min };
      real_t del_theta { eta2theta(del_eta) };
      return dchi * math::exp(chi) * (SQR(r) + a_sqr)
             * math::sqrt(ONE + TWO * r / (SQR(r) + a_sqr)) * (ONE - math::cos(del_theta));
    }
/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a non-diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/angle_stretch_forQSph.h"
#include "metrics_utils/x_code_cart_forGSph.h"
#include "metrics_utils/x_code_sph_forQSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forGsph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forGR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forQSph.h"
  };

}    // namespace ntt

#endif
