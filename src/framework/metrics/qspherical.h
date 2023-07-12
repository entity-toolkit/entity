#ifndef FRAMEWORK_METRICS_QSPHERICAL_H
#define FRAMEWORK_METRICS_QSPHERICAL_H

#include "wrapper.h"

#include "metric_base.h"

#include "utils/qmath.h"

namespace ntt {
  /**
   * Flat metric in quasi-spherical system.
   * chi, eta, phi = log(r-r0), f(h, theta), phi
   *
   * !TODO: change `phi_min`.
   * @tparam D dimension.
   */
  template <Dimension D>
  class Metric : public MetricBase<D> {
  private:
    const real_t r0, h, chi_min, phi_min, eta_min;
    const real_t dchi, deta, dphi;
    const real_t dchi_inv, deta_inv, dphi_inv;
    const real_t dchi_sqr, deta_sqr, dphi_sqr;

  public:
    const real_t dx_min;

    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const real_t*             params)
      : MetricBase<D> { "qspherical", resolution, extent },
        r0 { params[0] },
        h { params[1] },
        chi_min { math::log(this->x1_min - r0) },
        eta_min { ZERO },
        phi_min { ZERO },
        dchi { (math::log(this->x1_max - r0) - chi_min) / this->nx1 },
        deta { static_cast<real_t>(constant::PI) / this->nx2 },
        dphi { static_cast<real_t>(constant::TWO_PI) / this->nx3 },
        dchi_inv { ONE / dchi },
        deta_inv { ONE / deta },
        dphi_inv { ONE / dphi },
        dchi_sqr { SQR(dchi) },
        deta_sqr { SQR(deta) },
        dphi_sqr { SQR(dphi) },
        dx_min { findSmallestCell() } {
      if constexpr ((D == Dim1) || (D == Dim3)) {
        NTTHostError("Qspherical can only be defined for 2D");
      }
    }
    ~Metric() = default;

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    auto findSmallestCell() const -> real_t {
      if constexpr (D == Dim2) {
        real_t min_dx { -1.0 };
        for (int i { 0 }; i < this->nx1; ++i) {
          for (int j { 0 }; j < this->nx2; ++j) {
            real_t i_ { (real_t)(i) + HALF };
            real_t j_ { (real_t)(j) + HALF };
            real_t dx1_ { this->h_11({ i_, j_ }) };
            real_t dx2_ { this->h_22({ i_, j_ }) };
            real_t dx = 1.0 / math::sqrt(1.0 / dx1_ + 1.0 / dx2_);
            if ((min_dx >= dx) || (min_dx < 0.0)) {
              min_dx = dx;
            }
          }
        }
        return min_dx;
      } else {
        NTTHostError("min cell finding not implemented for 3D qspherical");
      }
      return ZERO;
    }

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>& x) const -> real_t {
      if constexpr (D != Dim1) {
        real_t chi { x[0] * dchi + chi_min };
        return dchi_sqr * math::exp(2.0 * chi);
      } else {
        NTTError("1D qspherical not available");
        return ZERO;
      }
    }
    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& x) const -> real_t {
      if constexpr (D != Dim1) {
        real_t chi { x[0] * dchi + chi_min };
        real_t r { r0 + math::exp(chi) };
        real_t eta { x[1] * deta + eta_min };
        real_t dtheta_deta_ { dtheta_deta(eta) };
        return deta_sqr * SQR(dtheta_deta_) * r * r;
      } else {
        NTTError("1D qspherical not available");
        return ZERO;
      }
    }
    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& x) const -> real_t {
      if constexpr (D != Dim1) {
        real_t chi { x[0] * dchi + chi_min };
        real_t r { r0 + math::exp(chi) };
        real_t eta { x[1] * deta + eta_min };
        real_t theta { eta2theta(eta) };
        real_t sin_theta { math::sin(theta) };
        return r * r * sin_theta * sin_theta;
      } else {
        NTTError("1D qspherical not available");
        return ZERO;
      }
    }
    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      if constexpr (D != Dim1) {
        real_t chi { x[0] * dchi + chi_min };
        real_t r { r0 + math::exp(chi) };
        real_t eta { x[1] * deta + eta_min };
        real_t theta { eta2theta(eta) };
        real_t sin_theta { math::sin(theta) };
        real_t dtheta_deta_ { dtheta_deta(eta) };
        return dchi * deta * math::exp(chi) * r * r * sin_theta * dtheta_deta_;
      } else {
        NTTError("1D qspherical not available");
        return ZERO;
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
     *
     * @param x1 radial coordinate along the axis (code units).
     * @returns Area at the pole.
     */
    Inline auto polar_area(const real_t& x1) const -> real_t {
      if constexpr (D != Dim1) {
        real_t chi { x1 * dchi + chi_min };
        real_t r { r0 + math::exp(chi) };
        real_t del_eta { HALF * deta };
        real_t del_theta { eta2theta(del_eta) };
        return dchi * math::exp(chi) * r * r * (ONE - math::cos(del_theta));
      } else {
        NTTError("1D qspherical not available");
        return ZERO;
      }
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/angle_stretch_forQSph.h"
#include "metrics_utils/x_code_cart_forGSph.h"
#include "metrics_utils/x_code_sph_forQSph.h"
#include "metrics_utils/x_code_phys_forGSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forGSph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forSR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forQSph.h"
  };
}    // namespace ntt

#endif