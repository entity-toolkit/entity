#ifndef FRAMEWORK_METRICS_KERR_SCHILD_H
#define FRAMEWORK_METRICS_KERR_SCHILD_H

#include "wrapper.h"

#include "metric_base.h"

#include <cassert>
#include <cmath>

namespace ntt {
  /**
   * Kerr metric with zero spin and zero mass in Kerr-Schild coordinates (for testing).
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
    const real_t rh, a, a_sqr;

  public:
    const real_t dx_min;

    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const real_t*             params)
      : MetricBase<D> { "kerr_schild_0", resolution, extent },
        rh { params[5] },
        a { ZERO },
        a_sqr { ZERO },
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

    [[nodiscard]] auto spin() const -> const real_t& {
      return a;
    }

    [[nodiscard]] auto rhorizon() const -> const real_t& {
      return rh;
    }
    Inline auto h_11(const coord_t<D>& x) const -> real_t {
      return dr_sqr;
    }
    Inline auto h_22(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      return dtheta_sqr * SQR(r);
    }
    Inline auto h_33(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      const real_t theta { x[1] * dtheta };
      if constexpr (D == Dim2) {
        return SQR(r * math::sin(theta));
      } else {
        return dphi_sqr * SQR(r * math::sin(theta));
      }
    }
    Inline auto h_13(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }
    Inline auto h11(const coord_t<D>& x) const -> real_t {
      return SQR(dr_inv);
    }
    Inline auto h22(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      return SQR(dtheta_inv / r);
    }
    Inline auto h33(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      const real_t theta { x[1] * dtheta };
      if constexpr (D == Dim2) {
        return ONE / SQR(r * math::sin(theta));
      } else {
        return SQR(dphi_inv) / SQR(r * math::sin(theta));
      }
    }
    Inline auto h13(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }
    Inline auto alpha(const coord_t<D>& x) const -> real_t {
      return ONE;
    }
    Inline auto beta1(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      const real_t theta { x[1] * dtheta };
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(r) * math::sin(theta);
      } else {
        return dr * dtheta * dphi * SQR(r) * math::sin(theta);
      }
    }
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(r);
      } else {
        return dr * dtheta * dphi * SQR(r);
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
     * @param x coordinate array in code units
     * @returns Area at the pole.
     */
    Inline auto polar_area(const coord_t<D>& x) const -> real_t {
      real_t r { x[0] * dr + this->x1_min };
      real_t del_theta { x[1] * dtheta };
      return dr * SQR(r) * (ONE - math::cos(del_theta));
    }
/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a non-diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/x_code_cart_forGSph.h"
#include "metrics_utils/x_code_sph_forSph.h"

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