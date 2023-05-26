#ifndef FRAMEWORK_METRICS_SPHERICAL_H
#define FRAMEWORK_METRICS_SPHERICAL_H

#include "wrapper.h"

#include "metric_base.h"

#include <cassert>
#include <cmath>

namespace ntt {
  /**
   * Flat metric in spherical system: diag(-1, 1, r^2, r^2 sin(th)^2).
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Metric : public MetricBase<D> {
  private:
    const real_t dr, dtheta, dphi;
    const real_t dr_inv, dtheta_inv, dphi_inv;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;

  public:
    const real_t dx_min;

    Metric(std::vector<unsigned int> resolution, std::vector<real_t> extent, const real_t*)
      : MetricBase<D> { "spherical", resolution, extent },
        dr((this->x1_max - this->x1_min) / this->nx1),
        dtheta(constant::PI / this->nx2),
        dphi(constant::TWO_PI / this->nx3),
        dr_inv { ONE / dr },
        dtheta_inv { ONE / dtheta },
        dphi_inv { ONE / dphi },
        dr_sqr { SQR(dr) },
        dtheta_sqr { SQR(dtheta) },
        dphi_sqr { SQR(dphi) },
        dx_min { findSmallestCell() } {}
    ~Metric() = default;

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>&) const -> real_t {
      return dr_sqr;
    }
    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& x) const -> real_t {
      real_t r { x[0] * dr + this->x1_min };
      return dtheta_sqr * SQR(r);
    }
    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& x) const -> real_t {
      real_t r { x[0] * dr + this->x1_min };
      real_t theta { x[1] * dtheta };
      real_t sin_theta { math::sin(theta) };
      if constexpr (D == Dim2) {
        return SQR(r) * SQR(sin_theta);
      } else {
        return dphi_sqr * SQR(r) * SQR(sin_theta);
      }
    }
    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      real_t r { x[0] * dr + this->x1_min };
      real_t theta { x[1] * dtheta };
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(r) * math::sin(theta);
      } else {
        return dr * dtheta * dphi * SQR(r) * math::sin(theta);
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
      real_t r { x1 * dr + this->x1_min };
      real_t del_theta { HALF * dtheta };
      return dr * SQR(r) * (ONE - math::cos(del_theta));
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/x_code_cart_forGSph.h"
#include "metrics_utils/x_code_sph_forSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forGsph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forSR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forSph.h"

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    auto findSmallestCell() const -> real_t {
      if constexpr (D == Dim2) {
        auto dx1 { dr };
        auto dx2 { this->x1_min * dtheta };
        return ONE / math::sqrt(ONE / SQR(dx1) + ONE / SQR(dx2));
      } else {
        NTTHostError("min cell finding not implemented for 3D spherical");
      }
      return ZERO;
    }
  };

}    // namespace ntt

#endif
