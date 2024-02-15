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
    constexpr static Dimension PrtlD = Dim3;

    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const real_t*) :
      MetricBase<D> { "spherical", resolution, extent },
      dr((this->x1_max - this->x1_min) / this->nx1),
      dtheta((this->x2_max - this->x2_min) / this->nx2),
      dphi((this->x3_max - this->x3_min) / this->nx3),
      dr_inv { ONE / dr },
      dtheta_inv { ONE / dtheta },
      dphi_inv { ONE / dphi },
      dr_sqr { SQR(dr) },
      dtheta_sqr { SQR(dtheta) },
      dphi_sqr { SQR(dphi) } {
      this->set_dxMin(find_dxMin());
    }

    ~Metric() = default;

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>&) const -> real_t {
      if constexpr (D != Dim1) {
        return dr_sqr;
      } else {
        NTTError("1D spherical not available");
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
        return dtheta_sqr * SQR(x[0] * dr + this->x1_min);
      } else {
        NTTError("1D spherical not available");
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
      if constexpr (D == Dim2) {
        return SQR(x[0] * dr + this->x1_min) *
               SQR(math::sin(x[1] * dtheta + this->x2_min));
      } else if constexpr (D == Dim3) {
        return dphi_sqr * SQR(x[0] * dr + this->x1_min) *
               SQR(math::sin(x[1] * dtheta + this->x2_min));
      } else {
        NTTError("1D spherical not available");
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
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(x[0] * dr + this->x1_min) *
               math::sin(x[1] * dtheta + this->x2_min);
      } else if constexpr (D == Dim3) {
        return dr * dtheta * dphi * SQR(x[0] * dr + this->x1_min) *
               math::sin(x[1] * dtheta + this->x2_min);
      } else {
        NTTError("1D spherical not available");
        return ZERO;
      }
    }

    /**
     * Square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(x[0] * dr + this->x1_min);
      } else if constexpr (D == Dim3) {
        return dr * dtheta * dphi * SQR(x[0] * dr + this->x1_min);
      } else {
        NTTError("1D spherical not available");
        return ZERO;
      }
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     *
     * @param x1 radial coordinate along the axis (code units).
     * @returns Area at the pole.
     */
    Inline auto polar_area(const real_t& x1) const -> real_t {
      return dr * SQR(x1 * dr + this->x1_min) * (ONE - math::cos(HALF * dtheta));
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/param_forSR.h"
#include "metrics_utils/x_code_cart_forGSph.h"
#include "metrics_utils/x_code_phys_forGSph.h"
#include "metrics_utils/x_code_sph_forSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forGSph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forSR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forSph.h"

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
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

} // namespace ntt

#endif
