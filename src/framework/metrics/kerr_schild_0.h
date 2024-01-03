#ifndef FRAMEWORK_METRICS_KERR_SCHILD_H
#define FRAMEWORK_METRICS_KERR_SCHILD_H

#include "wrapper.h"

#include "metric_base.h"

#include <cassert>
#include <cmath>

namespace ntt {
  /**
   * Kerr metric with zero spin and zero mass in Kerr-Schild coordinates.
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

    const real_t dr, dtheta, dphi;
    const real_t dr_inv, dtheta_inv, dphi_inv;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;

  public:
    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const real_t*) :
      MetricBase<D> { "kerr_schild_0", resolution, extent },
      rh_ { ZERO },
      rg_ { ZERO },
      a { ZERO },
      a_sqr { ZERO },
      dr { (this->x1_max - this->x1_min) / this->nx1 },
      dtheta { (this->x2_max - this->x2_min) / this->nx2 },
      dphi { (this->x3_max - this->x3_min) / this->nx3 },
      dr_inv { ONE / dr },
      dtheta_inv { ONE / dtheta },
      dphi_inv { ONE / dphi },
      dr_sqr { dr * dr },
      dtheta_sqr { dtheta * dtheta },
      dphi_sqr { dphi * dphi } {
      this->set_dxMin(find_dxMin());
    }

    ~Metric() = default;

    [[nodiscard]]
    Inline auto spin() const -> const real_t& {
      return a;
    }

    [[nodiscard]]
    Inline auto rhorizon() const -> const real_t& {
      return rh_;
    }

    [[nodiscard]]
    Inline auto rg() const -> const real_t& {
      return rg_;
    }

    /**
     * Metric component 11.
     *
     * @param xi coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>&) const -> real_t {
      return dr_sqr;
    }

    /**
     * Metric component 22.
     *
     * @param xi coordinate array in code units
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& xi) const -> real_t {
      return dtheta_sqr * SQR(xi[0] * dr + this->x1_min);
    }

    /**
     * Metric component 33.
     *
     * @param xi coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dim2) {
        return SQR((xi[0] * dr + this->x1_min) *
                   math::sin(xi[1] * dtheta + this->x2_min));
      } else {
        return dphi_sqr * SQR((xi[0] * dr + this->x1_min) *
                              math::sin(xi[1] * dtheta + this->x2_min));
      }
    }

    /**
     * Metric component 13.
     *
     * @param xi coordinate array in code units
     * @returns h_13 (covariant, lower index) metric component.
     */
    Inline auto h_13(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    /**
     * Inverse metric component 11 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^11 (contravariant, upper index) metric component.
     */
    Inline auto h11(const coord_t<D>&) const -> real_t {
      return SQR(dr_inv);
    }

    /**
     * Inverse metric component 22 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^22 (contravariant, upper index) metric component.
     */
    Inline auto h22(const coord_t<D>& xi) const -> real_t {
      return SQR(dtheta_inv / (xi[0] * dr + this->x1_min));
    }

    /**
     * Inverse metric component 33 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^33 (contravariant, upper index) metric component.
     */
    Inline auto h33(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dim2) {
        return ONE / SQR((xi[0] * dr + this->x1_min) *
                         math::sin(xi[1] * dtheta + this->x2_min));
      } else {
        return SQR(dphi_inv) / SQR((xi[0] * dr + this->x1_min) *
                                   math::sin(xi[1] * dtheta + this->x2_min));
      }
    }

    /**
     * Inverse metric component 13 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^13 (contravariant, upper index) metric component.
     */
    Inline auto h13(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    /**
     * Lapse function.
     *
     * @param xi coordinate array in code units
     * @returns alpha.
     */
    Inline auto alpha(const coord_t<D>&) const -> real_t {
      return ONE;
    }

    /**
     * Radial component of shift vector.
     *
     * @param xi coordinate array in code units
     * @returns beta^1 (contravariant).
     */
    Inline auto beta1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    /**
     * Square root of the determinant of h-matrix.
     *
     * @param xi coordinate array in code units
     * @returns sqrt(det(h)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(xi[0] * dr + this->x1_min) *
               math::sin(xi[1] * dtheta + this->x2_min);
      } else {
        return dr * dtheta * dphi * SQR(xi[0] * dr + this->x1_min) *
               math::sin(xi[1] * dtheta + this->x2_min);
      }
    }

    /**
     * Square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param xi coordinate array in code units
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(xi[0] * dr + this->x1_min);
      } else {
        return dr * dtheta * dphi * SQR(xi[0] * dr + this->x1_min);
      }
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     * Approximate solution for the polar area.
     *
     * @param x1 coordinate in code units
     * @returns Area at the pole.
     */
    Inline auto polar_area(const real_t& x1) const -> real_t {
      return dr * SQR(x1 * dr + this->x1_min) * (ONE - math::cos(HALF * dtheta));
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a non-diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/param_forGR.h"
#include "metrics_utils/x_code_cart_forGSph.h"
#include "metrics_utils/x_code_phys_forGSph.h"
#include "metrics_utils/x_code_sph_forSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forGSph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forGR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forSph.h"

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     * @returns Minimum cell size of the grid [physical units].
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      if constexpr (D == Dim2) {
        real_t min_dx { -ONE };
        for (int i { 0 }; i < this->nx1; ++i) {
          for (int j { 0 }; j < this->nx2; ++j) {
            real_t        i_ { static_cast<real_t>(i) + HALF };
            real_t        j_ { static_cast<real_t>(j) + HALF };
            coord_t<Dim2> ij { i_, j_ };
            real_t        dx = ONE / (this->alpha(ij) *
                                 std::sqrt(this->h11(ij) + this->h22(ij)) +
                               this->beta1(ij));
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
} // namespace ntt

#endif