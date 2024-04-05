/**
 * @file metrics/spherical.h
 * @brief Flat space-time spherical metric class diag(-1, 1, r^2, r^2, sin(th)^2)
 * @implements
 *   - ntt::Spherical<> : ntt::MetricBase<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - metric_base.h
 *   - arch/kokkos_aliases.h
 *   - utils/error.h
 *   - utils/log.h
 *   - utils/numeric.h
 * @includes:
 *   - metrics_utils/param_forSR.h
 *   - metrics_utils/x_code_cart_forSRGSph.h
 *   - metrics_utils/x_code_phys_forGSph.h
 *   - metrics_utils/x_code_sph_forSph.h
 *   - metrics_utils/v3_cart_hat_cntrv_cov_forSRGSph.h
 *   - metrics_utils/v3_hat_cntrv_cov_forSR.h
 *   - metrics_utils/v3_phys_cov_cntrv_forSph.h
 * @namespaces:
 *   - ntt::
 * !TODO
 *   - 3D version of find_dxMin
 */

#ifndef METRICS_SPHERICAL_H
#define METRICS_SPHERICAL_H

#include "enums.h"
#include "global.h"
#include "metric_base.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <Dimension D>
  class Spherical : public MetricBase<D, Spherical<D>> {
    static_assert(D != Dim::_1D, "1D spherical not available");
    static_assert(D != Dim::_3D, "3D spherical not fully implemented");

  private:
    const real_t dr, dtheta, dphi;
    const real_t dr_inv, dtheta_inv, dphi_inv;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;

  public:
    static constexpr std::string_view Label { "spherical" };
    static constexpr Dimension        PrtlDim = Dim::_3D;
    static constexpr Coord            CoordType { Coord::Sph };
    using MetricBase<D, Spherical<D>>::x1_min;
    using MetricBase<D, Spherical<D>>::x1_max;
    using MetricBase<D, Spherical<D>>::x2_min;
    using MetricBase<D, Spherical<D>>::x2_max;
    using MetricBase<D, Spherical<D>>::x3_min;
    using MetricBase<D, Spherical<D>>::x3_max;
    using MetricBase<D, Spherical<D>>::nx1;
    using MetricBase<D, Spherical<D>>::nx2;
    using MetricBase<D, Spherical<D>>::nx3;
    using MetricBase<D, Spherical<D>>::set_dxMin;

    Spherical(std::vector<unsigned int>              res,
              std::vector<std::pair<real_t, real_t>> ext,
              const std::map<std::string, real_t>& = {}) :
      MetricBase<D, Spherical<D>> { res, ext },
      dr((x1_max - x1_min) / nx1),
      dtheta((x2_max - x2_min) / nx2),
      dphi((x3_max - x3_min) / nx3),
      dr_inv { ONE / dr },
      dtheta_inv { ONE / dtheta },
      dphi_inv { ONE / dphi },
      dr_sqr { SQR(dr) },
      dtheta_sqr { SQR(dtheta) },
      dphi_sqr { SQR(dphi) } {
      set_dxMin(find_dxMin());
    }

    ~Spherical() = default;

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
      return dtheta_sqr * SQR(x[0] * dr + x1_min);
    }

    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return SQR(x[0] * dr + x1_min) * SQR(math::sin(x[1] * dtheta + x2_min));
      } else if constexpr (D == Dim::_3D) {
        return dphi_sqr * SQR(x[0] * dr + x1_min) *
               SQR(math::sin(x[1] * dtheta + x2_min));
      }
    }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * SQR(x[0] * dr + x1_min) *
               math::sin(x[1] * dtheta + x2_min);
      } else if constexpr (D == Dim::_3D) {
        return dr * dtheta * dphi * SQR(x[0] * dr + x1_min) *
               math::sin(x[1] * dtheta + x2_min);
      }
    }

    /**
     * Square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * SQR(x[0] * dr + x1_min);
      } else if constexpr (D == Dim::_3D) {
        return dr * dtheta * dphi * SQR(x[0] * dr + x1_min);
      }
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     *
     * @param x1 radial coordinate along the axis (code units).
     * @returns Area at the pole.
     */
    Inline auto polar_area(const real_t& x1) const -> real_t {
      return dr * SQR(x1 * dr + x1_min) * (ONE - math::cos(HALF * dtheta));
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/x_code_cart_forSRGSph.h"
#include "metrics_utils/x_code_phys_forGSph.h"
#include "metrics_utils/x_code_sph_forSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forSRGSph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forSR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forSph.h"

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      // for 2D
      auto dx1 { dr };
      auto dx2 { x1_min * dtheta };
      return ONE / math::sqrt(ONE / SQR(dx1) + ONE / SQR(dx2));
    }
  };

} // namespace ntt

#endif // METRICS_SPHERICAL_H
