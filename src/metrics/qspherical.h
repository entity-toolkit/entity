/**
 * @file metrics/qspherical.h
 * @brief
 * Flat space-time qspherical metric class xi = log (r - r0), and eta,
 * where: theta = eta + 2h*eta * (PI - 2eta) * (PI - eta) / PI^2
 * @implements
 *   - ntt::QSpherical<> : ntt::MetricBase<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - metric_base.h
 *   - arch/kokkos_aliases.h
 *   - utils/comparators.h
 *   - utils/numeric.h
 * @includes:
 *   - metrics_utils/angle_stretch_forQSph.h
 *   - metrics_utils/param_forSR.h
 *   - metrics_utils/x_code_cart_forSRGSph.h
 *   - metrics_utils/x_code_phys_forGSph.h
 *   - metrics_utils/x_code_sph_forQSph.h
 *   - metrics_utils/v3_cart_hat_cntrv_cov_forSRGSph.h
 *   - metrics_utils/v3_hat_cntrv_cov_forSR.h
 *   - metrics_utils/v3_phys_cov_cntrv_forQSph.h
 * @namespaces:
 *   - ntt::
 * !TODO
 *   - 3D version of find_dxMin
 */

#ifndef METRICS_QSPHERICAL_H
#define METRICS_QSPHERICAL_H

#include "enums.h"
#include "global.h"
#include "metric_base.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/numeric.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <Dimension D>
  class QSpherical : public MetricBase<D, QSpherical<D>> {
    static_assert(D != Dim::_1D, "1D qspherical not available");
    static_assert(D != Dim::_3D, "3D qspherical not fully implemented");

  private:
    const real_t r0, h, chi_min, eta_min, phi_min;
    const real_t dchi, deta, dphi;
    const real_t dchi_inv, deta_inv, dphi_inv;
    const real_t dchi_sqr, deta_sqr, dphi_sqr;

  public:
    static constexpr std::string_view Label { "qspherical" };
    static constexpr Dimension        PrtlDim = Dim::_3D;
    static constexpr Coord::type   CoordType { Coord::QSPH };
    using MetricBase<D, QSpherical<D>>::x1_min;
    using MetricBase<D, QSpherical<D>>::x1_max;
    using MetricBase<D, QSpherical<D>>::x2_min;
    using MetricBase<D, QSpherical<D>>::x2_max;
    using MetricBase<D, QSpherical<D>>::x3_min;
    using MetricBase<D, QSpherical<D>>::x3_max;
    using MetricBase<D, QSpherical<D>>::nx1;
    using MetricBase<D, QSpherical<D>>::nx2;
    using MetricBase<D, QSpherical<D>>::nx3;
    using MetricBase<D, QSpherical<D>>::set_dxMin;

    QSpherical(std::vector<unsigned int>              res,
               std::vector<std::pair<real_t, real_t>> ext,
               const std::map<std::string, real_t>&   params) :
      MetricBase<D, QSpherical<D>> { res, ext },
      r0 { params.at("r0") },
      h { params.at("h") },
      chi_min { math::log(x1_min - r0) },
      eta_min { theta2eta(x2_min) },
      phi_min { x3_min },
      dchi { (math::log(x1_max - r0) - chi_min) / nx1 },
      deta { (theta2eta(x2_max) - eta_min) / nx2 },
      dphi { (x3_max - x3_min) / nx3 },
      dchi_inv { ONE / dchi },
      deta_inv { ONE / deta },
      dphi_inv { ONE / dphi },
      dchi_sqr { SQR(dchi) },
      deta_sqr { SQR(deta) },
      dphi_sqr { SQR(dphi) } {
      set_dxMin(find_dxMin());
    }

    ~QSpherical() = default;

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     *
     * @returns Minimum cell size of the grid [physical units].
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      // for 2D
      real_t min_dx { -1.0 };
      for (int i { 0 }; i < nx1; ++i) {
        for (int j { 0 }; j < nx2; ++j) {
          real_t i_ { (real_t)(i) + HALF };
          real_t j_ { (real_t)(j) + HALF };
          real_t dx1_ { h_11({ i_, j_ }) };
          real_t dx2_ { h_22({ i_, j_ }) };
          real_t dx = 1.0 / math::sqrt(1.0 / dx1_ + 1.0 / dx2_);
          if ((min_dx >= dx) || (min_dx < 0.0)) {
            min_dx = dx;
          }
        }
      }
      return min_dx;
    }

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>& x) const -> real_t {
      return dchi_sqr * math::exp(TWO * (x[0] * dchi + chi_min));
    }

    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& x) const -> real_t {
      return deta_sqr * SQR(dtheta_deta(x[1] * deta + eta_min)) *
             SQR(r0 + math::exp(x[0] * dchi + chi_min));
    }

    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return SQR((r0 + math::exp(x[0] * dchi + chi_min)) *
                   math::sin(eta2theta(x[1] * deta + eta_min)));
      } else if constexpr (D == Dim::_3D) {
        return dphi_sqr * SQR((r0 + math::exp(x[0] * dchi + chi_min)) *
                              math::sin(eta2theta(x[1] * deta + eta_min)));
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
        real_t exp_chi { math::exp(x[0] * dchi + chi_min) };
        return dchi * deta * exp_chi * dtheta_deta(x[1] * deta + eta_min) *
               SQR(r0 + exp_chi) * math::sin(eta2theta(x[1] * deta + eta_min));
      } else if constexpr (D == Dim::_3D) {
        real_t exp_chi { math::exp(x[0] * dchi + chi_min) };
        return dchi * deta * dphi * exp_chi * dtheta_deta(x[1] * deta + eta_min) *
               SQR(r0 + exp_chi) * math::sin(eta2theta(x[1] * deta + eta_min));
      }
    }

    /**
     * Square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param x coordinate array in code units
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      if constexpr (D != Dim::_1D) {
        real_t exp_chi { math::exp(x[0] * dchi + chi_min) };
        return dchi * deta * exp_chi * dtheta_deta(x[1] * deta + eta_min) *
               SQR(r0 + exp_chi);
      }
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     *
     * @param x1 radial coordinate along the axis (code units).
     * @returns Area at the pole.
     */
    Inline auto polar_area(const real_t& x1) const -> real_t {
      if constexpr (D != Dim::_1D) {
        real_t exp_chi { math::exp(x1 * dchi + chi_min) };
        return dchi * exp_chi * SQR(r0 + exp_chi) *
               (ONE - math::cos(eta2theta(HALF * deta)));
      }
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/angle_stretch_forQSph.h"
#include "metrics_utils/x_code_cart_forSRGSph.h"
#include "metrics_utils/x_code_phys_forGSph.h"
#include "metrics_utils/x_code_sph_forQSph.h"

#include "metrics_utils/v3_cart_hat_cntrv_cov_forSRGSph.h"
#include "metrics_utils/v3_hat_cntrv_cov_forSR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forQSph.h"
  };
} // namespace ntt

#endif // METRICS_QSPHERICAL_H