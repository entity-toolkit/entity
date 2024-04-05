/**
 * @file metrics/kerr_schild_0.h
 * @brief
 * Kerr metric with zero spin and zero mass
 * in Kerr-Schild coordinates (rg=c=1)
 * @implements
 *   - ntt::KerrSchild0<> : ntt::MetricBase<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - metric_base.h
 *   - arch/kokkos_aliases.h
 *   - utils/numeric.h
 * @includes:
 *   - metrics_utils/x_code_phys_forGSph.h
 *   - metrics_utils/x_code_sph_forSph.h
 *   - metrics_utils/v3_hat_cntrv_cov_forGR.h
 *   - metrics_utils/v3_phys_cov_cntrv_forSph.h
 * @namespaces:
 *   - ntt::
 * !TODO
 *   - 3D version of find_dxMin
 */

#ifndef METRICS_KERR_SCHILD_0_H
#define METRICS_KERR_SCHILD_0_H

#include "enums.h"
#include "global.h"
#include "metric_base.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <Dimension D>
  class KerrSchild0 : public MetricBase<D, KerrSchild0<D>> {
    static_assert(D != Dim::_1D, "1D kerr_schild_0 not available");
    static_assert(D != Dim::_3D, "3D kerr_schild_0 not fully implemented");

  private:
    const real_t dr, dtheta, dphi;
    const real_t dr_inv, dtheta_inv, dphi_inv;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;

  public:
    static constexpr std::string_view Label { "kerr_schild_0" };
    static constexpr Dimension        PrtlDim { D };
    static constexpr Coord            CoordType { Coord::Sph };
    using MetricBase<D, KerrSchild0<D>>::x1_min;
    using MetricBase<D, KerrSchild0<D>>::x1_max;
    using MetricBase<D, KerrSchild0<D>>::x2_min;
    using MetricBase<D, KerrSchild0<D>>::x2_max;
    using MetricBase<D, KerrSchild0<D>>::x3_min;
    using MetricBase<D, KerrSchild0<D>>::x3_max;
    using MetricBase<D, KerrSchild0<D>>::nx1;
    using MetricBase<D, KerrSchild0<D>>::nx2;
    using MetricBase<D, KerrSchild0<D>>::nx3;
    using MetricBase<D, KerrSchild0<D>>::set_dxMin;

    KerrSchild0(std::vector<unsigned int>              res,
                std::vector<std::pair<real_t, real_t>> ext,
                const std::map<std::string, real_t>& = {}) :
      MetricBase<D, KerrSchild0<D>> { res, ext },
      dr { (x1_max - x1_min) / nx1 },
      dtheta { (x2_max - x2_min) / nx2 },
      dphi { (x3_max - x3_min) / nx3 },
      dr_inv { ONE / dr },
      dtheta_inv { ONE / dtheta },
      dphi_inv { ONE / dphi },
      dr_sqr { dr * dr },
      dtheta_sqr { dtheta * dtheta },
      dphi_sqr { dphi * dphi } {
      set_dxMin(find_dxMin());
    }

    ~KerrSchild0() = default;

    [[nodiscard]]
    Inline auto spin() const -> const real_t& {
      return ZERO;
    }

    [[nodiscard]]
    Inline auto rhorizon() const -> const real_t& {
      return ZERO;
    }

    [[nodiscard]]
    Inline auto rg() const -> const real_t& {
      return ZERO;
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
      return dtheta_sqr * SQR(xi[0] * dr + x1_min);
    }

    /**
     * Metric component 33.
     *
     * @param xi coordinate array in code units
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return SQR((xi[0] * dr + x1_min) * math::sin(xi[1] * dtheta + x2_min));
      } else {
        return dphi_sqr *
               SQR((xi[0] * dr + x1_min) * math::sin(xi[1] * dtheta + x2_min));
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
      return SQR(dtheta_inv / (xi[0] * dr + x1_min));
    }

    /**
     * Inverse metric component 33 from h_ij.
     *
     * @param xi coordinate array in code units
     * @returns h^33 (contravariant, upper index) metric component.
     */
    Inline auto h33(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return ONE /
               SQR((xi[0] * dr + x1_min) * math::sin(xi[1] * dtheta + x2_min));
      } else {
        return SQR(dphi_inv) /
               SQR((xi[0] * dr + x1_min) * math::sin(xi[1] * dtheta + x2_min));
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
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * SQR(xi[0] * dr + x1_min) *
               math::sin(xi[1] * dtheta + x2_min);
      } else {
        return dr * dtheta * dphi * SQR(xi[0] * dr + x1_min) *
               math::sin(xi[1] * dtheta + x2_min);
      }
    }

    /**
     * Square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param xi coordinate array in code units
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * SQR(xi[0] * dr + x1_min);
      } else {
        return dr * dtheta * dphi * SQR(xi[0] * dr + x1_min);
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
      return dr * SQR(x1 * dr + x1_min) * (ONE - math::cos(HALF * dtheta));
    }

/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a non-diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/x_code_phys_forGSph.h"
#include "metrics_utils/x_code_sph_forSph.h"

#include "metrics_utils/v3_hat_cntrv_cov_forGR.h"
#include "metrics_utils/v3_phys_cov_cntrv_forSph.h"

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     * @returns Minimum cell size of the grid [physical units].
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      // for 2D
      real_t min_dx { -ONE };
      for (int i { 0 }; i < nx1; ++i) {
        for (int j { 0 }; j < nx2; ++j) {
          real_t            i_ { static_cast<real_t>(i) + HALF };
          real_t            j_ { static_cast<real_t>(j) + HALF };
          coord_t<Dim::_2D> ij { i_, j_ };
          real_t dx = ONE / (alpha(ij) * std::sqrt(h11(ij) + h22(ij)) + beta1(ij));
          if ((min_dx > dx) || (min_dx < 0.0)) {
            min_dx = dx;
          }
        }
      }
      return min_dx;
    }
  };
} // namespace ntt

#endif // METRICS_KERR_SCHILD_0_H