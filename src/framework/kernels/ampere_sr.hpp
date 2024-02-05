#ifndef FRAMEWORK_KERNELS_AMPERE_SR_H
#define FRAMEWORK_KERNELS_AMPERE_SR_H

#include "wrapper.h"

#include "meshblock/fields.h"

namespace ntt {
  /**
   * @brief Algorithm for the Ampere's law: `dE/dt = curl B` in curvilinear space.
   * @tparam D Dimension.
   * @tparam M Metric.
   */
  template <Dimension D, class M>
  class Ampere_kernel {
    ndfield_t<D, 6>   EB;
    const M           metric;
    const std::size_t i2max;
    const real_t      coeff;
    bool              is_axis_i2min { false }, is_axis_i2max { false };

  public:
    Ampere_kernel(const ndfield_t<D, 6>& EB,
                  const M&               metric,
                  real_t                 coeff,
                  std::size_t            ni2,
                  const std::vector<std::vector<BoundaryCondition>>& boundaries) :
      EB { EB },
      metric { metric },
      i2max { ni2 + N_GHOSTS },
      coeff { coeff } {
      if constexpr ((D == Dim2) || (D == Dim3)) {
        NTTHostErrorIf(boundaries.size() < 2, "boundaries defined incorrectly");
        is_axis_i2min = (boundaries[1][0] == BoundaryCondition::AXIS);
        is_axis_i2max = (boundaries[1][1] == BoundaryCondition::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim2) {
        constexpr std::size_t i2min { N_GHOSTS };
        const real_t          i1_ { COORD(i1) };
        const real_t          i2_ { COORD(i2) };

        const real_t inv_sqrt_detH_0pH { ONE /
                                         metric.sqrt_det_h({ i1_, i2_ + HALF }) };
        const real_t h3_mHpH { metric.h_33({ i1_ - HALF, i2_ + HALF }) };
        const real_t h3_pHpH { metric.h_33({ i1_ + HALF, i2_ + HALF }) };

        if ((i2 == i2min) && is_axis_i2min) {
          // theta = 0
          const real_t inv_polar_area_pH0 { ONE / metric.polar_area(i1_ + HALF) };
          EB(i1, i2, em::ex1) += inv_polar_area_pH0 * coeff *
                                 (h3_pHpH * EB(i1, i2, em::bx3));
          EB(i1, i2, em::ex2) += coeff * inv_sqrt_detH_0pH *
                                 (h3_mHpH * EB(i1 - 1, i2, em::bx3) -
                                  h3_pHpH * EB(i1, i2, em::bx3));
        } else if ((i2 == i2max) && is_axis_i2max) {
          // theta = pi
          const real_t inv_polar_area_pH0 { ONE / metric.polar_area(i1_ + HALF) };
          const real_t h3_pHmH { metric.h_33({ i1_ + HALF, i2_ - HALF }) };
          EB(i1, i2, em::ex1) -= inv_polar_area_pH0 * coeff *
                                 (h3_pHmH * EB(i1, i2, em::bx3));
        } else {
          const real_t inv_sqrt_detH_00 { ONE / metric.sqrt_det_h({ i1_, i2_ }) };
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };
          const real_t h1_0mH { metric.h_11({ i1_, i2_ - HALF }) };
          const real_t h1_0pH { metric.h_11({ i1_, i2_ + HALF }) };
          const real_t h2_pH0 { metric.h_22({ i1_ + HALF, i2_ }) };
          const real_t h2_mH0 { metric.h_22({ i1_ - HALF, i2_ }) };
          const real_t h3_pHmH { metric.h_33({ i1_ + HALF, i2_ - HALF }) };
          EB(i1, i2, em::ex1) += coeff * inv_sqrt_detH_pH0 *
                                 (h3_pHpH * EB(i1, i2, em::bx3) -
                                  h3_pHmH * EB(i1, i2 - 1, em::bx3));
          EB(i1, i2, em::ex2) += coeff * inv_sqrt_detH_0pH *
                                 (h3_mHpH * EB(i1 - 1, i2, em::bx3) -
                                  h3_pHpH * EB(i1, i2, em::bx3));
          EB(i1, i2, em::ex3) += coeff * inv_sqrt_detH_00 *
                                 (h1_0mH * EB(i1, i2 - 1, em::bx1) -
                                  h1_0pH * EB(i1, i2, em::bx1) +
                                  h2_pH0 * EB(i1, i2, em::bx2) -
                                  h2_mH0 * EB(i1 - 1, i2, em::bx2));
        }
      } else {
        NTTError("Ampere_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t, index_t, index_t) const {
      if constexpr (D == Dim3) {
        NTTError("not implemented");
      } else {
        NTTError("Ampere_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Add the currents to the E field with the appropriate conversion.
   * @tparam D Dimension.
   */
  template <Dimension D, class M>
  class CurrentsAmpere_kernel {
    ndfield_t<D, 6>   E;
    ndfield_t<D, 3>   J;
    const M           metric;
    const std::size_t i2max;
    const real_t      coeff;
    const real_t      inv_n0;
    bool              is_axis_i2min { false };
    bool              is_axis_i2max { false };

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    CurrentsAmpere_kernel(
      const ndfield_t<D, 6>&                             E,
      const ndfield_t<D, 3>&                             J,
      const M&                                           metric,
      real_t                                             coeff,
      real_t                                             inv_n0,
      std::size_t                                        ni2,
      const std::vector<std::vector<BoundaryCondition>>& boundaries) :
      E { E },
      J { J },
      metric { metric },
      i2max { ni2 + N_GHOSTS },
      coeff { coeff },
      inv_n0 { inv_n0 } {
      if constexpr ((D == Dim2) || (D == Dim3)) {
        NTTHostErrorIf(boundaries.size() < 2, "boundaries defined incorrectly");
        is_axis_i2min = (boundaries[1][0] == BoundaryCondition::AXIS);
        is_axis_i2max = (boundaries[1][1] == BoundaryCondition::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim2) {
        constexpr std::size_t i2min { N_GHOSTS };
        const real_t          i1_ { COORD(i1) };
        const real_t          i2_ { COORD(i2) };

        // convert the "curly" current, to contravariant, normalized to
        // `J0=n0*q0` then add "curly" current with the right coefficient

        if ((i2 == i2min) && is_axis_i2min) {
          // theta = 0 (first active cell)
          // r
          J(i1, i2, cur::jx1) *= inv_n0 * HALF / metric.polar_area(i1_ + HALF);

          // theta
          J(i1, i2, cur::jx2) *= inv_n0 / metric.sqrt_det_h({ i1_, i2_ + HALF });
          E(i1, i2, em::ex2) += J(i1, i2, cur::jx2) * coeff;

          // phi
          J(i1, i2, cur::jx3) = ZERO;
        } else if ((i2 == i2max) && is_axis_i2max) {
          // theta = pi (first ghost cell from end)
          // r
          J(i1, i2, cur::jx1) *= inv_n0 * HALF / metric.polar_area(i1_ + HALF);

          // phi
          J(i1, i2, cur::jx3) = ZERO;
        } else {
          // 0 < theta < pi
          // r
          J(i1, i2, cur::jx1) *= inv_n0 / metric.sqrt_det_h({ i1_ + HALF, i2_ });

          // theta
          J(i1, i2, cur::jx2) *= inv_n0 / metric.sqrt_det_h({ i1_, i2_ + HALF });
          E(i1, i2, em::ex2) += J(i1, i2, cur::jx2) * coeff;

          // phi
          J(i1, i2, cur::jx3) *= inv_n0 / metric.sqrt_det_h({ i1_, i2_ });
          E(i1, i2, em::ex3)  += J(i1, i2, cur::jx3) * coeff;
        }

        E(i1, i2, em::ex1) += J(i1, i2, cur::jx1) * coeff;
      }

      else {
        NTTError("CurrentsAmpere_kernel: 2D implementation called for D != 2");
      }
    }
  };

} // namespace ntt

#endif // FRAMEWORK_KERNELS_AMPERE_SR_H
