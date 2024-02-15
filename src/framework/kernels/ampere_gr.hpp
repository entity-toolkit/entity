#ifndef FRAMEWORK_KERNELS_AMPERE_GR_H
#define FRAMEWORK_KERNELS_AMPERE_GR_H

#include "wrapper.h"

#include "meshblock/fields.h"

namespace ntt {

  /**
   * @brief Algorithms for Ampere's law:
   * @brief `d(Din)^i / dt = curl H_j`, `Dout += dt * d(Din)/dt`.
   * @tparam D Dimension.
   * @tparam M Metric.
   */
  template <Dimension D, class M>
  class Ampere_kernel {
    ndfield_t<D, 6>   Din;
    ndfield_t<D, 6>   Dout;
    ndfield_t<D, 6>   H;
    const M           metric;
    const std::size_t i2max;
    const real_t      coeff;
    bool              is_axis_i2min { false }, is_axis_i2max { false };

  public:
    Ampere_kernel(const ndfield_t<D, 6>& Din,
                  const ndfield_t<D, 6>& Dout,
                  const ndfield_t<D, 6>& H,
                  const M&               metric,
                  real_t                 coeff,
                  std::size_t            ni2,
                  const std::vector<std::vector<BoundaryCondition>>& boundaries) :
      Din { Din },
      Dout { Dout },
      H { H },
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

        if ((i2 == i2min) && is_axis_i2min) {
          // theta = 0
          const real_t inv_polar_area_pH { ONE / metric.polar_area(i1_ + HALF) };
          Dout(i1, i2, em::dx1) = Din(i1, i2, em::dx1) +
                                  inv_polar_area_pH * coeff * H(i1, i2, em::hx3);
          Dout(i1, i2, em::dx2) = Din(i1, i2, em::dx2) +
                                  coeff * inv_sqrt_detH_0pH *
                                    (H(i1 - 1, i2, em::hx3) - H(i1, i2, em::hx3));
        } else if ((i2 == i2max) && is_axis_i2max) {
          // theta = pi
          const real_t inv_polar_area_pH { ONE / metric.polar_area(i1_ + HALF) };
          Dout(i1, i2, em::dx1) = Din(i1, i2, em::dx1) - inv_polar_area_pH * coeff *
                                                           H(i1, i2 - 1, em::hx3);
        } else {
          const real_t inv_sqrt_detH_00 { ONE / metric.sqrt_det_h({ i1_, i2_ }) };
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };
          Dout(i1, i2, em::dx1) = Din(i1, i2, em::dx1) +
                                  coeff * inv_sqrt_detH_pH0 *
                                    (H(i1, i2, em::hx3) - H(i1, i2 - 1, em::hx3));
          Dout(i1, i2, em::dx2) = Din(i1, i2, em::dx2) +
                                  coeff * inv_sqrt_detH_0pH *
                                    (H(i1 - 1, i2, em::hx3) - H(i1, i2, em::hx3));
          Dout(i1, i2, em::dx3) = Din(i1, i2, em::dx3) +
                                  coeff * inv_sqrt_detH_00 *
                                    ((H(i1, i2 - 1, em::hx1) - H(i1, i2, em::hx1)) +
                                     (H(i1, i2, em::hx2) - H(i1 - 1, i2, em::hx2)));
        }
      } else {
        NTTError("Ampere_kernel: 2D implementation called for D != 2");
      }
    }
  };

  /**
   * @brief Add the currents to the D field with the appropriate conversion.
   * @tparam D Dimension.
   */
  template <Dimension D, class M>
  class CurrentsAmpere_kernel {
    ndfield_t<D, 6>   Df;
    ndfield_t<D, 3>   J;
    const M           metric;
    const std::size_t i2max;
    const real_t      coeff;
    bool              is_axis_i2min { false };
    bool              is_axis_i2max { false };

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock.
     */
    CurrentsAmpere_kernel(
      const ndfield_t<D, 6>&                             Df,
      const ndfield_t<D, 3>&                             J,
      const M&                                           metric,
      real_t                                             coeff,
      std::size_t                                        ni2,
      const std::vector<std::vector<BoundaryCondition>>& boundaries) :
      Df { Df },
      J { J },
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

        if ((i2 == i2min) && is_axis_i2min) {
          // theta = 0 (first active cell)
          Df(i1, i2, em::dx1) += J(i1, i2, cur::jx1) * HALF * coeff /
                                 metric.polar_area(i1_ + HALF);
          Df(i1, i2, em::dx2) += J(i1, i2, cur::jx2) * coeff * inv_sqrt_detH_0pH;
        } else if ((i2 == i2max) && is_axis_i2max) {
          // theta = pi (first ghost cell from end)
          Df(i1, i2, em::dx1) += J(i1, i2, cur::jx1) * HALF * coeff /
                                 metric.polar_area(i1_ + HALF);
        } else {
          // 0 < theta < pi
          const real_t inv_sqrt_detH_00 { ONE / metric.sqrt_det_h({ i1_, i2_ }) };
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };

          Df(i1, i2, em::dx1) += J(i1, i2, cur::jx1) * coeff * inv_sqrt_detH_pH0;
          Df(i1, i2, em::dx2) += J(i1, i2, cur::jx2) * coeff * inv_sqrt_detH_0pH;
          Df(i1, i2, em::dx3) += J(i1, i2, cur::jx3) * coeff * inv_sqrt_detH_00;
        }
      }

      else {
        NTTError("CurrentsAmpere_kernel: 2D implementation called for D != 2");
      }
    }
  };

} // namespace ntt

#endif // FRAMEWORK_KERNELS_AMPERE_GR_H
