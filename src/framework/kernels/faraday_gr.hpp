#ifndef FRAMEWORK_KERNELS_FARADAY_GR_H
#define FRAMEWORK_KERNELS_FARADAY_GR_H

#include "wrapper.h"

#include "meshblock/fields.h"

namespace ntt {

  /**
   * @brief Algorithms for Faraday's law:
   * @brief `d(Bin)^i / dt = -curl E_j`, `Bout += dt * d(Bin)/dt`.
   * @tparam D Dimension.
   * @tparam M Metric.
   */
  template <Dimension D, class M>
  class Faraday_kernel {
    ndfield_t<D, 6>   DBin;
    ndfield_t<D, 6>   DBout;
    ndfield_t<D, 6>   EH;
    const M           metric;
    const std::size_t i2max;
    const real_t      coeff;
    bool              is_axis_i2min { false };

  public:
    Faraday_kernel(const ndfield_t<D, 6>& DBin,
                   const ndfield_t<D, 6>& DBout,
                   const ndfield_t<D, 6>& EH,
                   const M&               metric,
                   real_t                 coeff,
                   std::size_t            ni2,
                   const std::vector<std::vector<BoundaryCondition>>& boundaries) :
      DBin { DBin },
      DBout { DBout },
      EH { EH },
      metric { metric },
      i2max { ni2 + N_GHOSTS },
      coeff { coeff } {
      if constexpr ((D == Dim2) || (D == Dim3)) {
        NTTHostErrorIf(boundaries.size() < 2, "boundaries defined incorrectly");
        is_axis_i2min = (boundaries[1][0] == BoundaryCondition::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim2) {
        constexpr std::size_t i2min { N_GHOSTS };
        const real_t          i1_ { COORD(i1) };
        const real_t          i2_ { COORD(i2) };
        const real_t          inv_sqrt_detH_0pH { ONE /
                                         metric.sqrt_det_h({ i1_, i2_ + HALF }) };
        const real_t          inv_sqrt_detH_pHpH { ONE / metric.sqrt_det_h(
                                                  { i1_ + HALF, i2_ + HALF }) };

        DBout(i1, i2, em::bx1) = DBin(i1, i2, em::bx1) +
                                 coeff * inv_sqrt_detH_0pH *
                                   (EH(i1, i2, em::ex3) - EH(i1, i2 + 1, em::ex3));
        if ((i2 != i2min) || !is_axis_i2min) {
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };
          DBout(i1, i2, em::bx2) = DBin(i1, i2, em::bx2) +
                                   coeff * inv_sqrt_detH_pH0 *
                                     (EH(i1 + 1, i2, em::ex3) -
                                      EH(i1, i2, em::ex3));
        }

        DBout(i1, i2, em::bx3) = DBin(i1, i2, em::bx3) +
                                 coeff * inv_sqrt_detH_pHpH *
                                   (EH(i1, i2 + 1, em::ex1) - EH(i1, i2, em::ex1) +
                                    EH(i1, i2, em::ex2) - EH(i1 + 1, i2, em::ex2));
      } else {
        NTTError("Faraday_kernel: 2D implementation called for D != 2");
      }
    }
  };

} // namespace ntt

#endif // FRAMEWORK_KERNELS_FARADAY_GR_H
