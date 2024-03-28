/**
 * @file faraday_sr.hpp
 * @brief Algorithms for Faraday's law in curvilinear SR
 * @implements
 *   - ntt::Faraday_kernel<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/kokkos_aliases.h
 *   - utils/error.h
 *   - utils/log.h
 *   - utils/numeric.h
 * @namespaces:
 *   - ntt::
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_FARADAY_SR_H
#define KERNELS_FARADAY_SR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

namespace ntt {

  /**
   * @brief Algorithm for the Faraday's law: `dB/dt = -curl E` in Curvilinear
   * space (diagonal metric).
   * @tparam D Dimension.
   */
  template <Dimension D, class M>
  class Faraday_kernel {
    ndfield_t<D, 6> EB;
    const M         metric;
    const real_t    coeff;
    bool            is_axis_i2min { false };

  public:
    Faraday_kernel(const ndfield_t<D, 6>&                          EB,
                   const M&                                        metric,
                   real_t                                          coeff,
                   const std::vector<std::vector<FieldsBC::type>>& boundaries) :
      EB { EB },
      metric { metric },
      coeff { coeff } {
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1][0] == FieldsBC::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        constexpr std::size_t i2min { N_GHOSTS };
        const real_t          i1_ { COORD(i1) };
        const real_t          i2_ { COORD(i2) };

        const real_t inv_sqrt_detH_0pH { ONE /
                                         metric.sqrt_det_h({ i1_, i2_ + HALF }) };
        const real_t inv_sqrt_detH_pHpH { ONE / metric.sqrt_det_h(
                                                  { i1_ + HALF, i2_ + HALF }) };
        const real_t h1_pHp1 { metric.h_11({ i1_ + HALF, i2_ + ONE }) };
        const real_t h1_pH0 { metric.h_11({ i1_ + HALF, i2_ }) };
        const real_t h2_p1pH { metric.h_22({ i1_ + ONE, i2_ + HALF }) };
        const real_t h2_0pH { metric.h_22({ i1_, i2_ + HALF }) };
        const real_t h3_00 { metric.h_33({ i1_, i2_ }) };
        const real_t h3_0p1 { metric.h_33({ i1_, i2_ + ONE }) };

        EB(i1, i2, em::bx1) += coeff * inv_sqrt_detH_0pH *
                               (h3_00 * EB(i1, i2, em::ex3) -
                                h3_0p1 * EB(i1, i2 + 1, em::ex3));
        if ((i2 != i2min) || !is_axis_i2min) {
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };
          const real_t h3_p10 { metric.h_33({ i1_ + ONE, i2_ }) };
          EB(i1, i2, em::bx2) += coeff * inv_sqrt_detH_pH0 *
                                 (h3_p10 * EB(i1 + 1, i2, em::ex3) -
                                  h3_00 * EB(i1, i2, em::ex3));
        }
        EB(i1, i2, em::bx3) += coeff * inv_sqrt_detH_pHpH *
                               (h1_pHp1 * EB(i1, i2 + 1, em::ex1) -
                                h1_pH0 * EB(i1, i2, em::ex1) +
                                h2_0pH * EB(i1, i2, em::ex2) -
                                h2_p1pH * EB(i1 + 1, i2, em::ex2));
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t, index_t, index_t) const {
      if constexpr (D == Dim::_3D) {
        raise::KernelNotImplementedError(HERE);
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace ntt

#endif // KERNELS_FARADAY_SR_H