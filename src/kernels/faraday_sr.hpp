/**
 * @file kernels/faraday_sr.hpp
 * @brief Algorithms for Faraday's law in curvilinear SR
 * @implements
 *   - kernel::sr::Faraday_kernel<>
 * @namespaces:
 *   - kernel::sr::
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_FARADAY_SR_HPP
#define KERNELS_FARADAY_SR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

// #define DIFF_4(ymm, ym, yp, ypp) (-1.125*ym + 0.04166666666666666*ymm + 1.125*yp - 0.04166666666666666*ypp)

namespace kernel::sr {
  using namespace ntt;

  /**
   * @brief Algorithm for the 's law: `dB/dt = -curl E` in Curvilinear
   * space (diagonal metric)
   */
  template <class M>
  class Faraday_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    ndfield_t<D, 6> EB;
    const M         metric;
    const std::size_t i1max;
    const std::size_t i2max;
    const real_t    coeff;
    bool            is_axis_i2min { false };

  public:
    Faraday_kernel(const ndfield_t<D, 6>&      EB,
                   const M&                    metric,
                   real_t                      coeff,
                  std::size_t                 ni1,
                  std::size_t                 ni2,
                   const boundaries_t<FldsBC>& boundaries)
      : EB { EB }
      , metric { metric }
      , i1max { ni1 + N_GHOSTS }
      , i2max { ni2 + N_GHOSTS }
      , coeff { coeff } {
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        constexpr std::size_t i2min { N_GHOSTS };
        constexpr std::size_t i1min { N_GHOSTS };
        const real_t          i1_ { COORD(i1) };
        const real_t          i2_ { COORD(i2) };

        const real_t inv_sqrt_detH_0pH { ONE /
                                         metric.sqrt_det_h({ i1_, i2_ + HALF }) };
        const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h( { i1_ + HALF, i2_ }) };
        const real_t inv_sqrt_detH_pHpH { ONE / metric.sqrt_det_h({ i1_ + HALF, i2_ + HALF }) };
  
        const real_t h1_pHm1 { metric.template h_<1, 1>({ i1_ + HALF, i2_ - ONE }) };
        const real_t h1_pH0 { metric.template h_<1, 1>({ i1_ + HALF, i2_ }) };
        const real_t h1_pHp1 { metric.template h_<1, 1>({ i1_ + HALF, i2_ + ONE }) };
        const real_t h1_pHp2 { metric.template h_<1, 1>({ i1_ + HALF, i2_ + TWO }) };

        const real_t h2_m1pH { metric.template h_<2, 2>({ i1_ - ONE, i2_ + HALF }) };
        const real_t h2_0pH { metric.template h_<2, 2>({ i1_, i2_ + HALF }) };
        const real_t h2_p1pH { metric.template h_<2, 2>({ i1_ + ONE, i2_ + HALF }) };
        const real_t h2_p2pH { metric.template h_<2, 2>({ i1_ + TWO, i2_ + HALF }) };

        const real_t h3_0m1 { metric.template h_<3, 3>({ i1_, i2_ - ONE }) };
        const real_t h3_00 { metric.template h_<3, 3>({ i1_, i2_ }) };
        const real_t h3_0p1 { metric.template h_<3, 3>({ i1_, i2_ + ONE }) };
        const real_t h3_0p2 { metric.template h_<3, 3>({ i1_, i2_ + TWO }) };
        const real_t h3_m10 { metric.template h_<3, 3>({ i1_ - ONE, i2_ }) };
        const real_t h3_p10 { metric.template h_<3, 3>({ i1_ + ONE, i2_ }) };
        const real_t h3_p20 { metric.template h_<3, 3>({ i1_ + TWO, i2_ }) };

        // If it fits, do fourth order stencil 
        if (i1 > i1min + 10 && i2 > i2min + 10 && i1 < i1max - 10 && i2 < i2max - 10 ) {

          const real_t amm = h3_0m1*EB(i1, i2 - 1, em::ex3);
          const real_t am = h3_00*EB(i1, i2, em::ex3);
          const real_t ap = h3_0p1*EB(i1, i2 + 1, em::ex3);
          const real_t app = h3_0p2*EB(i1, i2 + 2, em::ex3);    

          const real_t curlEr = inv_sqrt_detH_0pH * (-1.125*am + 0.04166666666666666*amm + 1.125*ap - 0.04166666666666666*app);

          const real_t bmm = h3_m10*EB(i1 - 1, i2, em::ex3);
          const real_t bm = h3_00*EB(i1, i2, em::ex3);
          const real_t bp = h3_p10*EB(i1 + 1, i2, em::ex3);
          const real_t bpp = h3_p20*EB(i1 + 2, i2, em::ex3);    

          const real_t curlEt = inv_sqrt_detH_pH0 * (-1.125*bm + 0.04166666666666666*bmm + 1.125*bp - 0.04166666666666666*bpp);

          const real_t cmm = h2_m1pH*EB(i1 - 1, i2, em::ex2);
          const real_t cm = h2_0pH*EB(i1, i2, em::ex2);
          const real_t cp = h2_p1pH*EB(i1 + 1, i2, em::ex2);
          const real_t cpp = h2_p2pH*EB(i1 + 2, i2, em::ex2); 

          const real_t dmm = h1_pHm1*EB(i1, i2 - 1, em::ex1);
          const real_t dm = h1_pH0*EB(i1, i2, em::ex1);
          const real_t dp = h1_pHp1*EB(i1, i2 + 1, em::ex1);
          const real_t dpp = h1_pHp2*EB(i1, i2 + 2, em::ex1);   

          const real_t curlEp = inv_sqrt_detH_pHpH * ((-1.125*cm + 0.04166666666666666*cmm + 1.125*cp - 0.04166666666666666*cpp) - (-1.125*dm + 0.04166666666666666*dmm + 1.125*dp - 0.04166666666666666*dpp));

          EB(i1, i2, em::bx1) -= coeff * curlEr;
          EB(i1, i2, em::bx2) -= coeff * curlEt;
          EB(i1, i2, em::bx3) -= coeff * curlEp;

        } else {

        EB(i1, i2, em::bx1) += coeff * inv_sqrt_detH_0pH *
                               (h3_00 * EB(i1, i2, em::ex3) -
                                h3_0p1 * EB(i1, i2 + 1, em::ex3));
        if ((i2 != i2min) || !is_axis_i2min) {
          EB(i1, i2, em::bx2) += coeff * inv_sqrt_detH_pH0 *
                                 (h3_p10 * EB(i1 + 1, i2, em::ex3) -
                                  h3_00 * EB(i1, i2, em::ex3));
        }
        EB(i1, i2, em::bx3) += coeff * inv_sqrt_detH_pHpH *
                               (h1_pHp1 * EB(i1, i2 + 1, em::ex1) -
                                h1_pH0 * EB(i1, i2, em::ex1) +
                                h2_0pH * EB(i1, i2, em::ex2) -
                                h2_p1pH * EB(i1 + 1, i2, em::ex2));

        }
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
} // namespace kernel::sr

#endif // KERNELS_FARADAY_SR_HPP
