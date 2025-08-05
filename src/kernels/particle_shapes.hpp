/**
 * @file kernels/particle_shapes.hpp
 * @brief Functions to compute particle shapes at specific locations on the grid.
 * @implements:
 *   - order_2<> -> void
 * @namespaces:
 *   - prtl_shape::
 */

#ifndef KERNELS_PARTICLE_SHAPES_HPP
#define KERNELS_PARTICLE_SHAPES_HPP

#include "global.h"

#include "utils/error.h"
#include "utils/numeric.h"

namespace prtl_shape {

  template <bool STAGGERED>
  Inline void order_2nd(const int&    i,
                        const real_t& di,
                        int&          i_min,
                        real_t&       S0,
                        real_t&       S1,
                        real_t&       S2) {
    if constexpr (not STAGGERED) { // compute at i positions
      if (di < HALF) {
        i_min = i - 1;
        S0    = HALF * SQR(HALF - di);
        S1    = THREE_FOURTHS - SQR(di);
        S2    = ONE - S0 - S1;
      } else {
        i_min = i;
        S0    = HALF * SQR(static_cast<real_t>(1.5) - di);
        S2    = HALF * SQR(di - HALF);
        S1    = ONE - S0 - S2;
      }
    } else { // compute at i + 1/2 positions
      i_min = i - 1;
      S1    = HALF + di - SQR(di);
      S2    = HALF * SQR(di);
      S0    = ONE - S1 - S2;
    }
  }

  template <bool STAGGERED>
  Inline void for_deposit_2nd(const int&    i_init,
                              const real_t& di_init,
                              const int&    i_fin,
                              const real_t& di_fin,
                              int&          i_min,
                              real_t&       iS_0,
                              real_t&       iS_1,
                              real_t&       iS_2,
                              real_t&       iS_3,
                              real_t&       fS_0,
                              real_t&       fS_1,
                              real_t&       fS_2,
                              real_t&       fS_3) {
    int i_init_min, i_fin_min;

    real_t iS_0_, iS_1_, iS_2_;
    real_t fS_0_, fS_1_, fS_2_;

    order_2nd<STAGGERED>(i_init, di_init, i_init_min, iS_0_, iS_1_, iS_2_);
    order_2nd<STAGGERED>(i_fin, di_fin, i_fin_min, fS_0_, fS_1_, fS_2_);

    if (i_init_min < i_fin_min) {
      i_min = i_init_min;
      iS_0  = iS_0_;
      iS_1  = iS_1_;
      iS_2  = iS_2_;
      iS_3  = ZERO;

      fS_0 = ZERO;
      fS_1 = iS_0_;
      fS_2 = iS_1_;
      fS_3 = iS_2_;
    } else if (i_init_min > i_fin_min) {
      i_min = i_fin_min;
      iS_0  = ZERO;
      iS_1  = iS_0_;
      iS_2  = iS_1_;
      iS_3  = iS_2_;

      fS_0 = iS_0_;
      fS_1 = iS_1_;
      fS_2 = iS_2_;
      fS_3 = ZERO;
    } else {
      i_min = i_init_min;
      iS_0  = iS_0_;
      iS_1  = iS_1_;
      iS_2  = iS_2_;
      iS_3  = ZERO;

      fS_0 = iS_0_;
      fS_1 = iS_1_;
      fS_2 = iS_2_;
      fS_3 = ZERO;
    }
  }

} // namespace prtl_shape

#endif // KERNELS_PARTICLE_SHAPES_HPP
