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

  template <bool STAGGERED, unsigned short O>
  Inline void order(const int& i, const real_t& di, int& i_min, real_t* S) {
    if constexpr (O == 2u) {
      if constexpr (not STAGGERED) { // compute at i positions
        if (di < HALF) {
          i_min = i - 1;
          S[0]  = HALF * SQR(HALF - di);
          S[1]  = THREE_FOURTHS - SQR(di);
          S[2]  = ONE - S[0] - S[1];
        } else {
          i_min = i;
          S[0]  = HALF * SQR(static_cast<real_t>(1.5) - di);
          S[2]  = HALF * SQR(di - HALF);
          S[1]  = ONE - S[0] - S[2];
        }
      } else { // compute at i + 1/2 positions
        i_min = i - 1;
        S[1]  = THREE_FOURTHS - SQR(di - HALF);
        S[2]  = HALF * SQR(di);
        S[0]  = ONE - S[1] - S[2];
      } // staggered
    } else if constexpr (O == 3u) {
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 2;
        S[0]  = HALF * THIRD * CUBE(ONE - di);
        S[3]  = HALF * THIRD * CUBE(di);
        S[1]  = HALF * THIRD * (FOUR - SIX * SQR(di) + THREE * CUBE(di));
        S[2]  = ONE - S[0] - S[1] - S[3];
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 2;
          S[0]  = HALF * THIRD * CUBE(HALF - di);
          S[3]  = HALF * THIRD * CUBE(HALF + di);
          S[1]  = HALF * THIRD *
                 (FOUR - SIX * SQR(HALF - di) + THREE * CUBE(HALF - di));
          S[2] = ONE - S[0] - S[1] - S[3];
        } else {
          i_min = i - 1;
          S[0]  = HALF * THIRD * CUBE(HALF + di);
          S[3]  = HALF * THIRD * CUBE(HALF + di);
          S[1]  = HALF * THIRD *
                 (FOUR - SIX * SQR(di - HALF) + THREE * CUBE(di - HALF));
          S[2] = ONE - S[0] - S[1] - S[3];
        }
      } // staggered
    } else if constexpr (O == 4u) {
      //        1/25 * ( 5/2 - |x|)^4                           |x| < 3/2
      // S(x) = 5/8 - |x|^2 + 32/45 * |x|^3 - 98/675 * |x|^4    3/2 ≤ |x| < 5/2
      //        0.0                                             |x| ≥ 5/2
      if constexpr (not STAGGERED) { // compute at i positions
        if (di < HALF) {
          i_min = i - 2;
          S[0]  = ONE / (FIVE * FIVE) * SQR(SQR(HALF - di));
          S[4]  = ONE / (FIVE * FIVE) * SQR(SQR(HALF + di));
          S[1]  = FIVE * INV_8 - SQR(ONE + di) +
                 static_cast<real_t>(32 / 45) * CUBE(ONE + di) -
                 static_cast<real_t>(98 / 675) * SQR(SQR(ONE + di));
          S[2] = FIVE * INV_8 - SQR(di) + static_cast<real_t>(32 / 45) * CUBE(di) -
                 static_cast<real_t>(98 / 675) * SQR(SQR(di));
          S[3] = ONE - S[0] - S[1] - S[2] - S[4];
        } else {
          i_min = i - 1;
          S[0]  = ONE / (FIVE * FIVE) * SQR(SQR(THREE * HALF - di));
          S[4]  = ONE / (FIVE * FIVE) * SQR(SQR(di - HALF));
          S[1] = FIVE * INV_8 - SQR(di) + static_cast<real_t>(32 / 45) * CUBE(di) -
                 static_cast<real_t>(98 / 675) * SQR(SQR(di));
          S[2] = FIVE * INV_8 - SQR(ONE - di) +
                 static_cast<real_t>(32 / 45) * CUBE(ONE - di) -
                 static_cast<real_t>(98 / 675) * SQR(SQR(ONE - di));
          S[3] = ONE - S[0] - S[1] - S[2] - S[4];
        }
      } else { // compute at i + 1/2 positions
        i_min = i - 2;
        S[0]  = ONE / (FIVE * FIVE) * SQR(SQR(ONE - di)); //
        S[4]  = ONE / (FIVE * FIVE) * SQR(SQR(di));       //
        S[1]  = FIVE * INV_8 - SQR(HALF + di) +
               static_cast<real_t>(32 / 45) * CUBE(HALF + di) -
               static_cast<real_t>(98 / 675) * SQR(SQR(HALF + di));
        S[2] = FIVE * INV_8 - SQR(HALF - di) +
               static_cast<real_t>(32 / 45) * CUBE(HALF - di) -
               static_cast<real_t>(98 / 675) * SQR(SQR(HALF - di));
        S[3] = ONE - S[0] - S[1] - S[2] - S[4];
      } // staggered
    } else if constexpr (O == 5u) {
      //        3/5 - |x|^2 + 5/6 * |x|^3 - 19/72 * |x|^4 + 13/432 * |x|^5   |x| < 2
      // S(x) = 1/135 * (3 - |x|)^5                                           2 ≤ |x| < 3
      //        0.0 |x| ≥ 3
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 2;
        S[0]  = static_cast<real_t>(1 / 135) * SQR(CUBE(ONE - di)); //
        S[1]  = static_cast<real_t>(3 / 5) - SQR(ONE + di) +
               static_cast<real_t>(5 / 6) * CUBE(ONE + di) -
               static_cast<real_t>(19 / 72) * SQR(SQR(ONE + di)) +
               static_cast<real_t>(13 / 432) * SQR(CUBE(ONE + di));
        S[2] = static_cast<real_t>(3 / 5) - SQR(di) +
               static_cast<real_t>(5 / 6) * CUBE(di) -
               static_cast<real_t>(19 / 72) * SQR(SQR(di)) +
               static_cast<real_t>(13 / 432) * SQR(CUBE(di));
        S[3] = static_cast<real_t>(3 / 5) - SQR(ONE - di) +
               static_cast<real_t>(5 / 6) * CUBE(ONE - di) -
               static_cast<real_t>(19 / 72) * SQR(SQR(ONE - di)) +
               static_cast<real_t>(13 / 432) * SQR(CUBE(ONE - di));
        S[5] = static_cast<real_t>(1 / 135) * SQR(CUBE(di));
        S[3] = ONE - S[0] - S[1] - S[2] - S[4];
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 3;
          S[0]  = static_cast<real_t>(1 / 135) * SQR(CUBE(HALF - di));
          S[1] = static_cast<real_t>(3 / 5) - SQR(static_cast<real_t>(1.5) + di) +
                 static_cast<real_t>(5 / 6) * CUBE(static_cast<real_t>(1.5) + di) -
                 static_cast<real_t>(19 / 72) *
                   SQR(SQR(static_cast<real_t>(1.5) + di)) +
                 static_cast<real_t>(13 / 432) *
                   SQR(CUBE(static_cast<real_t>(1.5) + di));
          S[2] = static_cast<real_t>(3 / 5) - SQR(HALF + di) +
                 static_cast<real_t>(5 / 6) * CUBE(HALF + di) -
                 static_cast<real_t>(19 / 72) * SQR(SQR(HALF + di)) +
                 static_cast<real_t>(13 / 432) * SQR(CUBE(HALF + di));
          S[3] = static_cast<real_t>(3 / 5) - SQR(HALF - di) +
                 static_cast<real_t>(5 / 6) * CUBE(HALF - di) -
                 static_cast<real_t>(19 / 72) * SQR(SQR(HALF - di)) +
                 static_cast<real_t>(13 / 432) * SQR(CUBE(HALF - di));
          S[5] = static_cast<real_t>(1 / 135) * SQR(CUBE(HALF + di));
          S[3] = ONE - S[0] - S[1] - S[2] - S[4];
        } else {
          i_min = i - 2;
          S[0]  = static_cast<real_t>(1 / 135) *
                 SQR(CUBE(static_cast<real_t>(1.5) - di));
          S[1] = static_cast<real_t>(3 / 5) - SQR(HALF + di) +
                 static_cast<real_t>(5 / 6) * CUBE(HALF + di) -
                 static_cast<real_t>(19 / 72) * SQR(SQR(HALF + di)) +
                 static_cast<real_t>(13 / 432) * SQR(CUBE(HALF + di));
          S[2] = static_cast<real_t>(3 / 5) - SQR(di - HALF) +
                 static_cast<real_t>(5 / 6) * CUBE(di - HALF) -
                 static_cast<real_t>(19 / 72) * SQR(SQR(di - HALF)) +
                 static_cast<real_t>(13 / 432) * SQR(CUBE(di - HALF));
          S[3] = static_cast<real_t>(3 / 5) - SQR(static_cast<real_t>(1.5) - di) +
                 static_cast<real_t>(5 / 6) * CUBE(static_cast<real_t>(1.5) - di) -
                 static_cast<real_t>(19 / 72) *
                   SQR(SQR(static_cast<real_t>(1.5) - di)) +
                 static_cast<real_t>(13 / 432) *
                   SQR(CUBE(static_cast<real_t>(1.5) - di));
          S[5] = static_cast<real_t>(1 / 135) * SQR(CUBE(di - HALF));
          S[3] = ONE - S[0] - S[1] - S[2] - S[4];
        }
      } // staggered
    }
  }

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

    real_t iS_[3], fS_[3];

    order<false, 2u>(i_init, di_init, i_init_min, iS_);
    order<false, 2u>(i_fin, di_fin, i_fin_min, fS_);

    if (i_init_min < i_fin_min) {
      i_min = i_init_min;
      iS_0  = iS_[0];
      iS_1  = iS_[1];
      iS_2  = iS_[2];
      iS_3  = ZERO;

      fS_0 = ZERO;
      fS_1 = iS_[0];
      fS_2 = iS_[1];
      fS_3 = iS_[2];
    } else if (i_init_min > i_fin_min) {
      i_min = i_fin_min;
      iS_0  = ZERO;
      iS_1  = iS_[0];
      iS_2  = iS_[1];
      iS_3  = iS_[2];

      fS_0 = iS_[0];
      fS_1 = iS_[1];
      fS_2 = iS_[2];
      fS_3 = ZERO;
    } else {
      i_min = i_init_min;
      iS_0  = iS_[0];
      iS_1  = iS_[1];
      iS_2  = iS_[2];
      iS_3  = ZERO;

      fS_0 = iS_[0];
      fS_1 = iS_[1];
      fS_2 = iS_[2];
      fS_3 = ZERO;
    }
  }

  template <int O>
  Inline void for_deposit(const int&    i_init,
                          const real_t& di_init,
                          const int&    i_fin,
                          const real_t& di_fin,
                          int&          i_min,
                          real_t*       iS,
                          real_t*       fS) {

    int i_init_min, i_fin_min;

    real_t iS_[O + 1], fS_[O + 1];

    order<false, O>(i_init, di_init, i_init_min, iS_);
    order<false, O>(i_fin, di_fin, i_fin_min, fS_);

    if (i_init_min < i_fin_min) {
      i_min = i_init_min;

#pragma unroll
      for (int j = 0; j < O; j++) {
        iS[j] = iS_[j];
      }
      iS[O + 1] = ZERO;

      fS[0] = ZERO;
#pragma unroll
      for (int j = 0; j < O; j++) {
        fS[j + 1] = fS_[j];
      }

    } else if (i_init_min > i_fin_min) {
      i_min = i_fin_min;

      iS[0] = ZERO;
#pragma unroll
      for (int j = 0; j < O; j++) {
        iS[j + 1] = iS_[j];
      }

#pragma unroll
      for (int j = 0; j < O; j++) {
        fS[j] = fS_[j];
      }
      fS[O + 1] = ZERO;

    } else {
      i_min = i_init_min;

#pragma unroll
      for (int j = 0; j < O; j++) {
        iS[j] = iS_[j];
      }
      iS[O + 1] = ZERO;

#pragma unroll
      for (int j = 0; j < O; j++) {
        fS[j] = fS_[j];
      }
      fS[O + 1] = ZERO;
    }
  }

} // namespace prtl_shape

#endif // KERNELS_PARTICLE_SHAPES_HPP
