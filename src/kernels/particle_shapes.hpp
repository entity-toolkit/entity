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
  Inline void order(const int& i, const real_t& di, int& i_min, real_t S[O + 1]) {
    if constexpr (O == 1u) {
      // S(x) = 1 - |x|      |x| < 1
      //        0.0          |x| ≥ 1
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i;
        S[0]  = ONE - di;
        S[1]  = di;
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 1;
          S[0]  = HALF - di;
          S[1]  = ONE - S[0];
        } else {
          i_min = i;
          S[0]  = static_cast<real_t>(1.5) - di;
          S[1]  = ONE - S[0];
        }
      } // staggered
    } else if constexpr (O == 2u) {
      //        3/4 - |x|^2              |x| < 1/2
      // S(x) = 1/2 * (3/2 - |x|)^2     1/2 ≤ |x| < 3/2
      //        0.0                      |x| ≥ 3/2
      if constexpr (not STAGGERED) { // compute at i positions
        if (di < HALF) {
          i_min = i - 1;
          S[0]  = HALF * SQR(HALF - di);
          S[1]  = THREE_FOURTHS - SQR(di);
          S[2]  = ONE - S[0] - S[1];
        } else {
          i_min = i;
          S[0]  = HALF * SQR(static_cast<real_t>(3.0 / 2.0) - di);
          S[1]  = THREE_FOURTHS - SQR(ONE - di);
          S[2]  = ONE - S[0] - S[1];
        }
      } else { // compute at i + 1/2 positions
        i_min = i - 1;
        S[0] = HALF * SQR(ONE - di);
        S[2] = HALF * SQR(di);
        S[1] = ONE - S[0] - S[2];
      } // staggered
    } else if constexpr (O == 3u) {
      //        1/6 * ( 4 - 6 * |x|^2 + 3 * |x|^3)    |x| < 1
      // S(x) = 1/6 * ( 2 - |x|)^3                    1 ≤ |x| < 2
      //        0.0                                   |x| ≥ 2
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 2;
        S[0]  = static_cast<real_t>(1.0 / 6.0) * CUBE(ONE - di);
        S[1]  = static_cast<real_t>(1.0 / 6.0) *
               (FOUR - SIX * SQR(di) + THREE * CUBE(di));
        S[3]  = static_cast<real_t>(1.0 / 6.0) * CUBE(di);
        S[2] = ONE - S[0] - S[1] - S[3];
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 2;
          S[0]  = static_cast<real_t>(1.0 / 6.0) * CUBE(HALF - di);
          S[1]  = static_cast<real_t>(1.0 / 6.0) *
                 (FOUR - SIX * SQR(HALF + di) + THREE * CUBE(HALF + di));
          S[3]  = static_cast<real_t>(1.0 / 6.0) * CUBE(HALF + di);
          S[2] = ONE - S[0] - S[1] - S[3];
        } else {
          i_min = i - 1;
          S[0]  = static_cast<real_t>(1.0 / 6.0) * CUBE(static_cast<real_t>(1.5) - di);
          S[1]  = static_cast<real_t>(1.0 / 6.0) *
                 (FOUR - SIX * SQR(di - HALF) + THREE * CUBE(di - HALF));
          S[3]  = static_cast<real_t>(1.0 / 6.0) * CUBE(di - HALF);
          S[2] = ONE - S[0] - S[1] - S[3];
        }
      } // staggered
    } else if constexpr (O == 4u) {
      //        5/8 - |x|^2 + 32/45 * |x|^3 - 98/675 * |x|^4    |x| < 3/2
      // S(x) = 1/25 * ( 5/2 - |x|)^4                           3/2 ≤ |x| < 5/2
      //        0.0                                             |x| ≥ 5/2
      if constexpr (not STAGGERED) { // compute at i positions
        if (di < HALF) {
          i_min = i - 2;
          S[0]  = static_cast<real_t>(1.0 / 25.0) * SQR(SQR(HALF - di));
          S[1]  = static_cast<real_t>(5.0 / 8.0) - SQR(ONE + di) +
                 static_cast<real_t>(32.0 / 45.0) * CUBE(ONE + di) -
                 static_cast<real_t>(98.0 / 675.0) * SQR(SQR(ONE + di));
          S[2]  = static_cast<real_t>(5.0 / 8.0) - SQR(di) +
                 static_cast<real_t>(32.0 / 45.0) * CUBE(di) -
                 static_cast<real_t>(98.0 / 675.0) * SQR(SQR(di));
          S[3]  = static_cast<real_t>(5.0 / 8.0) - SQR(ONE - di) +
                 static_cast<real_t>(32.0 / 45.0) * CUBE(ONE - di) -
                 static_cast<real_t>(98.0 / 675.0) * SQR(SQR(ONE - di));
          S[4]  = static_cast<real_t>(1.0 / 25.0) * SQR(SQR(HALF + di));
          S[2] = ONE - S[0] - S[1] - S[3] - S[4];
        } else {
          i_min = i - 1;
          S[0]  = static_cast<real_t>(1.0 / 25.0) * SQR(SQR(static_cast<real_t>(1.5) - di));
          S[1]  = static_cast<real_t>(5.0 / 8.0) - SQR(di) +
                 static_cast<real_t>(32.0 / 45.0) * CUBE(di) -
                 static_cast<real_t>(98.0 / 675.0) * SQR(SQR(di));
          S[3]  = static_cast<real_t>(5.0 / 8.0) - SQR(TWO - di) +
                 static_cast<real_t>(32.0 / 45.0) * CUBE(TWO - di) -
                 static_cast<real_t>(98.0 / 675.0) * SQR(SQR(TWO - di));
          S[4]  = static_cast<real_t>(1.0 / 25.0) * SQR(SQR(di - HALF));
          S[2] = ONE - S[0] - S[1] - S[3] - S[4];
        }
      } else { // compute at i + 1/2 positions
          i_min = i - 2;
          S[0]  = static_cast<real_t>(1.0 / 25.0) * SQR(SQR(ONE - di));
          S[1]  = static_cast<real_t>(5.0 / 8.0) - SQR(HALF + di) +
                 static_cast<real_t>(32.0 / 45.0) * CUBE(HALF + di) -
                 static_cast<real_t>(98.0 / 675.0) * SQR(SQR(HALF + di));
          S[3]  = static_cast<real_t>(5.0 / 8.0) - SQR(TWO - di) +
                 static_cast<real_t>(32.0 / 45.0) * CUBE(TWO - di) -
                 static_cast<real_t>(98.0 / 675.0) * SQR(SQR(TWO - di));
          S[4]  = static_cast<real_t>(1.0 / 25.0) * SQR(SQR(di));
          S[2] = ONE - S[0] - S[1] - S[3] - S[4];
      } // staggered
    } else if constexpr (O == 5u) {
      //        3/5 - |x|^2 + 5/6 * |x|^3 - 19/72 * |x|^4 + 13/432 * |x|^5   |x| < 2
      // S(x) = 1/135 * (3 - |x|)^5                                           2 ≤ |x| < 3
      //        0.0 |x| ≥ 3
      if constexpr (not STAGGERED) { // compute at i positions
        i_min = i - 2;
        S[0]  = static_cast<real_t>(1.0 / 135.0) * SQR(SQR(ONE + di))*(ONE - di);
        S[1]  = static_cast<real_t>(3.0 / 5.0) - SQR(ONE + di) +
               static_cast<real_t>(5.0 / 6.0) * CUBE(ONE + di) -
               static_cast<real_t>(19.0 / 72.0) * SQR(SQR(ONE + di)) +
               static_cast<real_t>(13.0 / 432.0) * SQR(SQR(ONE + di))*(ONE + di);
        S[2]  = static_cast<real_t>(3.0 / 5.0) - SQR(di) +
               static_cast<real_t>(5.0 / 6.0) * CUBE(di) -
               static_cast<real_t>(19.0 / 72.0) * SQR(SQR(di)) +
               static_cast<real_t>(13.0 / 432.0) * SQR(SQR(di)) * di;
        S[3]  = static_cast<real_t>(3.0 / 5.0) - SQR(ONE - di) +
               static_cast<real_t>(5.0 / 6.0) * CUBE(ONE - di) -
               static_cast<real_t>(19.0 / 72.0) * SQR(SQR(ONE - di)) +
               static_cast<real_t>(13.0 / 432.0) * SQR(SQR(ONE - di))*(ONE - di);
        S[4]  = static_cast<real_t>(3.0 / 5.0) - SQR(TWO - di) +
               static_cast<real_t>(5.0 / 6.0) * CUBE(TWO - di) -
               static_cast<real_t>(19.0 / 72.0) * SQR(SQR(TWO - di)) +
               static_cast<real_t>(13.0 / 432.0) * SQR(SQR(TWO - di))*(TWO - di);
        S[5]  = static_cast<real_t>(1.0 / 135.0) * SQR(SQR(di))*di;
      } else { // compute at i + 1/2 positions
        if (di < HALF) {
          i_min = i - 3;
          S[0]  = static_cast<real_t>(1.0 / 135.0) * SQR(CUBE(HALF - di));
          S[1]  = static_cast<real_t>(3.0 / 5.0) - SQR(static_cast<real_t>(1.5) + di) +
                static_cast<real_t>(5.0 / 6.0) * CUBE(static_cast<real_t>(1.5) + di) -
                static_cast<real_t>(19.0 / 72.0) * SQR(SQR(static_cast<real_t>(1.5) + di)) +
                static_cast<real_t>(13.0 / 432.0) * SQR(CUBE(static_cast<real_t>(1.5) + di));
          S[2]  = static_cast<real_t>(3.0 / 5.0) - SQR(HALF + di) +
                static_cast<real_t>(5.0 / 6.0) * CUBE(HALF + di) -
                static_cast<real_t>(19.0 / 72.0) * SQR(SQR(HALF + di)) +
                static_cast<real_t>(13.0 / 432.0) * SQR(CUBE(HALF + di));
          S[4]  = static_cast<real_t>(3.0 / 5.0) - SQR(static_cast<real_t>(1.5) - di) +
                static_cast<real_t>(5.0 / 6.0) * CUBE(static_cast<real_t>(1.5) - di) -
                static_cast<real_t>(19.0 / 72.0) * SQR(SQR(static_cast<real_t>(1.5) - di)) +
                static_cast<real_t>(13.0 / 432.0) * SQR(CUBE(static_cast<real_t>(1.5) - di));
          S[5]  = static_cast<real_t>(1.0 / 135.0) * SQR(CUBE(HALF + di));
          S[3] = ONE - S[0] - S[1] - S[2] - S[4] - S[5];
        } else {
          i_min = i - 2;
          S[0]  = static_cast<real_t>(1.0 / 135.0) * SQR(CUBE(static_cast<real_t>(1.5) - di));
          S[1]  = static_cast<real_t>(3.0 / 5.0) - SQR(HALF + di) +
                static_cast<real_t>(5.0 / 6.0) * CUBE(HALF + di) -
                static_cast<real_t>(19.0 / 72.0) * SQR(SQR(HALF + di)) +
                static_cast<real_t>(13.0 / 432.0) * SQR(CUBE(HALF + di));
          S[2]  = static_cast<real_t>(3.0 / 5.0) - SQR(di - HALF) +
                static_cast<real_t>(5.0 / 6.0) * CUBE(di - HALF) -
                static_cast<real_t>(19.0 / 72.0) * SQR(SQR(di - HALF)) +
                static_cast<real_t>(13.0 / 432.0) * SQR(CUBE(di - HALF));
          S[4]  = static_cast<real_t>(3.0 / 5.0) - SQR(static_cast<real_t>(2.5) - di) +
                static_cast<real_t>(5.0 / 6.0) * CUBE(static_cast<real_t>(2.5) - di) -
                static_cast<real_t>(19.0 / 72.0) * SQR(SQR(static_cast<real_t>(2.5) - di)) +
                static_cast<real_t>(13.0 / 432.0) * SQR(CUBE(static_cast<real_t>(2.5) - di));
          S[5]  = static_cast<real_t>(1.0 / 135.0) * SQR(CUBE(di - HALF));
          S[3] = ONE - S[0] - S[1] - S[2] - S[4] - S[5];
        }
      } // staggered
    }
  }


  template <int O>
  Inline void for_deposit(const int&    i_init,
                          const real_t& di_init,
                          const int&    i_fin,
                          const real_t& di_fin,
                          int&          i_min,
                          int&          i_max,
                          real_t        iS[O + 2],
                          real_t        fS[O + 2]) {

    /*
    The N-th order shape function per particle is a N+2 element array
    where the shape function contributes to only N+1 elements.
    We need to find which indices are contributing to the shape function
    For this we first compute the indices of the particle position

    Let * be the particle position at the current timestep
    Let x be the particle position at the previous timestep


              0      1    (...)    N     N+1
          ___________________________________
          |  x*  |  x*  |  ... |  x*  |      |   // i_init_min = i_fin_min
          |______|______|______|______|______|
          |  x   |  x*  |  ... |  x*  |  *   |   // i_init_min < i_fin_min
          |______|______|______|______|______|
          |  *   |  x*  |  ... |  x*  |  x   |   // i_init_min > i_fin_min
          |______|______|______|______|______|
    */

    int i_init_min, i_fin_min;

    real_t iS_[O + 1], fS_[O + 1];

    order<false, O>(i_init, di_init, i_init_min, iS_);
    order<false, O>(i_fin, di_fin, i_fin_min, fS_);

    if (i_init_min < i_fin_min) {
      i_min = i_init_min;
      i_max = i_min + O + 1;

#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        iS[j] = iS_[j];
      }
      iS[O + 1] = ZERO;

      fS[0] = ZERO;
#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        fS[j + 1] = fS_[j];
      }

    } else if (i_init_min > i_fin_min) {
      i_min = i_fin_min;
      i_max = i_min + O + 1;

      iS[0] = ZERO;
#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        iS[j + 1] = iS_[j];
      }

#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        fS[j] = fS_[j];
      }
      fS[O + 1] = ZERO;

    } else {
      i_min = i_init_min;
      i_max = i_min + O;

#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        iS[j] = iS_[j];
      }
      iS[O + 1] = ZERO;

#pragma unroll
      for (int j = 0; j < O + 1; j++) {
        fS[j] = fS_[j];
      }
      fS[O + 1] = ZERO;
    }
  }
} // namespace prtl_shape

#endif // KERNELS_PARTICLE_SHAPES_HPP
