/**
 * @file kernels/current_deposit.hpp
 * @brief Covariant algorithms for the current deposition
 * @implements
 *   - kernel::DepositCurrents_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_CURRENTS_DEPOSIT_HPP
#define KERNELS_CURRENTS_DEPOSIT_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

namespace kernel {
  using namespace ntt;

  /**
   * @brief Algorithm for the current deposition
   */
  template <SimEngine::type S, class M, unsigned short O = 1u>
  class DepositCurrents_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    scatter_ndfield_t<D, 3>  J;
    const array_t<int*>      i1, i2, i3;
    const array_t<int*>      i1_prev, i2_prev, i3_prev;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const M                  metric;
    const real_t             charge, inv_dt;

    Inline void shape_function_2nd(real_t&        S0_0,
                                   real_t&        S0_1,
                                   real_t&        S0_2,
                                   real_t&        S0_3,
                                   real_t&        S1_0,
                                   real_t&        S1_1,
                                   real_t&        S1_2,
                                   real_t&        S1_3,
                                   ncells_t&      i_min,
                                   bool&          update_i2,
                                   const index_t& i,
                                   const real_t&  di,
                                   const index_t& i_prev,
                                   const real_t&  di_prev) const {
      /*
        Shape function per particle is a 4 element array.
        We need to find which indices are contributing to the shape function
        For this we first compute the indices of the particle position

        Let * be the particle position at the current timestep
        Let x be the particle position at the previous timestep


          (-1)    0      1      2      3
        ___________________________________
        |      |  x*  |  x*  |  x*  |      |   // shift_i = 0
        |______|______|______|______|______|
        |      |  x   |  x*  |  x*  |  *   |   // shift_i = 1
        |______|______|______|______|______|
        |  *   |  x*  |  x*  |  x   |      |   // shift_i = -1
        |______|______|______|______|______|
      */

      // find shift in indices
      const int di_less_half = static_cast<int>(di < static_cast<prtldx_t>(0.5));
      const int di_prev_less_half = static_cast<int>(
        di_prev < static_cast<prtldx_t>(0.5));

      const int shift_i = (i - di_less_half) - (i_prev - di_prev_less_half);

      // find the minimum index of the shape function
      i_min = Kokkos::min((i - di_less_half), (i_prev - di_prev_less_half));

      // center index of the shape function
      const auto di_center_prev = static_cast<real_t>(1 - di_prev_less_half) -
                                  di_prev;
      const auto di_center = static_cast<real_t>(1 - di_less_half) - di;

      // find indices and define shape function
      if (shift_i == 1) {
        /*
            (-1)    0      1      2      3
          ___________________________________
          |      |  x   |  x*  |  x*  |  *   |   // shift_i = 1
          |______|______|______|______|______|
        */
        update_i2 = true;

        S0_0 = HALF * SQR(HALF + di_center_prev);
        S0_1 = THREE_FOURTHS - SQR(di_center_prev);
        S0_2 = HALF * SQR(HALF - di_center_prev);
        S0_3 = ZERO;

        S1_0 = ZERO;
        S1_1 = HALF * SQR(HALF + di_center);
        S1_2 = THREE_FOURTHS - SQR(di_center);
        S1_3 = HALF * SQR(HALF - di_center);
      } else if (shift_i == -1) {
        /*
            (-1)    0      1      2      3
          ___________________________________
          |  *   |  x*  |  x*  |  x   |      |   // shift_i = -1
          |______|______|______|______|______|
        */
        update_i2 = true;

        S0_0 = ZERO;
        S0_1 = HALF * SQR(HALF + di_center_prev);
        S0_2 = THREE_FOURTHS - SQR(di_center_prev);
        S0_3 = HALF * SQR(HALF - di_center_prev);

        S1_0 = HALF * SQR(HALF + di_center);
        S1_1 = THREE_FOURTHS - SQR(di_center);
        S1_2 = HALF * SQR(HALF - di_center);
        S1_3 = ZERO;

      } else if (shift_i == 0) {
        /*
            (-1)    0      1      2      3
          ___________________________________
          |      |  x*  |  x*  |  x*  |      |   // shift_i = 0
          |______|______|______|______|______|
        */
        update_i2 = false;

        S0_0 = HALF * SQR(HALF + di_center_prev);
        S0_1 = THREE_FOURTHS - SQR(di_center_prev);
        S0_2 = HALF * SQR(HALF - di_center_prev);
        S0_3 = ZERO;

        S1_0 = HALF * SQR(HALF + di_center);
        S1_1 = THREE_FOURTHS - SQR(di_center);
        S1_2 = HALF * SQR(HALF - di_center);
        S1_3 = ZERO;
      } else {
        raise::KernelError(HERE, "Invalid shift in indices");
      }

      // account for ghost cells here to shorten J update expression
      i_min += N_GHOSTS;
    }

    Inline void shape_function_3rd(real_t&        S0_0,
                                   real_t&        S0_1,
                                   real_t&        S0_2,
                                   real_t&        S0_3,
                                   real_t&        S0_4,
                                   real_t&        S1_0,
                                   real_t&        S1_1,
                                   real_t&        S1_2,
                                   real_t&        S1_3,
                                   real_t&        S1_4,
                                   ncells_t&      i_min,
                                   bool&          update_i3,
                                   const index_t& i,
                                   const real_t&  di,
                                   const index_t& i_prev,
                                   const real_t&  di_prev) const {
      /*
        Shape function per particle is a 4 element array.
        We need to find which indices are contributing to the shape function
        For this we first compute the indices of the particle position

        Let * be the particle position at the current timestep
        Let x be the particle position at the previous timestep


          (-1)    0      1      2      3       4
        __________________________________________
        |      |  x*  |  x*  |  x*  |  x*  |      |      // shift_i = 0
        |______|______|______|______|______|______|
        |      |  x   |  x*  |  x*  |  x*  |   *  |      // shift_i = 1
        |______|______|______|______|______|______|
        |  *   |  x*  |  x*  |  x*  |  x   |      |      // shift_i = -1
        |______|______|______|______|______|______|
      */

      // find shift in indices
      const int di_less_half = static_cast<int>(di < static_cast<prtldx_t>(0.5));
      const int di_prev_less_half = static_cast<int>(
        di_prev < static_cast<prtldx_t>(0.5));

      const int shift_i = (i - di_less_half) - (i_prev - di_prev_less_half);

      // find the minimum index of the shape function
      i_min = Kokkos::min((i - di_less_half), (i_prev - di_prev_less_half));

      // center index of the shape function
      const auto di_center_prev = static_cast<real_t>(1 - di_prev_less_half) -
                                  di_prev;
      const auto di_center_prev2 = SQR(di_center_prev);
      const auto di_center_prev3 = di_center_prev2 * di_center_prev;

      const auto di_center  = static_cast<real_t>(1 - di_less_half) - di;
      const auto di_center2 = SQR(di_center);
      const auto di_center3 = di_center2 * di_center;

      // find indices and define shape function
      if (shift_i == 1) {
        /*
            (-1)    0      1      2      3      4
          __________________________________________
          |      |  x   |  x*  |  x*  |  x*  |   *  |   // shift_i = 1
          |______|______|______|______|______|______|
        */
        update_i3 = true;

        S0_0 = static_cast<real_t>(1 / 6) * (ONE - di_center_prev3) -
               HALF * (di_center_prev - di_center_prev2);
        S0_1 = static_cast<real_t>(2 / 3) - di_center_prev2 + HALF * di_center_prev3;
        S0_2 = static_cast<real_t>(1 / 6) +
               HALF * (di_center_prev + di_center_prev2 - di_center_prev3);
        S0_3 = static_cast<real_t>(1 / 6) * di_center_prev3;
        S0_4 = ZERO;

        S1_0 = ZERO;
        S1_1 = static_cast<real_t>(1 / 6) * (ONE - di_center3) -
               HALF * (di_center - di_center2);
        S1_2 = static_cast<real_t>(2 / 3) - di_center2 + HALF * di_center3;
        S1_3 = static_cast<real_t>(1 / 6) +
               HALF * (di_center + di_center2 - di_center3);
        S1_4 = static_cast<real_t>(1 / 6) * di_center3;
      } else if (shift_i == -1) {
        /*
            (-1)    0      1      2      3      4
          _________________________________________
          |  *   |  x*  |  x*  |  x*  |  x   |     |   // shift_i = -1
          |______|______|______|______|______|_____|
        */
        update_i3 = true;

        S0_0 = ZERO;
        S0_1 = static_cast<real_t>(1 / 6) * (ONE - di_center_prev3) -
               HALF * (di_center_prev - di_center_prev2);
        S0_2 = static_cast<real_t>(2 / 3) - di_center_prev2 + HALF * di_center_prev3;
        S0_3 = static_cast<real_t>(1 / 6) +
               HALF * (di_center_prev + di_center_prev2 - di_center_prev3);
        S0_4 = static_cast<real_t>(1 / 6) * di_center_prev3;

        S1_0 = static_cast<real_t>(1 / 6) * (ONE - di_center3) -
               HALF * (di_center - di_center2);
        S1_1 = static_cast<real_t>(2 / 3) - di_center2 + HALF * di_center3;
        S1_2 = static_cast<real_t>(1 / 6) +
               HALF * (di_center + di_center2 - di_center3);
        S1_3 = static_cast<real_t>(1 / 6) * di_center3;
        S1_4 = ZERO;

      } else if (shift_i == 0) {
        /*
            (-1)    0      1      2      3      4
          __________________________________________
          |      |  x*  |  x*  |  x*  |  x*  |      |  // shift_i = 0
          |______|______|______|______|______|______|
        */
        update_i3 = false;

        S0_0 = static_cast<real_t>(1 / 6) * (ONE - di_center_prev3) -
               HALF * (di_center_prev - di_center_prev2);
        S0_1 = static_cast<real_t>(2 / 3) - di_center_prev2 + HALF * di_center_prev3;
        S0_2 = static_cast<real_t>(1 / 6) +
               HALF * (di_center_prev + di_center_prev2 - di_center_prev3);
        S0_3 = static_cast<real_t>(1 / 6) * di_center_prev3;
        S0_4 = ZERO;

        S1_0 = static_cast<real_t>(1 / 6) * (ONE - di_center3) -
               HALF * (di_center - di_center2);
        S1_1 = static_cast<real_t>(2 / 3) - di_center2 + HALF * di_center3;
        S1_2 = static_cast<real_t>(1 / 6) +
               HALF * (di_center + di_center2 - di_center3);
        S1_3 = static_cast<real_t>(1 / 6) * di_center3;
        S1_4 = ZERO;
      } else {
        raise::KernelError(HERE, "Invalid shift in indices");
      }

      // account for ghost cells here to shorten J update expression
      i_min += N_GHOSTS;
    }

    Inline void W(real_t* _S, real_t x) const {

      if constexpr (O == 2) {

        _S[0] = HALF * SQR(HALF - x);
        _S[1] = THREE_FOURTHS - SQR(x);
        _S[2] = HALF * SQR(HALF + x);

      } else if constexpr (O == 3) {

        const auto x2 = x * x;
        const auto x3 = x2 * x;

        _S[0] = static_cast<real_t>(1 / 6) * (ONE - x3) - HALF * SQR(x - x2);
        _S[1] = static_cast<real_t>(2 / 3) - x2 + HALF * x3;
        _S[2] = static_cast<real_t>(1 / 6) + HALF * (x + x2 + x3);
        _S[3] = static_cast<real_t>(1 / 6) * x3;

      } else if constexpr (O == 4) {

        const auto x2 = x * x;
        const auto x3 = x2 * x;
        const auto x4 = x2 * x2;

        _S[0] = static_cast<real_t>(1 / 384) - static_cast<real_t>(1 / 48) * x +
                static_cast<real_t>(1 / 16) * x2 -
                static_cast<real_t>(1 / 12) * x3 +
                static_cast<real_t>(1 / 24) * x4;
        _S[1] = static_cast<real_t>(19 / 96) - static_cast<real_t>(11 / 24) * x +
                static_cast<real_t>(1 / 4) * x2 +
                static_cast<real_t>(1 / 6) * x3 - static_cast<real_t>(1 / 6) * x4;
        _S[2] = static_cast<real_t>(115 / 192) - static_cast<real_t>(5 / 8) * x2 +
                static_cast<real_t>(1 / 4) * x4;
        _S[3] = static_cast<real_t>(19 / 96) + static_cast<real_t>(11 / 24) * x +
                static_cast<real_t>(1 / 4) * x2 -
                static_cast<real_t>(1 / 6) * x3 - static_cast<real_t>(1 / 6) * x4;
        _S[4] = static_cast<real_t>(1 / 384) + static_cast<real_t>(1 / 48) * x +
                static_cast<real_t>(1 / 16) * x2 +
                static_cast<real_t>(1 / 12) * x3 +
                static_cast<real_t>(1 / 24) * x4;

      } else if constexpr (O == 5) {

        const auto x2 = x * x;
        const auto x3 = x2 * x;
        const auto x4 = x2 * x2;
        const auto x5 = x3 * x2;
        const auto x6 = x3 * x3;

        _S[0] = static_cast<real_t>(1.0 / 46080.0) -
                static_cast<real_t>(1.0 / 3840.0) * x +
                static_cast<real_t>(1.0 / 384.0) * x2 -
                static_cast<real_t>(1.0 / 96.0) * x3 +
                static_cast<real_t>(1.0 / 72.0) * x4 -
                static_cast<real_t>(1.0 / 144.0) * x5 +
                static_cast<real_t>(1.0 / 720.0) * x6;

        _S[1] = static_cast<real_t>(13.0 / 9216.0) -
                static_cast<real_t>(11.0 / 768.0) * x +
                static_cast<real_t>(1.0 / 48.0) * x2 +
                static_cast<real_t>(5.0 / 72.0) * x3 -
                static_cast<real_t>(1.0 / 8.0) * x4 +
                static_cast<real_t>(5.0 / 144.0) * x5 -
                static_cast<real_t>(1.0 / 144.0) * x6;

        _S[2] = static_cast<real_t>(115.0 / 768.0) -
                static_cast<real_t>(5.0 / 24.0) * x2 +
                static_cast<real_t>(1.0 / 8.0) * x4 -
                static_cast<real_t>(1.0 / 72.0) * x6;

        _S[3] = static_cast<real_t>(115.0 / 768.0) -
                static_cast<real_t>(5.0 / 24.0) * x2 +
                static_cast<real_t>(1.0 / 8.0) * x4 -
                static_cast<real_t>(1.0 / 72.0) * x6;

        _S[4] = static_cast<real_t>(13.0 / 9216.0) +
                static_cast<real_t>(11.0 / 768.0) * x +
                static_cast<real_t>(1.0 / 48.0) * x2 -
                static_cast<real_t>(5.0 / 72.0) * x3 -
                static_cast<real_t>(1.0 / 8.0) * x4 -
                static_cast<real_t>(5.0 / 144.0) * x5 -
                static_cast<real_t>(1.0 / 144.0) * x6;

        _S[5] = static_cast<real_t>(1.0 / 46080.0) +
                static_cast<real_t>(1.0 / 3840.0) * x +
                static_cast<real_t>(1.0 / 384.0) * x2 +
                static_cast<real_t>(1.0 / 96.0) * x3 +
                static_cast<real_t>(1.0 / 72.0) * x4 +
                static_cast<real_t>(1.0 / 144.0) * x5 +
                static_cast<real_t>(1.0 / 720.0) * x6;

      } else if constexpr (O == 6) {

        const auto x2 = x * x;
        const auto x3 = x2 * x;
        const auto x4 = x2 * x2;
        const auto x5 = x3 * x2;
        const auto x6 = x3 * x3;

        _S[0] = static_cast<real_t>(1.0 / 40320.0) -
                static_cast<real_t>(1.0 / 4480.0) * x +
                static_cast<real_t>(1.0 / 640.0) * x2 -
                static_cast<real_t>(1.0 / 192.0) * x3 +
                static_cast<real_t>(1.0 / 144.0) * x4 -
                static_cast<real_t>(1.0 / 288.0) * x5 +
                static_cast<real_t>(1.0 / 1440.0) * x6;

        _S[1] = static_cast<real_t>(1.0 / 1344.0) -
                static_cast<real_t>(1.0 / 160.0) * x +
                static_cast<real_t>(5.0 / 192.0) * x2 -
                static_cast<real_t>(1.0 / 48.0) * x3 -
                static_cast<real_t>(1.0 / 48.0) * x4 +
                static_cast<real_t>(5.0 / 288.0) * x5 -
                static_cast<real_t>(1.0 / 288.0) * x6;

        _S[2] = static_cast<real_t>(17.0 / 336.0) -
                static_cast<real_t>(5.0 / 48.0) * x2 +
                static_cast<real_t>(1.0 / 12.0) * x4 -
                static_cast<real_t>(1.0 / 144.0) * x6;

        _S[3] = static_cast<real_t>(151.0 / 252.0) -
                static_cast<real_t>(35.0 / 48.0) * x2 +
                static_cast<real_t>(5.0 / 12.0) * x4 -
                static_cast<real_t>(1.0 / 36.0) * x6;

        _S[4] = static_cast<real_t>(17.0 / 336.0) -
                static_cast<real_t>(5.0 / 48.0) * x2 +
                static_cast<real_t>(1.0 / 12.0) * x4 -
                static_cast<real_t>(1.0 / 144.0) * x6;

        _S[5] = static_cast<real_t>(1.0 / 1344.0) +
                static_cast<real_t>(1.0 / 160.0) * x +
                static_cast<real_t>(5.0 / 192.0) * x2 +
                static_cast<real_t>(1.0 / 48.0) * x3 -
                static_cast<real_t>(1.0 / 48.0) * x4 -
                static_cast<real_t>(5.0 / 288.0) * x5 -
                static_cast<real_t>(1.0 / 288.0) * x6;

        _S[6] = static_cast<real_t>(1.0 / 40320.0) +
                static_cast<real_t>(1.0 / 4480.0) * x +
                static_cast<real_t>(1.0 / 640.0) * x2 +
                static_cast<real_t>(1.0 / 192.0) * x3 +
                static_cast<real_t>(1.0 / 144.0) * x4 +
                static_cast<real_t>(1.0 / 288.0) * x5 +
                static_cast<real_t>(1.0 / 1440.0) * x6;

      } else if constexpr (O == 7) {

        const auto x2 = x * x;
        const auto x3 = x2 * x;
        const auto x4 = x2 * x2;
        const auto x5 = x3 * x2;
        const auto x6 = x3 * x3;
        const auto x7 = x4 * x3;

        _S[0] = static_cast<real_t>(1.0 / 645120.0) -
                static_cast<real_t>(1.0 / 64512.0) * x +
                static_cast<real_t>(1.0 / 9216.0) * x2 -
                static_cast<real_t>(1.0 / 3072.0) * x3 +
                static_cast<real_t>(1.0 / 2304.0) * x4 -
                static_cast<real_t>(1.0 / 4608.0) * x5 +
                static_cast<real_t>(1.0 / 23040.0) * x6 -
                static_cast<real_t>(1.0 / 161280.0) * x7;

        _S[1] = static_cast<real_t>(1.0 / 9216.0) -
                static_cast<real_t>(5.0 / 4608.0) * x +
                static_cast<real_t>(35.0 / 9216.0) * x2 -
                static_cast<real_t>(7.0 / 768.0) * x3 -
                static_cast<real_t>(7.0 / 1152.0) * x4 +
                static_cast<real_t>(35.0 / 4608.0) * x5 -
                static_cast<real_t>(5.0 / 4608.0) * x6 +
                static_cast<real_t>(1.0 / 9216.0) * x7;

        _S[2] = static_cast<real_t>(25.0 / 1536.0) -
                static_cast<real_t>(35.0 / 768.0) * x2 +
                static_cast<real_t>(7.0 / 192.0) * x4 -
                static_cast<real_t>(1.0 / 96.0) * x6;

        _S[3] = static_cast<real_t>(245.0 / 384.0) -
                static_cast<real_t>(245.0 / 192.0) * x2 +
                static_cast<real_t>(49.0 / 48.0) * x4 -
                static_cast<real_t>(7.0 / 72.0) * x6;

        _S[4] = _S[3]; // symmetry

        _S[5] = _S[2]; // symmetry

        _S[6] = static_cast<real_t>(1 / 9216) + static_cast<real_t>(5 / 4608) * x +
                static_cast<real_t>(35 / 9216) * x2 +
                static_cast<real_t>(7 / 768) * x3 -
                static_cast<real_t>(7 / 1152) * x4 -
                static_cast<real_t>(35 / 4608) * x5 -
                static_cast<real_t>(5 / 4608) * x6 -
                static_cast<real_t>(1 / 9216) * x7;

        _S[7] = static_cast<real_t>(1 / 645120) +
                static_cast<real_t>(1 / 64512) * x +
                static_cast<real_t>(1 / 9216) * x2 +
                static_cast<real_t>(1 / 3072) * x3 +
                static_cast<real_t>(1 / 2304) * x4 +
                static_cast<real_t>(1 / 4608) * x5 +
                static_cast<real_t>(1 / 23040) * x6 +
                static_cast<real_t>(1 / 161280) * x7;

      } else if constexpr (O == 8) {

        const auto x2 = x * x;
        const auto x3 = x2 * x;
        const auto x4 = x2 * x2;
        const auto x5 = x3 * x2;
        const auto x6 = x3 * x3;
        const auto x7 = x4 * x3;
        const auto x8 = x4 * x4;

        _S[0] = static_cast<real_t>(1.0 / 10321920.0) -
                static_cast<real_t>(1.0 / 1146880.0) * x +
                static_cast<real_t>(1.0 / 161280.0) * x2 -
                static_cast<real_t>(1.0 / 53760.0) * x3 +
                static_cast<real_t>(1.0 / 43008.0) * x4 -
                static_cast<real_t>(1.0 / 96768.0) * x5 +
                static_cast<real_t>(1.0 / 645120.0) * x6 -
                static_cast<real_t>(1.0 / 1032192.0) * x7 +
                static_cast<real_t>(1.0 / 4134528.0) * x8;

        _S[1] = static_cast<real_t>(1.0 / 129024.0) -
                static_cast<real_t>(1.0 / 14336.0) * x +
                static_cast<real_t>(17.0 / 43008.0) * x2 -
                static_cast<real_t>(17.0 / 21504.0) * x3 +
                static_cast<real_t>(17.0 / 21504.0) * x4 -
                static_cast<real_t>(17.0 / 43008.0) * x5 +
                static_cast<real_t>(1.0 / 14336.0) * x6 -
                static_cast<real_t>(1.0 / 129024.0) * x7 +
                static_cast<real_t>(1.0 / 1032192.0) * x8;

        _S[2] = static_cast<real_t>(361.0 / 64512.0) -
                static_cast<real_t>(153.0 / 14336.0) * x2 +
                static_cast<real_t>(51.0 / 14336.0) * x4 -
                static_cast<real_t>(17.0 / 43008.0) * x6 +
                static_cast<real_t>(1.0 / 1032192.0) * x8;

        _S[3] = static_cast<real_t>(3061.0 / 16128.0) -
                static_cast<real_t>(170.0 / 1792.0) * x2 +
                static_cast<real_t>(34.0 / 1536.0) * x4 -
                static_cast<real_t>(17.0 / 16128.0) * x6;

        _S[4] = static_cast<real_t>(257135.0 / 32256.0) -
                static_cast<real_t>(1785.0 / 896.0) * x2 +
                static_cast<real_t>(255.0 / 256.0) * x4 -
                static_cast<real_t>(85.0 / 1152.0) * x6;

        _S[5] = _S[3]; // symmetry

        _S[6] = _S[2]; // symmetry

        _S[7] = static_cast<real_t>(1 / 129024) +
                static_cast<real_t>(1 / 14336) * x +
                static_cast<real_t>(17 / 43008) * x2 +
                static_cast<real_t>(17 / 21504) * x3 +
                static_cast<real_t>(17 / 21504) * x4 +
                static_cast<real_t>(17 / 43008) * x5 +
                static_cast<real_t>(1 / 14336) * x6 +
                static_cast<real_t>(1 / 129024) * x7 +
                static_cast<real_t>(1 / 1032192) * x8;

        _S[8] = static_cast<real_t>(1 / 10321920) +
                static_cast<real_t>(1 / 1146880) * x +
                static_cast<real_t>(1 / 161280) * x2 +
                static_cast<real_t>(1 / 53760) * x3 +
                static_cast<real_t>(1 / 43008) * x4 +
                static_cast<real_t>(1 / 96768) * x5 +
                static_cast<real_t>(1 / 645120) * x6 +
                static_cast<real_t>(1 / 1032192) * x7 +
                static_cast<real_t>(1 / 4134528) * x8;

      } else {
        raise::KernelError(HERE, "Invalid order of shape function!");
      }
    }

    Inline void shape_function_Nth(real_t*        S0,
                                   real_t*        S1,
                                   ncells_t&      i_min,
                                   const index_t& i,
                                   const real_t&  di,
                                   const index_t& i_prev,
                                   const real_t&  di_prev) const {
      /*
        Shape function per particle is a O+1 element array.
        We need to find which indices are contributing to the shape function
        For this we first compute the indices of the particle position

        Let * be the particle position at the current timestep
        Let x be the particle position at the previous timestep


          (-1)    0      1      ...    N     N+1
        __________________________________________
        |      |  x*  |  x*  |  //  |  x*  |      |      // shift_i = 0
        |______|______|______|______|______|______|
        |      |  x   |  x*  |  //  |  x*  |   *  |      // shift_i = 1
        |______|______|______|______|______|______|
        |  *   |  x*  |  x*  |  //  |  x   |      |      // shift_i = -1
        |______|______|______|______|______|______|
      */

      // find shift in indices
      // ToDo: fix
      const int di_less_half = static_cast<int>(di < static_cast<prtldx_t>(0.5));
      const int di_prev_less_half = static_cast<int>(
        di_prev < static_cast<prtldx_t>(0.5));

      const int shift_i = (i - di_less_half) - (i_prev - di_prev_less_half);

      // find the minimum index of the shape function -> ToDo!
      i_min = Kokkos::min((i - di_less_half), (i_prev - di_prev_less_half));

      // center index of the shape function -> ToDo!
      const auto di_center_prev = static_cast<real_t>(1 - di_prev_less_half) -
                                  di_prev;
      const auto di_center = static_cast<real_t>(1 - di_less_half) - di;
      // ToDo: end fix

      real_t _S0[O+1], _S1[O+1];
      // apply shape function
      W(_S0, di_center_prev);
      W(_S1, di_center);

      // find indices and define shape function
      if (shift_i == 1) {
        /*
            (-1)    0      1      ...     N     N+1
          __________________________________________
          |      |  x   |  x*  |  //  |  x*  |   *  |   // shift_i = 1
          |______|______|______|______|______|______|
        */

        for (int j = 0; j < O; j++) {
          S0[j] = _S0[j];
        }
        S0[O + 1] = ZERO;

        S1[0] = ZERO;
        for (int j = 0; j < O; j++) {
          S1[j+1] = _S1[j];
        }

      } else if (shift_i == -1) {
        /*
            (-1)    0      1     ...     N     N+1
          __________________________________________
          |  *   |  x*  |  x*  |  //  |  x   |      |   // shift_i = -1
          |______|______|______|______|______|______|
        */

        S0[0] = ZERO;
        for (int j = 0; j < O; j++) {
          S0[j+1] = _S0[j];
        }

        for (int j = 0; j < O; j++) {
          S1[j] = _S1[j];
        }
        S1[O+1] = ZERO;

      } else if (shift_i == 0) {
        /*
            (-1)    0      1     ...      N     N+1
          __________________________________________
          |      |  x*  |  x*  |  //  |  x*  |      |  // shift_i = 0
          |______|______|______|______|______|______|
        */

        for (int j = 0; j < O; j++) {
          S0[j] = _S0[j];
        }
        S0[O + 1] = ZERO;

        for (int j = 0; j < O; j++) {
          S1[j] = _S1[j];
        }
        S1[O + 1] = ZERO;
      } else {
        raise::KernelError(HERE, "Invalid shift in indices");
      }

      // account for ghost cells here to shorten J update expression
      i_min += N_GHOSTS;
    }

  public:
    /**
     * @brief explicit constructor.
     */
    DepositCurrents_kernel(const scatter_ndfield_t<D, 3>& scatter_cur,
                           const array_t<int*>&           i1,
                           const array_t<int*>&           i2,
                           const array_t<int*>&           i3,
                           const array_t<int*>&           i1_prev,
                           const array_t<int*>&           i2_prev,
                           const array_t<int*>&           i3_prev,
                           const array_t<prtldx_t*>&      dx1,
                           const array_t<prtldx_t*>&      dx2,
                           const array_t<prtldx_t*>&      dx3,
                           const array_t<prtldx_t*>&      dx1_prev,
                           const array_t<prtldx_t*>&      dx2_prev,
                           const array_t<prtldx_t*>&      dx3_prev,
                           const array_t<real_t*>&        ux1,
                           const array_t<real_t*>&        ux2,
                           const array_t<real_t*>&        ux3,
                           const array_t<real_t*>&        phi,
                           const array_t<real_t*>&        weight,
                           const array_t<short*>&         tag,
                           const M&                       metric,
                           real_t                         charge,
                           const real_t                   dt)
      : J { scatter_cur }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , i1_prev { i1_prev }
      , i2_prev { i2_prev }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , dx1_prev { dx1_prev }
      , dx2_prev { dx2_prev }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , metric { metric }
      , charge { charge }
      , inv_dt { ONE / dt } {
      raise::ErrorIf(
        (O == 2u and N_GHOSTS < 2),
        "Order of interpolation is 2, but number of ghost cells is < 2",
        HERE);
    }

    /**
     * @brief Iteration of the loop over particles.
     * @param p index.
     */
    Inline auto operator()(index_t p) const -> void {
      if (tag(p) == ParticleTag::dead) {
        return;
      }
      // recover particle velocity to deposit in unsimulated direction
      vec_t<Dim::_3D> vp { ZERO };
      {
        coord_t<M::PrtlDim> xp { ZERO };
        if constexpr (D == Dim::_1D) {
          xp[0] = i_di_to_Xi(i1(p), dx1(p));
        } else if constexpr (D == Dim::_2D) {
          if constexpr (M::PrtlDim == Dim::_3D) {
            xp[0] = i_di_to_Xi(i1(p), dx1(p));
            xp[1] = i_di_to_Xi(i2(p), dx2(p));
            xp[2] = phi(p);
          } else {
            xp[0] = i_di_to_Xi(i1(p), dx1(p));
            xp[1] = i_di_to_Xi(i2(p), dx2(p));
          }
        } else {
          xp[0] = i_di_to_Xi(i1(p), dx1(p));
          xp[1] = i_di_to_Xi(i2(p), dx2(p));
          xp[2] = i_di_to_Xi(i3(p), dx3(p));
        }
        auto inv_energy { ZERO };
        if constexpr (S == SimEngine::SRPIC) {
          metric.template transform_xyz<Idx::XYZ, Idx::U>(xp,
                                                          { ux1(p), ux2(p), ux3(p) },
                                                          vp);
          inv_energy = ONE / math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
        } else {
          metric.template transform<Idx::D, Idx::U>(xp,
                                                    { ux1(p), ux2(p), ux3(p) },
                                                    vp);
          inv_energy = ONE / math::sqrt(ONE + ux1(p) * vp[0] + ux2(p) * vp[1] +
                                        ux3(p) * vp[2]);
        }
        if (Kokkos::isnan(vp[2]) || Kokkos::isinf(vp[2])) {
          vp[2] = ZERO;
        }
        vp[0] *= inv_energy;
        vp[1] *= inv_energy;
        vp[2] *= inv_energy;
      }

      const real_t coeff { weight(p) * charge };

      // ToDo: interpolation_order as parameter
      if constexpr (O == 1u) {
        /*
          Zig-zag deposit
        */

        const auto dxp_r_1 { static_cast<prtldx_t>(i1(p) == i1_prev(p)) *
                             (dx1(p) + dx1_prev(p)) *
                             static_cast<prtldx_t>(INV_2) };

        const real_t Wx1_1 { INV_2 * (dxp_r_1 + dx1_prev(p) +
                                      static_cast<real_t>(i1(p) > i1_prev(p))) };
        const real_t Wx1_2 { INV_2 * (dx1(p) + dxp_r_1 +
                                      static_cast<real_t>(
                                        static_cast<int>(i1(p) > i1_prev(p)) +
                                        i1_prev(p) - i1(p))) };
        const real_t Fx1_1 { (static_cast<real_t>(i1(p) > i1_prev(p)) +
                              dxp_r_1 - dx1_prev(p)) *
                             coeff * inv_dt };
        const real_t Fx1_2 { (static_cast<real_t>(
                                i1(p) - i1_prev(p) -
                                static_cast<int>(i1(p) > i1_prev(p))) +
                              dx1(p) - dxp_r_1) *
                             coeff * inv_dt };

        auto J_acc = J.access();

        // tuple_t<prtldx_t, D> dxp_r;
        if constexpr (D == Dim::_1D) {
          const real_t Fx2_1 { HALF * vp[1] * coeff };
          const real_t Fx2_2 { HALF * vp[1] * coeff };

          const real_t Fx3_1 { HALF * vp[2] * coeff };
          const real_t Fx3_2 { HALF * vp[2] * coeff };

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx1) += Fx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx1)      += Fx1_2;

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx2)     += Fx2_1 * (ONE - Wx1_1);
          J_acc(i1_prev(p) + N_GHOSTS + 1, cur::jx2) += Fx2_1 * Wx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx2)          += Fx2_2 * (ONE - Wx1_2);
          J_acc(i1(p) + N_GHOSTS + 1, cur::jx2)      += Fx2_2 * Wx1_2;

          J_acc(i1_prev(p) + N_GHOSTS, cur::jx3)     += Fx3_1 * (ONE - Wx1_1);
          J_acc(i1_prev(p) + N_GHOSTS + 1, cur::jx3) += Fx3_1 * Wx1_1;
          J_acc(i1(p) + N_GHOSTS, cur::jx3)          += Fx3_2 * (ONE - Wx1_2);
          J_acc(i1(p) + N_GHOSTS + 1, cur::jx3)      += Fx3_2 * Wx1_2;
        } else if constexpr (D == Dim::_2D || D == Dim::_3D) {
          const auto dxp_r_2 { static_cast<prtldx_t>(i2(p) == i2_prev(p)) *
                               (dx2(p) + dx2_prev(p)) *
                               static_cast<prtldx_t>(INV_2) };

          const real_t Wx2_1 { INV_2 * (dxp_r_2 + dx2_prev(p) +
                                        static_cast<real_t>(i2(p) > i2_prev(p))) };
          const real_t Wx2_2 { INV_2 * (dx2(p) + dxp_r_2 +
                                        static_cast<real_t>(
                                          static_cast<int>(i2(p) > i2_prev(p)) +
                                          i2_prev(p) - i2(p))) };
          const real_t Fx2_1 { (static_cast<real_t>(i2(p) > i2_prev(p)) +
                                dxp_r_2 - dx2_prev(p)) *
                               coeff * inv_dt };
          const real_t Fx2_2 { (static_cast<real_t>(
                                  i2(p) - i2_prev(p) -
                                  static_cast<int>(i2(p) > i2_prev(p))) +
                                dx2(p) - dxp_r_2) *
                               coeff * inv_dt };

          if constexpr (D == Dim::_2D) {
            const real_t Fx3_1 { HALF * vp[2] * coeff };
            const real_t Fx3_2 { HALF * vp[2] * coeff };

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * Wx2_1;
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS, cur::jx1) += Fx1_2 *
                                                                   (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS + 1, cur::jx1) += Fx1_2 * Wx2_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * Wx1_1;
            J_acc(i1(p) + N_GHOSTS, i2(p) + N_GHOSTS, cur::jx2) += Fx2_2 *
                                                                   (ONE - Wx1_2);
            J_acc(i1(p) + N_GHOSTS + 1, i2(p) + N_GHOSTS, cur::jx2) += Fx2_2 * Wx1_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_2 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_1 * Wx1_1 * Wx2_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS + 1,
                  cur::jx3) += Fx3_2 * Wx1_2 * Wx2_2;
          } else {
            const auto   dxp_r_3 { static_cast<prtldx_t>(i3(p) == i3_prev(p)) *
                                 (dx3(p) + dx3_prev(p)) *
                                 static_cast<prtldx_t>(INV_2) };
            const real_t Wx3_1 { INV_2 * (dxp_r_3 + dx3_prev(p) +
                                          static_cast<real_t>(i3(p) > i3_prev(p))) };
            const real_t Wx3_2 { INV_2 * (dx3(p) + dxp_r_3 +
                                          static_cast<real_t>(
                                            static_cast<int>(i3(p) > i3_prev(p)) +
                                            i3_prev(p) - i3(p))) };
            const real_t Fx3_1 { (static_cast<real_t>(i3(p) > i3_prev(p)) +
                                  dxp_r_3 - dx3_prev(p)) *
                                 coeff * inv_dt };
            const real_t Fx3_2 { (static_cast<real_t>(
                                    i3(p) - i3_prev(p) -
                                    static_cast<int>(i3(p) > i3_prev(p))) +
                                  dx3(p) - dxp_r_3) *
                                 coeff * inv_dt };

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1) * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx1) += Fx1_1 * Wx2_1 * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * (ONE - Wx2_1) * Wx3_1;
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_1 * Wx2_1 * Wx3_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx1) += Fx1_2 * (ONE - Wx2_2) * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx1) += Fx1_2 * Wx2_2 * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_2 * (ONE - Wx2_2) * Wx3_2;
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx1) += Fx1_2 * Wx2_2 * Wx3_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1) * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx2) += Fx2_1 * Wx1_1 * (ONE - Wx3_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_1 * (ONE - Wx1_1) * Wx3_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_1 * Wx1_1 * Wx3_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx2) += Fx2_2 * (ONE - Wx1_2) * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx2) += Fx2_2 * Wx1_2 * (ONE - Wx3_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_2 * (ONE - Wx1_2) * Wx3_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS + 1,
                  cur::jx2) += Fx2_2 * Wx1_2 * Wx3_2;

            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_1 * (ONE - Wx2_1);
            J_acc(i1_prev(p) + N_GHOSTS,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * (ONE - Wx1_1) * Wx2_1;
            J_acc(i1_prev(p) + N_GHOSTS + 1,
                  i2_prev(p) + N_GHOSTS + 1,
                  i3_prev(p) + N_GHOSTS,
                  cur::jx3) += Fx3_1 * Wx1_1 * Wx2_1;

            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * (ONE - Wx2_2);
            J_acc(i1(p) + N_GHOSTS,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * (ONE - Wx1_2) * Wx2_2;
            J_acc(i1(p) + N_GHOSTS + 1,
                  i2(p) + N_GHOSTS + 1,
                  i3(p) + N_GHOSTS,
                  cur::jx3) += Fx3_2 * Wx1_2 * Wx2_2;
          }
        }
      } else if constexpr (O == 2u) {
        /*
          Higher order charge conserving current deposition based on
          Esirkepov (2001) https://ui.adsabs.harvard.edu/abs/2001CoPhC.135..144E/abstract

          We need to define the follwowing variable:
          - Shape functions in spatial directions for the particle position
            before and after the current timestep.
            S0_*, S1_*
          - Density composition matrix
            Wx_*, Wy_*, Wz_*
        */

        /*
            x - direction
        */

        // shape function at previous timestep
        real_t   S0x_0, S0x_1, S0x_2, S0x_3;
        // shape function at current timestep
        real_t   S1x_0, S1x_1, S1x_2, S1x_3;
        // indices of the shape function
        ncells_t ix_min;
        bool     update_x2;
        // find indices and define shape function
        // clang-format off
        shape_function_2nd(S0x_0, S0x_1, S0x_2, S0x_3,
                           S1x_0, S1x_1, S1x_2, S1x_3,
                           ix_min, update_x2,
                           i1(p), dx1(p),
                           i1_prev(p), dx1_prev(p));
        // clang-format on

        if constexpr (D == Dim::_1D) {
          // ToDo
        } else if constexpr (D == Dim::_2D) {

          /*
            y - direction
          */

          // shape function at previous timestep
          real_t   S0y_0, S0y_1, S0y_2, S0y_3;
          // shape function at current timestep
          real_t   S1y_0, S1y_1, S1y_2, S1y_3;
          // indices of the shape function
          ncells_t iy_min;
          bool     update_y2;
          // find indices and define shape function
          // clang-format off
          shape_function_2nd(S0y_0, S0y_1, S0y_2, S0y_3,
                             S1y_0, S1y_1, S1y_2, S1y_3,
                             iy_min, update_y2,
                             i2(p), dx2(p),
                             i2_prev(p), dx2_prev(p));
          // clang-format on

          // Esirkepov 2001, Eq. 38
          /*
              x - component
          */
          // Calculate weight function - unrolled
          const auto Wx_0_0 = HALF * (S1x_0 - S0x_0) * (S0y_0 + S1y_0);
          const auto Wx_0_1 = HALF * (S1x_0 - S0x_0) * (S0y_1 + S1y_1);
          const auto Wx_0_2 = HALF * (S1x_0 - S0x_0) * (S0y_2 + S1y_2);
          const auto Wx_0_3 = HALF * (S1x_0 - S0x_0) * (S0y_3 + S1y_3);

          const auto Wx_1_0 = HALF * (S1x_1 - S0x_1) * (S0y_0 + S1y_0);
          const auto Wx_1_1 = HALF * (S1x_1 - S0x_1) * (S0y_1 + S1y_1);
          const auto Wx_1_2 = HALF * (S1x_1 - S0x_1) * (S0y_2 + S1y_2);
          const auto Wx_1_3 = HALF * (S1x_1 - S0x_1) * (S0y_3 + S1y_3);

          const auto Wx_2_0 = HALF * (S1x_2 - S0x_2) * (S0y_0 + S1y_0);
          const auto Wx_2_1 = HALF * (S1x_2 - S0x_2) * (S0y_1 + S1y_1);
          const auto Wx_2_2 = HALF * (S1x_2 - S0x_2) * (S0y_2 + S1y_2);
          const auto Wx_2_3 = HALF * (S1x_2 - S0x_2) * (S0y_3 + S1y_3);

          // Unrolled calculations for Wy
          const auto Wy_0_0 = HALF * (S1x_0 + S0x_0) * (S1y_0 - S0y_0);
          const auto Wy_0_1 = HALF * (S1x_0 + S0x_0) * (S1y_1 - S0y_1);
          const auto Wy_0_2 = HALF * (S1x_0 + S0x_0) * (S1y_2 - S0y_2);

          const auto Wy_1_0 = HALF * (S1x_1 + S0x_1) * (S1y_0 - S0y_0);
          const auto Wy_1_1 = HALF * (S1x_1 + S0x_1) * (S1y_1 - S0y_1);
          const auto Wy_1_2 = HALF * (S1x_1 + S0x_1) * (S1y_2 - S0y_2);

          const auto Wy_2_0 = HALF * (S1x_2 + S0x_2) * (S1y_0 - S0y_0);
          const auto Wy_2_1 = HALF * (S1x_2 + S0x_2) * (S1y_1 - S0y_1);
          const auto Wy_2_2 = HALF * (S1x_2 + S0x_2) * (S1y_2 - S0y_2);

          const auto Wy_3_0 = HALF * (S1x_3 + S0x_3) * (S1y_0 - S0y_0);
          const auto Wy_3_1 = HALF * (S1x_3 + S0x_3) * (S1y_1 - S0y_1);
          const auto Wy_3_2 = HALF * (S1x_3 + S0x_3) * (S1y_2 - S0y_2);

          // Unrolled calculations for Wz
          const auto Wz_0_0 = THIRD * (S1y_0 * (HALF * S0x_0 + S1x_0) +
                                       S0y_0 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_1 = THIRD * (S1y_1 * (HALF * S0x_0 + S1x_0) +
                                       S0y_1 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_2 = THIRD * (S1y_2 * (HALF * S0x_0 + S1x_0) +
                                       S0y_2 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_3 = THIRD * (S1y_3 * (HALF * S0x_0 + S1x_0) +
                                       S0y_3 * (HALF * S1x_0 + S0x_0));

          const auto Wz_1_0 = THIRD * (S1y_0 * (HALF * S0x_1 + S1x_1) +
                                       S0y_0 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_1 = THIRD * (S1y_1 * (HALF * S0x_1 + S1x_1) +
                                       S0y_1 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_2 = THIRD * (S1y_2 * (HALF * S0x_1 + S1x_1) +
                                       S0y_2 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_3 = THIRD * (S1y_3 * (HALF * S0x_1 + S1x_1) +
                                       S0y_3 * (HALF * S1x_1 + S0x_1));

          const auto Wz_2_0 = THIRD * (S1y_0 * (HALF * S0x_2 + S1x_2) +
                                       S0y_0 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_1 = THIRD * (S1y_1 * (HALF * S0x_2 + S1x_2) +
                                       S0y_1 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_2 = THIRD * (S1y_2 * (HALF * S0x_2 + S1x_2) +
                                       S0y_2 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_3 = THIRD * (S1y_3 * (HALF * S0x_2 + S1x_2) +
                                       S0y_3 * (HALF * S1x_2 + S0x_2));

          const auto Wz_3_0 = THIRD * (S1y_0 * (HALF * S0x_3 + S1x_3) +
                                       S0y_0 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_1 = THIRD * (S1y_1 * (HALF * S0x_3 + S1x_3) +
                                       S0y_1 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_2 = THIRD * (S1y_2 * (HALF * S0x_3 + S1x_3) +
                                       S0y_2 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_3 = THIRD * (S1y_3 * (HALF * S0x_3 + S1x_3) +
                                       S0y_3 * (HALF * S1x_3 + S0x_3));

          const real_t Qdxdt = coeff * inv_dt;
          const real_t Qdydt = coeff * inv_dt;
          const real_t QVz   = coeff * vp[2];

          // Esirkepov - Eq. 39
          // x-component
          const auto jx_0_0 = -Qdxdt * Wx_0_0;
          const auto jx_1_0 = jx_0_0 - Qdxdt * Wx_1_0;
          const auto jx_2_0 = jx_1_0 - Qdxdt * Wx_2_0;

          const auto jx_0_1 = -Qdxdt * Wx_0_1;
          const auto jx_1_1 = jx_0_1 - Qdxdt * Wx_1_1;
          const auto jx_2_1 = jx_1_1 - Qdxdt * Wx_2_1;

          const auto jx_0_2 = -Qdxdt * Wx_0_2;
          const auto jx_1_2 = jx_0_2 - Qdxdt * Wx_1_2;
          const auto jx_2_2 = jx_1_2 - Qdxdt * Wx_2_2;

          const auto jx_0_3 = -Qdxdt * Wx_0_3;
          const auto jx_1_3 = jx_0_3 - Qdxdt * Wx_1_3;
          const auto jx_2_3 = jx_1_3 - Qdxdt * Wx_2_3;

          // y-component
          const auto jy_0_0 = -Qdydt * Wy_0_0;
          const auto jy_0_1 = jy_0_0 - Qdydt * Wy_0_1;
          const auto jy_0_2 = jy_0_1 - Qdydt * Wy_0_2;

          const auto jy_1_0 = -Qdydt * Wy_1_0;
          const auto jy_1_1 = jy_1_0 - Qdydt * Wy_1_1;
          const auto jy_1_2 = jy_1_1 - Qdydt * Wy_1_2;

          const auto jy_2_0 = -Qdydt * Wy_2_0;
          const auto jy_2_1 = jy_2_0 - Qdydt * Wy_2_1;
          const auto jy_2_2 = jy_2_1 - Qdydt * Wy_2_2;

          const auto jy_3_0 = -Qdydt * Wy_3_0;
          const auto jy_3_1 = jy_3_0 - Qdydt * Wy_3_1;
          const auto jy_3_2 = jy_3_1 - Qdydt * Wy_3_2;

          /*
            Current update
          */
          auto J_acc = J.access();

          /*
              x - component
          */
          J_acc(ix_min, iy_min, cur::jx1)     += jx_0_0;
          J_acc(ix_min, iy_min + 1, cur::jx1) += jx_0_1;
          J_acc(ix_min, iy_min + 2, cur::jx1) += jx_0_2;

          J_acc(ix_min + 1, iy_min, cur::jx1)     += jx_1_0;
          J_acc(ix_min + 1, iy_min + 1, cur::jx1) += jx_1_1;
          J_acc(ix_min + 1, iy_min + 2, cur::jx1) += jx_1_2;

          if (update_x2) {
            J_acc(ix_min + 2, iy_min, cur::jx1)     += jx_2_0;
            J_acc(ix_min + 2, iy_min + 1, cur::jx1) += jx_2_1;
            J_acc(ix_min + 2, iy_min + 2, cur::jx1) += jx_2_2;
          }

          if (update_y2) {
            J_acc(ix_min + 1, iy_min + 3, cur::jx1) += jx_1_3;
            J_acc(ix_min, iy_min + 3, cur::jx1)     += jx_0_3;
          }

          if (update_x2 && update_y2) {
            J_acc(ix_min + 2, iy_min + 3, cur::jx1) += jx_2_3;
          }

          /*
              y - component
          */
          J_acc(ix_min, iy_min, cur::jx2)     += jy_0_0;
          J_acc(ix_min + 1, iy_min, cur::jx2) += jy_1_0;
          J_acc(ix_min + 2, iy_min, cur::jx2) += jy_2_0;

          J_acc(ix_min, iy_min + 1, cur::jx2)     += jy_0_1;
          J_acc(ix_min + 1, iy_min + 1, cur::jx2) += jy_1_1;
          J_acc(ix_min + 2, iy_min + 1, cur::jx2) += jy_2_1;

          if (update_x2) {
            J_acc(ix_min + 3, iy_min + 1, cur::jx2) += jy_3_1;
            J_acc(ix_min + 3, iy_min, cur::jx2)     += jy_3_0;
          }

          if (update_y2) {
            J_acc(ix_min, iy_min + 2, cur::jx2)     += jy_0_2;
            J_acc(ix_min + 1, iy_min + 2, cur::jx2) += jy_1_2;
            J_acc(ix_min + 2, iy_min + 2, cur::jx2) += jy_2_2;
          }

          if (update_x2 && update_y2) {
            J_acc(ix_min + 3, iy_min + 2, cur::jx2) += jy_3_2;
          }
          /*
              z - component, unsimulated direction
          */
          J_acc(ix_min, iy_min, cur::jx3)     += QVz * Wz_0_0;
          J_acc(ix_min, iy_min + 1, cur::jx3) += QVz * Wz_0_1;
          J_acc(ix_min, iy_min + 2, cur::jx3) += QVz * Wz_0_2;

          J_acc(ix_min + 1, iy_min, cur::jx3)     += QVz * Wz_1_0;
          J_acc(ix_min + 1, iy_min + 1, cur::jx3) += QVz * Wz_1_1;
          J_acc(ix_min + 1, iy_min + 2, cur::jx3) += QVz * Wz_1_2;

          J_acc(ix_min + 2, iy_min, cur::jx3)     += QVz * Wz_2_0;
          J_acc(ix_min + 2, iy_min + 1, cur::jx3) += QVz * Wz_2_1;
          J_acc(ix_min + 2, iy_min + 2, cur::jx3) += QVz * Wz_2_2;

          if (update_x2) {
            J_acc(ix_min + 3, iy_min, cur::jx3)     += QVz * Wz_3_0;
            J_acc(ix_min + 3, iy_min + 1, cur::jx3) += QVz * Wz_3_1;
            J_acc(ix_min + 3, iy_min + 2, cur::jx3) += QVz * Wz_3_2;
          }

          if (update_y2) {
            J_acc(ix_min, iy_min + 3, cur::jx3)     += QVz * Wz_0_3;
            J_acc(ix_min + 1, iy_min + 3, cur::jx3) += QVz * Wz_1_3;
            J_acc(ix_min + 2, iy_min + 3, cur::jx3) += QVz * Wz_2_3;
          }
          if (update_x2 && update_y2) {
            J_acc(ix_min + 3, iy_min + 3, cur::jx3) += QVz * Wz_3_3;
          }

        } else if constexpr (D == Dim::_3D) {
          /*
            y - direction
          */

          // shape function at previous timestep
          real_t   S0y_0, S0y_1, S0y_2, S0y_3;
          // shape function at current timestep
          real_t   S1y_0, S1y_1, S1y_2, S1y_3;
          // indices of the shape function
          ncells_t iy_min;
          bool     update_y2;
          // find indices and define shape function
          // clang-format off
          shape_function_2nd(S0y_0, S0y_1, S0y_2, S0y_3,
                             S1y_0, S1y_1, S1y_2, S1y_3,
                             iy_min, update_y2,
                             i2(p), dx2(p),
                             i2_prev(p), dx2_prev(p));
          // clang-format on

          /*
            y - direction
          */

          // shape function at previous timestep
          real_t   S0z_0, S0z_1, S0z_2, S0z_3;
          // shape function at current timestep
          real_t   S1z_0, S1z_1, S1z_2, S1z_3;
          // indices of the shape function
          ncells_t iz_min;
          bool     update_z2;
          // find indices and define shape function
          // clang-format off
          shape_function_2nd(S0z_0, S0z_1, S0z_2, S0z_3,
                             S1z_0, S1z_1, S1z_2, S1z_3,
                             iz_min, update_z2,
                             i3(p), dx3(p),
                             i3_prev(p), dx3_prev(p));
          // clang-format on

          // Unrolled calculations for Wx, Wy, and Wz
          // clang-format off
          const auto Wx_0_0_0 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
                                 HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          const auto Wx_0_0_1 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
                                 HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          const auto Wx_0_0_2 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
                                 HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          const auto Wx_0_0_3 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
                                 HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          
          const auto Wx_0_1_0 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
                                 HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          const auto Wx_0_1_1 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
                                 HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          const auto Wx_0_1_2 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
                                 HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          const auto Wx_0_1_3 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
                                 HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          
          const auto Wx_0_2_0 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
                                 HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          const auto Wx_0_2_1 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
                                 HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          const auto Wx_0_2_2 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
                                 HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          const auto Wx_0_2_3 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
                                 HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          
          const auto Wx_0_3_0 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
                                 HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          const auto Wx_0_3_1 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
                                 HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          const auto Wx_0_3_2 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
                                 HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          const auto Wx_0_3_3 = THIRD * (S1x_0 - S0x_0) *
                                ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
                                 HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          
          const auto Wx_1_0_0 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
                                 HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          const auto Wx_1_0_1 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
                                 HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          const auto Wx_1_0_2 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
                                 HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          const auto Wx_1_0_3 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
                                 HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          
          const auto Wx_1_1_0 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
                                 HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          const auto Wx_1_1_1 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
                                 HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          const auto Wx_1_1_2 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
                                 HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          const auto Wx_1_1_3 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
                                 HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          
          const auto Wx_1_2_0 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
                                 HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          const auto Wx_1_2_1 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
                                 HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          const auto Wx_1_2_2 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
                                 HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          const auto Wx_1_2_3 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
                                 HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          
          const auto Wx_1_3_0 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
                                 HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          const auto Wx_1_3_1 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
                                 HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          const auto Wx_1_3_2 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
                                 HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          const auto Wx_1_3_3 = THIRD * (S1x_1 - S0x_1) *
                                ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
                                 HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));
          
          const auto Wx_2_0_0 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_0 * S0z_0 + S1y_0 * S1z_0) +
                                 HALF * (S0z_0 * S1y_0 + S0y_0 * S1z_0));
          const auto Wx_2_0_1 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_0 * S0z_1 + S1y_0 * S1z_1) +
                                 HALF * (S0z_1 * S1y_0 + S0y_0 * S1z_1));
          const auto Wx_2_0_2 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_0 * S0z_2 + S1y_0 * S1z_2) +
                                 HALF * (S0z_2 * S1y_0 + S0y_0 * S1z_2));
          const auto Wx_2_0_3 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_0 * S0z_3 + S1y_0 * S1z_3) +
                                 HALF * (S0z_3 * S1y_0 + S0y_0 * S1z_3));
          
          const auto Wx_2_1_0 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_1 * S0z_0 + S1y_1 * S1z_0) +
                                 HALF * (S0z_0 * S1y_1 + S0y_1 * S1z_0));
          const auto Wx_2_1_1 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_1 * S0z_1 + S1y_1 * S1z_1) +
                                 HALF * (S0z_1 * S1y_1 + S0y_1 * S1z_1));
          const auto Wx_2_1_2 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_1 * S0z_2 + S1y_1 * S1z_2) +
                                 HALF * (S0z_2 * S1y_1 + S0y_1 * S1z_2));
          const auto Wx_2_1_3 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_1 * S0z_3 + S1y_1 * S1z_3) +
                                 HALF * (S0z_3 * S1y_1 + S0y_1 * S1z_3));
          
          const auto Wx_2_2_0 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_2 * S0z_0 + S1y_2 * S1z_0) +
                                 HALF * (S0z_0 * S1y_2 + S0y_2 * S1z_0));
          const auto Wx_2_2_1 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_2 * S0z_1 + S1y_2 * S1z_1) +
                                 HALF * (S0z_1 * S1y_2 + S0y_2 * S1z_1));
          const auto Wx_2_2_2 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_2 * S0z_2 + S1y_2 * S1z_2) +
                                 HALF * (S0z_2 * S1y_2 + S0y_2 * S1z_2));
          const auto Wx_2_2_3 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_2 * S0z_3 + S1y_2 * S1z_3) +
                                 HALF * (S0z_3 * S1y_2 + S0y_2 * S1z_3));
          
          const auto Wx_2_3_0 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_3 * S0z_0 + S1y_3 * S1z_0) +
                                 HALF * (S0z_0 * S1y_3 + S0y_3 * S1z_0));
          const auto Wx_2_3_1 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_3 * S0z_1 + S1y_3 * S1z_1) +
                                 HALF * (S0z_1 * S1y_3 + S0y_3 * S1z_1));
          const auto Wx_2_3_2 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_3 * S0z_2 + S1y_3 * S1z_2) +
                                 HALF * (S0z_2 * S1y_3 + S0y_3 * S1z_2));
          const auto Wx_2_3_3 = THIRD * (S1x_2 - S0x_2) *
                                ((S0y_3 * S0z_3 + S1y_3 * S1z_3) +
                                 HALF * (S0z_3 * S1y_3 + S0y_3 * S1z_3));

          const real_t Qdxdt = coeff * inv_dt;

          const auto jx_0_0_0 =          - Qdxdt * Wx_0_0_0;
          const auto jx_1_0_0 = jx_0_0_0 - Qdxdt * Wx_1_0_0;
          const auto jx_2_0_0 = jx_1_0_0 - Qdxdt * Wx_2_0_0;
          const auto jx_0_1_0 =          - Qdxdt * Wx_0_1_0;
          const auto jx_1_1_0 = jx_0_1_0 - Qdxdt * Wx_1_1_0;
          const auto jx_2_1_0 = jx_1_1_0 - Qdxdt * Wx_2_1_0;
          const auto jx_0_2_0 =          - Qdxdt * Wx_0_2_0;
          const auto jx_1_2_0 = jx_0_2_0 - Qdxdt * Wx_1_2_0;
          const auto jx_2_2_0 = jx_1_2_0 - Qdxdt * Wx_2_2_0;
          const auto jx_0_3_0 =          - Qdxdt * Wx_0_3_0;
          const auto jx_1_3_0 = jx_0_3_0 - Qdxdt * Wx_1_3_0;
          const auto jx_2_3_0 = jx_1_3_0 - Qdxdt * Wx_2_3_0;

          const auto jx_0_0_1 =          - Qdxdt * Wx_0_0_1;
          const auto jx_1_0_1 = jx_0_0_1 - Qdxdt * Wx_1_0_1;
          const auto jx_2_0_1 = jx_1_0_1 - Qdxdt * Wx_2_0_1;
          const auto jx_0_1_1 =          - Qdxdt * Wx_0_1_1;
          const auto jx_1_1_1 = jx_0_1_1 - Qdxdt * Wx_1_1_1;
          const auto jx_2_1_1 = jx_1_1_1 - Qdxdt * Wx_2_1_1;
          const auto jx_0_2_1 =          - Qdxdt * Wx_0_2_1;
          const auto jx_1_2_1 = jx_0_2_1 - Qdxdt * Wx_1_2_1;
          const auto jx_2_2_1 = jx_1_2_1 - Qdxdt * Wx_2_2_1;
          const auto jx_0_3_1 =          - Qdxdt * Wx_0_3_1;
          const auto jx_1_3_1 = jx_0_3_1 - Qdxdt * Wx_1_3_1;
          const auto jx_2_3_1 = jx_1_3_1 - Qdxdt * Wx_2_3_1;

          const auto jx_0_0_2 =          - Qdxdt * Wx_0_0_2;
          const auto jx_1_0_2 = jx_0_0_2 - Qdxdt * Wx_1_0_2;
          const auto jx_2_0_2 = jx_1_0_2 - Qdxdt * Wx_2_0_2;
          const auto jx_0_1_2 =          - Qdxdt * Wx_0_1_2;
          const auto jx_1_1_2 = jx_0_1_2 - Qdxdt * Wx_1_1_2;
          const auto jx_2_1_2 = jx_1_1_2 - Qdxdt * Wx_2_1_2;
          const auto jx_0_2_2 =          - Qdxdt * Wx_0_2_2;
          const auto jx_1_2_2 = jx_0_2_2 - Qdxdt * Wx_1_2_2;
          const auto jx_2_2_2 = jx_1_2_2 - Qdxdt * Wx_2_2_2;
          const auto jx_0_3_2 =          - Qdxdt * Wx_0_3_2;
          const auto jx_1_3_2 = jx_0_3_2 - Qdxdt * Wx_1_3_2;
          const auto jx_2_3_2 = jx_1_3_2 - Qdxdt * Wx_2_3_2;

          const auto jx_0_0_3 =          - Qdxdt * Wx_0_0_3;
          const auto jx_1_0_3 = jx_0_0_3 - Qdxdt * Wx_1_0_3;
          const auto jx_2_0_3 = jx_1_0_3 - Qdxdt * Wx_2_0_3;
          const auto jx_0_1_3 =          - Qdxdt * Wx_0_1_3;
          const auto jx_1_1_3 = jx_0_1_3 - Qdxdt * Wx_1_1_3;
          const auto jx_2_1_3 = jx_1_1_3 - Qdxdt * Wx_2_1_3;
          const auto jx_0_2_3 =          - Qdxdt * Wx_0_2_3;
          const auto jx_1_2_3 = jx_0_2_3 - Qdxdt * Wx_1_2_3;
          const auto jx_2_2_3 = jx_1_2_3 - Qdxdt * Wx_2_2_3;
          const auto jx_0_3_3 =          - Qdxdt * Wx_0_3_3;
          const auto jx_1_3_3 = jx_0_3_3 - Qdxdt * Wx_1_3_3;
          const auto jx_2_3_3 = jx_1_3_3 - Qdxdt * Wx_2_3_3;
          
          /*
            y-component
          */
          const auto Wy_0_0_0 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
                                 HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          const auto Wy_0_0_1 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
                                 HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          const auto Wy_0_0_2 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
                                 HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          const auto Wy_0_0_3 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
                                 HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          
          const auto Wy_0_1_0 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
                                 HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          const auto Wy_0_1_1 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
                                 HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          const auto Wy_0_1_2 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
                                 HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          const auto Wy_0_1_3 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
                                 HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          
          const auto Wy_0_2_0 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_0 * S0z_0 + S1x_0 * S1z_0 +
                                 HALF * (S0z_0 * S1x_0 + S0x_0 * S1z_0));
          const auto Wy_0_2_1 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_0 * S0z_1 + S1x_0 * S1z_1 +
                                 HALF * (S0z_1 * S1x_0 + S0x_0 * S1z_1));
          const auto Wy_0_2_2 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_0 * S0z_2 + S1x_0 * S1z_2 +
                                 HALF * (S0z_2 * S1x_0 + S0x_0 * S1z_2));
          const auto Wy_0_2_3 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_0 * S0z_3 + S1x_0 * S1z_3 +
                                 HALF * (S0z_3 * S1x_0 + S0x_0 * S1z_3));
          
          const auto Wy_1_0_0 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
                                 HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          const auto Wy_1_0_1 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
                                 HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          const auto Wy_1_0_2 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
                                 HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          const auto Wy_1_0_3 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
                                 HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          
          const auto Wy_1_1_0 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
                                 HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          const auto Wy_1_1_1 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
                                 HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          const auto Wy_1_1_2 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
                                 HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          const auto Wy_1_1_3 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
                                 HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));
          
          const auto Wy_1_2_0 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_1 * S0z_0 + S1x_1 * S1z_0 +
                                 HALF * (S0z_0 * S1x_1 + S0x_1 * S1z_0));
          const auto Wy_1_2_1 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_1 * S0z_1 + S1x_1 * S1z_1 +
                                 HALF * (S0z_1 * S1x_1 + S0x_1 * S1z_1));
          const auto Wy_1_2_2 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_1 * S0z_2 + S1x_1 * S1z_2 +
                                 HALF * (S0z_2 * S1x_1 + S0x_1 * S1z_2));
          const auto Wy_1_2_3 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_1 * S0z_3 + S1x_1 * S1z_3 +
                                 HALF * (S0z_3 * S1x_1 + S0x_1 * S1z_3));

          const auto Wy_2_0_0 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
                                 HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
          const auto Wy_2_0_1 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_2 * S0z_1 + S1x_2 * S1z_1 +
                                 HALF * (S0z_1 * S1x_2 + S0x_2 * S1z_1));
          const auto Wy_2_0_2 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_2 * S0z_2 + S1x_2 * S1z_2 +
                                 HALF * (S0z_2 * S1x_2 + S0x_2 * S1z_2));
          const auto Wy_2_0_3 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_2 * S0z_3 + S1x_2 * S1z_3 +
                                 HALF * (S0z_3 * S1x_2 + S0x_2 * S1z_3));
          
          const auto Wy_2_1_0 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
                                 HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
          const auto Wy_2_1_1 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_2 * S0z_1 + S1x_2 * S1z_1 +
                                 HALF * (S0z_1 * S1x_2 + S0x_2 * S1z_1));
          const auto Wy_2_1_2 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_2 * S0z_2 + S1x_2 * S1z_2 +
                                 HALF * (S0z_2 * S1x_2 + S0x_2 * S1z_2));
          const auto Wy_2_1_3 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_2 * S0z_3 + S1x_2 * S1z_3 +
                                 HALF * (S0z_3 * S1x_2 + S0x_2 * S1z_3));
          
          const auto Wy_2_2_0 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_2 * S0z_0 + S1x_2 * S1z_0 +
                                 HALF * (S0z_0 * S1x_2 + S0x_2 * S1z_0));
          const auto Wy_2_2_1 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_2 * S0z_1 + S1x_2 * S1z_1 +
                                 HALF * (S0z_1 * S1x_2 + S0x_2 * S1z_1));
          const auto Wy_2_2_2 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_2 * S0z_2 + S1x_2 * S1z_2 +
                                 HALF * (S0z_2 * S1x_2 + S0x_2 * S1z_2));
          const auto Wy_2_2_3 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_2 * S0z_3 + S1x_2 * S1z_3 +
                                 HALF * (S0z_3 * S1x_2 + S0x_2 * S1z_3));
          
          const auto Wy_3_0_0 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_3 * S0z_0 + S1x_3 * S1z_0 +
                                 HALF * (S0z_0 * S1x_3 + S0x_3 * S1z_0));
          const auto Wy_3_0_1 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_3 * S0z_1 + S1x_3 * S1z_1 +
                                 HALF * (S0z_1 * S1x_3 + S0x_3 * S1z_1));
          const auto Wy_3_0_2 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_3 * S0z_2 + S1x_3 * S1z_2 +
                                 HALF * (S0z_2 * S1x_3 + S0x_3 * S1z_2));
          const auto Wy_3_0_3 = THIRD * (S1y_0 - S0y_0) *
                                (S0x_3 * S0z_3 + S1x_3 * S1z_3 +
                                 HALF * (S0z_3 * S1x_3 + S0x_3 * S1z_3));
          
          const auto Wy_3_1_0 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_3 * S0z_0 + S1x_3 * S1z_0 +
                                 HALF * (S0z_0 * S1x_3 + S0x_3 * S1z_0));
          const auto Wy_3_1_1 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_3 * S0z_1 + S1x_3 * S1z_1 +
                                 HALF * (S0z_1 * S1x_3 + S0x_3 * S1z_1));
          const auto Wy_3_1_2 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_3 * S0z_2 + S1x_3 * S1z_2 +
                                 HALF * (S0z_2 * S1x_3 + S0x_3 * S1z_2));
          const auto Wy_3_1_3 = THIRD * (S1y_1 - S0y_1) *
                                (S0x_3 * S0z_3 + S1x_3 * S1z_3 +
                                 HALF * (S0z_3 * S1x_3 + S0x_3 * S1z_3));
          
          const auto Wy_3_2_0 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_3 * S0z_0 + S1x_3 * S1z_0 +
                                 HALF * (S0z_0 * S1x_3 + S0x_3 * S1z_0));
          const auto Wy_3_2_1 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_3 * S0z_1 + S1x_3 * S1z_1 +
                                 HALF * (S0z_1 * S1x_3 + S0x_3 * S1z_1));
          const auto Wy_3_2_2 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_3 * S0z_2 + S1x_3 * S1z_2 +
                                 HALF * (S0z_2 * S1x_3 + S0x_3 * S1z_2));
          const auto Wy_3_2_3 = THIRD * (S1y_2 - S0y_2) *
                                (S0x_3 * S0z_3 + S1x_3 * S1z_3 +
                                 HALF * (S0z_3 * S1x_3 + S0x_3 * S1z_3));
          
          const real_t Qdydt = coeff * inv_dt;

          const auto jy_0_0_0 =          - Qdydt * Wy_0_0_0;
          const auto jy_0_1_0 = jy_0_0_0 - Qdydt * Wy_0_1_0;
          const auto jy_0_2_0 = jy_0_1_0 - Qdydt * Wy_0_2_0;
          const auto jy_1_0_0 =          - Qdydt * Wy_1_0_0;
          const auto jy_1_1_0 = jy_1_0_0 - Qdydt * Wy_1_1_0;
          const auto jy_1_2_0 = jy_1_1_0 - Qdydt * Wy_1_2_0;
          const auto jy_2_0_0 =          - Qdydt * Wy_2_0_0;
          const auto jy_2_1_0 = jy_2_0_0 - Qdydt * Wy_2_1_0;
          const auto jy_2_2_0 = jy_2_1_0 - Qdydt * Wy_2_2_0;
          const auto jy_3_0_0 =          - Qdydt * Wy_3_0_0;
          const auto jy_3_1_0 = jy_3_0_0 - Qdydt * Wy_3_1_0;
          const auto jy_3_2_0 = jy_3_1_0 - Qdydt * Wy_3_2_0;

          const auto jy_0_0_1 =          - Qdydt * Wy_0_0_1;
          const auto jy_0_1_1 = jy_0_0_1 - Qdydt * Wy_0_1_1;
          const auto jy_0_2_1 = jy_0_1_1 - Qdydt * Wy_0_2_1;
          const auto jy_1_0_1 =          - Qdydt * Wy_1_0_1;
          const auto jy_1_1_1 = jy_1_0_1 - Qdydt * Wy_1_1_1;
          const auto jy_1_2_1 = jy_1_1_1 - Qdydt * Wy_1_2_1;
          const auto jy_2_0_1 =          - Qdydt * Wy_2_0_1;
          const auto jy_2_1_1 = jy_2_0_1 - Qdydt * Wy_2_1_1;
          const auto jy_2_2_1 = jy_2_1_1 - Qdydt * Wy_2_2_1;
          const auto jy_3_0_1 =          - Qdydt * Wy_3_0_1;
          const auto jy_3_1_1 = jy_3_0_1 - Qdydt * Wy_3_1_1;
          const auto jy_3_2_1 = jy_3_1_1 - Qdydt * Wy_3_2_1;

          const auto jy_0_0_2 =          - Qdydt * Wy_0_0_2;
          const auto jy_0_1_2 = jy_0_0_2 - Qdydt * Wy_0_1_2;
          const auto jy_0_2_2 = jy_0_1_2 - Qdydt * Wy_0_2_2;
          const auto jy_1_0_2 =          - Qdydt * Wy_1_0_2;
          const auto jy_1_1_2 = jy_1_0_2 - Qdydt * Wy_1_1_2;
          const auto jy_1_2_2 = jy_1_1_2 - Qdydt * Wy_1_2_2;
          const auto jy_2_0_2 =          - Qdydt * Wy_2_0_2;
          const auto jy_2_1_2 = jy_2_0_2 - Qdydt * Wy_2_1_2;
          const auto jy_2_2_2 = jy_2_1_2 - Qdydt * Wy_2_2_2;
          const auto jy_3_0_2 =          - Qdydt * Wy_3_0_2;
          const auto jy_3_1_2 = jy_3_0_2 - Qdydt * Wy_3_1_2;
          const auto jy_3_2_2 = jy_3_1_2 - Qdydt * Wy_3_2_2;

          const auto jy_0_0_3 =          - Qdydt * Wy_0_0_3;
          const auto jy_0_1_3 = jy_0_0_3 - Qdydt * Wy_0_1_3;
          const auto jy_0_2_3 = jy_0_1_3 - Qdydt * Wy_0_2_3;
          const auto jy_1_0_3 =          - Qdydt * Wy_1_0_3;
          const auto jy_1_1_3 = jy_1_0_3 - Qdydt * Wy_1_1_3;
          const auto jy_1_2_3 = jy_1_1_3 - Qdydt * Wy_1_2_3;
          const auto jy_2_0_3 =          - Qdydt * Wy_2_0_3;
          const auto jy_2_1_3 = jy_2_0_3 - Qdydt * Wy_2_1_3;
          const auto jy_2_2_3 = jy_2_1_3 - Qdydt * Wy_2_2_3;
          const auto jy_3_0_3 =          - Qdydt * Wy_3_0_3;
          const auto jy_3_1_3 = jy_3_0_3 - Qdydt * Wy_3_1_3;
          const auto jy_3_2_3 = jy_3_1_3 - Qdydt * Wy_3_2_3;

          /*
            z - component
          */
          const auto Wz_0_0_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_0 * S0y_0 + S1x_0 * S1y_0 +
                                 HALF * (S0x_0 * S1y_0 + S0y_0 * S1x_0));
          const auto Wz_0_0_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_0 * S0y_0 + S1x_0 * S1y_0 +
                                 HALF * (S0x_0 * S1y_0 + S0y_0 * S1x_0));
          const auto Wz_0_0_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_0 * S0y_0 + S1x_0 * S1y_0 +
                                 HALF * (S0x_0 * S1y_0 + S0y_0 * S1x_0));
          
          const auto Wz_0_1_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_0 * S0y_1 + S1x_0 * S1y_1 +
                                 HALF * (S0x_0 * S1y_1 + S0y_1 * S1x_0));
          const auto Wz_0_1_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_0 * S0y_1 + S1x_0 * S1y_1 +
                                 HALF * (S0x_0 * S1y_1 + S0y_1 * S1x_0));
          const auto Wz_0_1_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_0 * S0y_1 + S1x_0 * S1y_1 +
                                 HALF * (S0x_0 * S1y_1 + S0y_1 * S1x_0));
          
          const auto Wz_0_2_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_0 * S0y_2 + S1x_0 * S1y_2 +
                                 HALF * (S0x_0 * S1y_2 + S0y_2 * S1x_0));
          const auto Wz_0_2_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_0 * S0y_2 + S1x_0 * S1y_2 +
                                 HALF * (S0x_0 * S1y_2 + S0y_2 * S1x_0));
          const auto Wz_0_2_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_0 * S0y_2 + S1x_0 * S1y_2 +
                                 HALF * (S0x_0 * S1y_2 + S0y_2 * S1x_0));
          
          const auto Wz_0_3_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_0 * S0y_3 + S1x_0 * S1y_3 +
                                 HALF * (S0x_0 * S1y_3 + S0y_3 * S1x_0));
          const auto Wz_0_3_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_0 * S0y_3 + S1x_0 * S1y_3 +
                                 HALF * (S0x_0 * S1y_3 + S0y_3 * S1x_0));
          const auto Wz_0_3_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_0 * S0y_3 + S1x_0 * S1y_3 +
                                 HALF * (S0x_0 * S1y_3 + S0y_3 * S1x_0));
          
          // Unrolled loop for Wz[i][j][k] with i = 1 and interp_order + 2 = 4
          const auto Wz_1_0_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_1 * S0y_0 + S1x_1 * S1y_0 +
                                 HALF * (S0x_1 * S1y_0 + S0y_0 * S1x_1));
          const auto Wz_1_0_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_1 * S0y_0 + S1x_1 * S1y_0 +
                                 HALF * (S0x_1 * S1y_0 + S0y_0 * S1x_1));
          const auto Wz_1_0_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_1 * S0y_0 + S1x_1 * S1y_0 +
                                 HALF * (S0x_1 * S1y_0 + S0y_0 * S1x_1));
          
          const auto Wz_1_1_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_1 * S0y_1 + S1x_1 * S1y_1 +
                                 HALF * (S0x_1 * S1y_1 + S0y_1 * S1x_1));
          const auto Wz_1_1_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_1 * S0y_1 + S1x_1 * S1y_1 +
                                 HALF * (S0x_1 * S1y_1 + S0y_1 * S1x_1));
          const auto Wz_1_1_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_1 * S0y_1 + S1x_1 * S1y_1 +
                                 HALF * (S0x_1 * S1y_1 + S0y_1 * S1x_1));
          
          const auto Wz_1_2_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_1 * S0y_2 + S1x_1 * S1y_2 +
                                 HALF * (S0x_1 * S1y_2 + S0y_2 * S1x_1));
          const auto Wz_1_2_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_1 * S0y_2 + S1x_1 * S1y_2 +
                                 HALF * (S0x_1 * S1y_2 + S0y_2 * S1x_1));
          const auto Wz_1_2_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_1 * S0y_2 + S1x_1 * S1y_2 +
                                 HALF * (S0x_1 * S1y_2 + S0y_2 * S1x_1));
          
          const auto Wz_1_3_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_1 * S0y_3 + S1x_1 * S1y_3 +
                                 HALF * (S0x_1 * S1y_3 + S0y_3 * S1x_1));
          const auto Wz_1_3_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_1 * S0y_3 + S1x_1 * S1y_3 +
                                 HALF * (S0x_1 * S1y_3 + S0y_3 * S1x_1));
          const auto Wz_1_3_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_1 * S0y_3 + S1x_1 * S1y_3 +
                                 HALF * (S0x_1 * S1y_3 + S0y_3 * S1x_1));

          // Unrolled loop for Wz[i][j][k] with i = 2 and interp_order + 2 = 4
          const auto Wz_2_0_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_2 * S0y_0 + S1x_2 * S1y_0 +
                                 HALF * (S0x_2 * S1y_0 + S0y_0 * S1x_2));
          const auto Wz_2_0_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_2 * S0y_0 + S1x_2 * S1y_0 +
                                 HALF * (S0x_2 * S1y_0 + S0y_0 * S1x_2));
          const auto Wz_2_0_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_2 * S0y_0 + S1x_2 * S1y_0 +
                                 HALF * (S0x_2 * S1y_0 + S0y_0 * S1x_2));
          
          const auto Wz_2_1_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_2 * S0y_1 + S1x_2 * S1y_1 +
                                 HALF * (S0x_2 * S1y_1 + S0y_1 * S1x_2));
          const auto Wz_2_1_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_2 * S0y_1 + S1x_2 * S1y_1 +
                                 HALF * (S0x_2 * S1y_1 + S0y_1 * S1x_2));
          const auto Wz_2_1_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_2 * S0y_1 + S1x_2 * S1y_1 +
                                 HALF * (S0x_2 * S1y_1 + S0y_1 * S1x_2));
          
          const auto Wz_2_2_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_2 * S0y_2 + S1x_2 * S1y_2 +
                                 HALF * (S0x_2 * S1y_2 + S0y_2 * S1x_2));
          const auto Wz_2_2_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_2 * S0y_2 + S1x_2 * S1y_2 +
                                 HALF * (S0x_2 * S1y_2 + S0y_2 * S1x_2));
          const auto Wz_2_2_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_2 * S0y_2 + S1x_2 * S1y_2 +
                                 HALF * (S0x_2 * S1y_2 + S0y_2 * S1x_2));
          
          const auto Wz_2_3_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_2 * S0y_3 + S1x_2 * S1y_3 +
                                 HALF * (S0x_2 * S1y_3 + S0y_3 * S1x_2));
          const auto Wz_2_3_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_2 * S0y_3 + S1x_2 * S1y_3 +
                                 HALF * (S0x_2 * S1y_3 + S0y_3 * S1x_2));
          const auto Wz_2_3_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_2 * S0y_3 + S1x_2 * S1y_3 +
                                 HALF * (S0x_2 * S1y_3 + S0y_3 * S1x_2));

          // Unrolled loop for Wz[i][j][k] with i = 3 and interp_order + 2 = 4
          const auto Wz_3_0_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_3 * S0y_0 + S1x_3 * S1y_0 +
                                 HALF * (S0x_3 * S1y_0 + S0y_0 * S1x_3));
          const auto Wz_3_0_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_3 * S0y_0 + S1x_3 * S1y_0 +
                                 HALF * (S0x_3 * S1y_0 + S0y_0 * S1x_3));
          const auto Wz_3_0_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_3 * S0y_0 + S1x_3 * S1y_0 +
                                 HALF * (S0x_3 * S1y_0 + S0y_0 * S1x_3));
          
          const auto Wz_3_1_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_3 * S0y_1 + S1x_3 * S1y_1 +
                                 HALF * (S0x_3 * S1y_1 + S0y_1 * S1x_3));
          const auto Wz_3_1_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_3 * S0y_1 + S1x_3 * S1y_1 +
                                 HALF * (S0x_3 * S1y_1 + S0y_1 * S1x_3));
          const auto Wz_3_1_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_3 * S0y_1 + S1x_3 * S1y_1 +
                                 HALF * (S0x_3 * S1y_1 + S0y_1 * S1x_3));
          
          const auto Wz_3_2_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_3 * S0y_2 + S1x_3 * S1y_2 +
                                 HALF * (S0x_3 * S1y_2 + S0y_2 * S1x_3));
          const auto Wz_3_2_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_3 * S0y_2 + S1x_3 * S1y_2 +
                                 HALF * (S0x_3 * S1y_2 + S0y_2 * S1x_3));
          const auto Wz_3_2_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_3 * S0y_2 + S1x_3 * S1y_2 +
                                 HALF * (S0x_3 * S1y_2 + S0y_2 * S1x_3));
          
          const auto Wz_3_3_0 = THIRD * (S1z_0 - S0z_0) *
                                (S0x_3 * S0y_3 + S1x_3 * S1y_3 +
                                 HALF * (S0x_3 * S1y_3 + S0y_3 * S1x_3));
          const auto Wz_3_3_1 = THIRD * (S1z_1 - S0z_1) *
                                (S0x_3 * S0y_3 + S1x_3 * S1y_3 +
                                 HALF * (S0x_3 * S1y_3 + S0y_3 * S1x_3));
          const auto Wz_3_3_2 = THIRD * (S1z_2 - S0z_2) *
                                (S0x_3 * S0y_3 + S1x_3 * S1y_3 +
                                 HALF * (S0x_3 * S1y_3 + S0y_3 * S1x_3));

          const real_t Qdzdt = coeff * inv_dt;

          const auto jz_0_0_0 =          - Qdzdt * Wz_0_0_0;
          const auto jz_0_0_1 = jz_0_0_0 - Qdzdt * Wz_0_0_1;
          const auto jz_0_0_2 = jz_0_0_1 - Qdzdt * Wz_0_0_2;
          const auto jz_0_1_0 =          - Qdzdt * Wz_0_1_0;
          const auto jz_0_1_1 = jz_0_1_0 - Qdzdt * Wz_0_1_1;
          const auto jz_0_1_2 = jz_0_1_1 - Qdzdt * Wz_0_1_2;
          const auto jz_0_2_0 =          - Qdzdt * Wz_0_2_0;
          const auto jz_0_2_1 = jz_0_2_0 - Qdzdt * Wz_0_2_1;
          const auto jz_0_2_2 = jz_0_2_1 - Qdzdt * Wz_0_2_2;
          const auto jz_0_3_0 =          - Qdzdt * Wz_0_3_0;
          const auto jz_0_3_1 = jz_0_3_0 - Qdzdt * Wz_0_3_1;
          const auto jz_0_3_2 = jz_0_3_1 - Qdzdt * Wz_0_3_2;

          const auto jz_1_0_0 =          - Qdzdt * Wz_1_0_0;
          const auto jz_1_0_1 = jz_1_0_0 - Qdzdt * Wz_1_0_1;
          const auto jz_1_0_2 = jz_1_0_1 - Qdzdt * Wz_1_0_2;
          const auto jz_1_1_0 =          - Qdzdt * Wz_1_1_0;
          const auto jz_1_1_1 = jz_1_1_0 - Qdzdt * Wz_1_1_1;
          const auto jz_1_1_2 = jz_1_1_1 - Qdzdt * Wz_1_1_2;
          const auto jz_1_2_0 =          - Qdzdt * Wz_1_2_0;
          const auto jz_1_2_1 = jz_1_2_0 - Qdzdt * Wz_1_2_1;
          const auto jz_1_2_2 = jz_1_2_1 - Qdzdt * Wz_1_2_2;
          const auto jz_1_3_0 =          - Qdzdt * Wz_1_3_0;
          const auto jz_1_3_1 = jz_1_3_0 - Qdzdt * Wz_1_3_1;
          const auto jz_1_3_2 = jz_1_3_1 - Qdzdt * Wz_1_3_2;

          const auto jz_2_0_0 =          - Qdzdt * Wz_2_0_0;
          const auto jz_2_0_1 = jz_2_0_0 - Qdzdt * Wz_2_0_1;
          const auto jz_2_0_2 = jz_2_0_1 - Qdzdt * Wz_2_0_2;
          const auto jz_2_1_0 =          - Qdzdt * Wz_2_1_0;
          const auto jz_2_1_1 = jz_2_1_0 - Qdzdt * Wz_2_1_1;
          const auto jz_2_1_2 = jz_2_1_1 - Qdzdt * Wz_2_1_2;
          const auto jz_2_2_0 =          - Qdzdt * Wz_2_2_0;
          const auto jz_2_2_1 = jz_2_2_0 - Qdzdt * Wz_2_2_1;
          const auto jz_2_2_2 = jz_2_2_1 - Qdzdt * Wz_2_2_2;
          const auto jz_2_3_0 =          - Qdzdt * Wz_2_3_0;
          const auto jz_2_3_1 = jz_2_3_0 - Qdzdt * Wz_2_3_1;
          const auto jz_2_3_2 = jz_2_3_1 - Qdzdt * Wz_2_3_2;

          const auto jz_3_0_0 =          - Qdzdt * Wz_3_0_0;
          const auto jz_3_0_1 = jz_3_0_0 - Qdzdt * Wz_3_0_1;
          const auto jz_3_0_2 = jz_3_0_1 - Qdzdt * Wz_3_0_2;
          const auto jz_3_1_0 =          - Qdzdt * Wz_3_1_0;
          const auto jz_3_1_1 = jz_3_1_0 - Qdzdt * Wz_3_1_1;
          const auto jz_3_1_2 = jz_3_1_1 - Qdzdt * Wz_3_1_2;
          const auto jz_3_2_0 =          - Qdzdt * Wz_3_2_0;
          const auto jz_3_2_1 = jz_3_2_0 - Qdzdt * Wz_3_2_1;
          const auto jz_3_2_2 = jz_3_2_1 - Qdzdt * Wz_3_2_2;
          const auto jz_3_3_0 =          - Qdzdt * Wz_3_3_0;
          const auto jz_3_3_1 = jz_3_3_0 - Qdzdt * Wz_3_3_1;
          const auto jz_3_3_2 = jz_3_3_1 - Qdzdt * Wz_3_3_2;


          /*
            Current update
          */
          auto J_acc = J.access();
          
          J_acc(ix_min,     iy_min,     iz_min,     cur::jx1) += jx_0_0_0;
          J_acc(ix_min,     iy_min,     iz_min + 1, cur::jx1) += jx_0_0_1;
          J_acc(ix_min,     iy_min,     iz_min + 2, cur::jx1) += jx_0_0_2;
          J_acc(ix_min,     iy_min + 1, iz_min,     cur::jx1) += jx_0_1_0;
          J_acc(ix_min,     iy_min + 1, iz_min + 1, cur::jx1) += jx_0_1_1;
          J_acc(ix_min,     iy_min + 1, iz_min + 2, cur::jx1) += jx_0_1_2;
          J_acc(ix_min,     iy_min + 2, iz_min,     cur::jx1) += jx_0_2_0;
          J_acc(ix_min,     iy_min + 2, iz_min + 1, cur::jx1) += jx_0_2_1;
          J_acc(ix_min,     iy_min + 2, iz_min + 2, cur::jx1) += jx_0_2_2;
          J_acc(ix_min + 1, iy_min,     iz_min,     cur::jx1) += jx_1_0_0;
          J_acc(ix_min + 1, iy_min,     iz_min + 1, cur::jx1) += jx_1_0_1;
          J_acc(ix_min + 1, iy_min,     iz_min + 2, cur::jx1) += jx_1_0_2;
          J_acc(ix_min + 1, iy_min + 1, iz_min,     cur::jx1) += jx_1_1_0;
          J_acc(ix_min + 1, iy_min + 1, iz_min + 1, cur::jx1) += jx_1_1_1;
          J_acc(ix_min + 1, iy_min + 1, iz_min + 2, cur::jx1) += jx_1_1_2;
          J_acc(ix_min + 1, iy_min + 2, iz_min,     cur::jx1) += jx_1_2_0;
          J_acc(ix_min + 1, iy_min + 2, iz_min + 1, cur::jx1) += jx_1_2_1;
          J_acc(ix_min + 1, iy_min + 2, iz_min + 2, cur::jx1) += jx_1_2_2;
          
          if (update_x2)
          {
            J_acc(ix_min + 2, iy_min,     iz_min,     cur::jx1) += jx_2_0_0;
            J_acc(ix_min + 2, iy_min,     iz_min + 1, cur::jx1) += jx_2_0_1;
            J_acc(ix_min + 2, iy_min,     iz_min + 2, cur::jx1) += jx_2_0_2;
            J_acc(ix_min + 2, iy_min + 1, iz_min,     cur::jx1) += jx_2_1_0;
            J_acc(ix_min + 2, iy_min + 1, iz_min + 1, cur::jx1) += jx_2_1_1;
            J_acc(ix_min + 2, iy_min + 1, iz_min + 2, cur::jx1) += jx_2_1_2;
            J_acc(ix_min + 2, iy_min + 2, iz_min,     cur::jx1) += jx_2_2_0;
            J_acc(ix_min + 2, iy_min + 2, iz_min + 1, cur::jx1) += jx_2_2_1;
            J_acc(ix_min + 2, iy_min + 2, iz_min + 2, cur::jx1) += jx_2_2_2;

            if (update_y2)
            {
              J_acc(ix_min + 2, iy_min + 3, iz_min,     cur::jx1) += jx_2_3_0;
              J_acc(ix_min + 2, iy_min + 3, iz_min + 1, cur::jx1) += jx_2_3_1;
              J_acc(ix_min + 2, iy_min + 3, iz_min + 2, cur::jx1) += jx_2_3_2;
            }

            if (update_z2)
            {
              J_acc(ix_min + 2, iy_min,     iz_min + 3, cur::jx1) += jx_2_0_3;
              J_acc(ix_min + 2, iy_min + 1, iz_min + 3, cur::jx1) += jx_2_1_3;
              J_acc(ix_min + 2, iy_min + 2, iz_min + 3, cur::jx1) += jx_2_2_3;

              if (update_y2)
              {
                J_acc(ix_min + 2, iy_min + 3, iz_min + 3, cur::jx1) += jx_2_3_3;
              }
            }
          }
          //
          if (update_y2)
          {
            J_acc(ix_min,     iy_min + 3, iz_min,     cur::jx1) += jx_0_3_0;
            J_acc(ix_min,     iy_min + 3, iz_min + 1, cur::jx1) += jx_0_3_1;
            J_acc(ix_min,     iy_min + 3, iz_min + 2, cur::jx1) += jx_0_3_2;
            J_acc(ix_min + 1, iy_min + 3, iz_min,     cur::jx1) += jx_1_3_0;
            J_acc(ix_min + 1, iy_min + 3, iz_min + 1, cur::jx1) += jx_1_3_1;
            J_acc(ix_min + 1, iy_min + 3, iz_min + 2, cur::jx1) += jx_1_3_2;
          }

          if (update_z2)
          {
            J_acc(ix_min,     iy_min,     iz_min + 3, cur::jx1) += jx_0_0_3;
            J_acc(ix_min,     iy_min + 1, iz_min + 3, cur::jx1) += jx_0_1_3;
            J_acc(ix_min,     iy_min + 2, iz_min + 3, cur::jx1) += jx_0_2_3;
            J_acc(ix_min + 1, iy_min,     iz_min + 3, cur::jx1) += jx_1_0_3;
            J_acc(ix_min + 1, iy_min + 1, iz_min + 3, cur::jx1) += jx_1_1_3;
            J_acc(ix_min + 1, iy_min + 2, iz_min + 3, cur::jx1) += jx_1_2_3;

            if (update_y2)
            {
              J_acc(ix_min,     iy_min + 3, iz_min + 3, cur::jx1) += jx_0_3_3;
              J_acc(ix_min + 1, iy_min + 3, iz_min + 3, cur::jx1) += jx_1_3_3;
            }
          }


          /*
            y-component
          */
          J_acc(ix_min,     iy_min,     iz_min,     cur::jx2) += jy_0_0_0;
          J_acc(ix_min,     iy_min,     iz_min + 1, cur::jx2) += jy_0_0_1;
          J_acc(ix_min,     iy_min,     iz_min + 2, cur::jx2) += jy_0_0_2;
          J_acc(ix_min,     iy_min + 1, iz_min,     cur::jx2) += jy_0_1_0;
          J_acc(ix_min,     iy_min + 1, iz_min + 1, cur::jx2) += jy_0_1_1;
          J_acc(ix_min,     iy_min + 1, iz_min + 2, cur::jx2) += jy_0_1_2;
          J_acc(ix_min + 1, iy_min,     iz_min,     cur::jx2) += jy_1_0_0;
          J_acc(ix_min + 1, iy_min,     iz_min + 1, cur::jx2) += jy_1_0_1;
          J_acc(ix_min + 1, iy_min,     iz_min + 2, cur::jx2) += jy_1_0_2;
          J_acc(ix_min + 1, iy_min + 1, iz_min,     cur::jx2) += jy_1_1_0;
          J_acc(ix_min + 1, iy_min + 1, iz_min + 1, cur::jx2) += jy_1_1_1;
          J_acc(ix_min + 1, iy_min + 1, iz_min + 2, cur::jx2) += jy_1_1_2;
          J_acc(ix_min + 2, iy_min,     iz_min,     cur::jx2) += jy_2_0_0;
          J_acc(ix_min + 2, iy_min,     iz_min + 1, cur::jx2) += jy_2_0_1;
          J_acc(ix_min + 2, iy_min,     iz_min + 2, cur::jx2) += jy_2_0_2;
          J_acc(ix_min + 2, iy_min + 1, iz_min,     cur::jx2) += jy_2_1_0;
          J_acc(ix_min + 2, iy_min + 1, iz_min + 1, cur::jx2) += jy_2_1_1;
          J_acc(ix_min + 2, iy_min + 1, iz_min + 2, cur::jx2) += jy_2_1_2;
          
          if (update_x2)
          {
            J_acc(ix_min + 3, iy_min,     iz_min,     cur::jx2) += jy_3_0_0;
            J_acc(ix_min + 3, iy_min,     iz_min + 1, cur::jx2) += jy_3_0_1;
            J_acc(ix_min + 3, iy_min,     iz_min + 2, cur::jx2) += jy_3_0_2;
            J_acc(ix_min + 3, iy_min + 1, iz_min,     cur::jx2) += jy_3_1_0;
            J_acc(ix_min + 3, iy_min + 1, iz_min + 1, cur::jx2) += jy_3_1_1;
            J_acc(ix_min + 3, iy_min + 1, iz_min + 2, cur::jx2) += jy_3_1_2;
  
            if (update_z2)
            {
              J_acc(ix_min + 3, iy_min,     iz_min + 3, cur::jx2) += jy_3_0_3;
              J_acc(ix_min + 3, iy_min + 1, iz_min + 3, cur::jx2) += jy_3_1_3;
            }
          }

          if (update_y2)
          {
            J_acc(ix_min,     iy_min + 2, iz_min,     cur::jx2) += jy_0_2_0;
            J_acc(ix_min,     iy_min + 2, iz_min + 1, cur::jx2) += jy_0_2_1;
            J_acc(ix_min,     iy_min + 2, iz_min + 2, cur::jx2) += jy_0_2_2;
            J_acc(ix_min + 1, iy_min + 2, iz_min,     cur::jx2) += jy_1_2_0;
            J_acc(ix_min + 1, iy_min + 2, iz_min + 1, cur::jx2) += jy_1_2_1;
            J_acc(ix_min + 1, iy_min + 2, iz_min + 2, cur::jx2) += jy_1_2_2;
            J_acc(ix_min + 2, iy_min + 2, iz_min,     cur::jx2) += jy_2_2_0;
            J_acc(ix_min + 2, iy_min + 2, iz_min + 1, cur::jx2) += jy_2_2_1;
            J_acc(ix_min + 2, iy_min + 2, iz_min + 2, cur::jx2) += jy_2_2_2;

            if (update_x2)
            {
              J_acc(ix_min + 3, iy_min + 2, iz_min,     cur::jx2) += jy_3_2_0;
              J_acc(ix_min + 3, iy_min + 2, iz_min + 1, cur::jx2) += jy_3_2_1;
              J_acc(ix_min + 3, iy_min + 2, iz_min + 2, cur::jx2) += jy_3_2_2;

              if (update_z2)
              {
                J_acc(ix_min + 2, iy_min + 2, iz_min + 3, cur::jx2) += jy_2_2_3;
                J_acc(ix_min + 3, iy_min + 2, iz_min + 3, cur::jx2) += jy_3_2_3;
              }
            }

            if (update_z2)
            {
              J_acc(ix_min,     iy_min + 2, iz_min + 3, cur::jx2) += jy_0_2_3;
              J_acc(ix_min + 1, iy_min + 2, iz_min + 3, cur::jx2) += jy_1_2_3;
            }
          }

          if (update_z2)
          {
            J_acc(ix_min,     iy_min,     iz_min + 3, cur::jx2) += jy_0_0_3;
            J_acc(ix_min,     iy_min + 1, iz_min + 3, cur::jx2) += jy_0_1_3;
            J_acc(ix_min + 1, iy_min,     iz_min + 3, cur::jx2) += jy_1_0_3;
            J_acc(ix_min + 1, iy_min + 1, iz_min + 3, cur::jx2) += jy_1_1_3;
            J_acc(ix_min + 2, iy_min,     iz_min + 3, cur::jx2) += jy_2_0_3;
            J_acc(ix_min + 2, iy_min + 1, iz_min + 3, cur::jx2) += jy_2_1_3;
          }

          /*
            z-component
          */            
          J_acc(ix_min,     iy_min,     iz_min,     cur::jx3) += jz_0_0_0;
          J_acc(ix_min,     iy_min,     iz_min + 1, cur::jx3) += jz_0_0_1;
          J_acc(ix_min,     iy_min + 1, iz_min,     cur::jx3) += jz_0_1_0;
          J_acc(ix_min,     iy_min + 1, iz_min + 1, cur::jx3) += jz_0_1_1;
          J_acc(ix_min,     iy_min + 2, iz_min,     cur::jx3) += jz_0_2_0;
          J_acc(ix_min,     iy_min + 2, iz_min + 1, cur::jx3) += jz_0_2_1;
          J_acc(ix_min + 1, iy_min,     iz_min,     cur::jx3) += jz_1_0_0;
          J_acc(ix_min + 1, iy_min,     iz_min + 1, cur::jx3) += jz_1_0_1;
          J_acc(ix_min + 1, iy_min + 1, iz_min,     cur::jx3) += jz_1_1_0;
          J_acc(ix_min + 1, iy_min + 1, iz_min + 1, cur::jx3) += jz_1_1_1;
          J_acc(ix_min + 1, iy_min + 2, iz_min,     cur::jx3) += jz_1_2_0;
          J_acc(ix_min + 1, iy_min + 2, iz_min + 1, cur::jx3) += jz_1_2_1;
          J_acc(ix_min + 2, iy_min,     iz_min,     cur::jx3) += jz_2_0_0;
          J_acc(ix_min + 2, iy_min,     iz_min + 1, cur::jx3) += jz_2_0_1;
          J_acc(ix_min + 2, iy_min + 1, iz_min,     cur::jx3) += jz_2_1_0;
          J_acc(ix_min + 2, iy_min + 1, iz_min + 1, cur::jx3) += jz_2_1_1;
          J_acc(ix_min + 2, iy_min + 2, iz_min,     cur::jx3) += jz_2_2_0;
          J_acc(ix_min + 2, iy_min + 2, iz_min + 1, cur::jx3) += jz_2_2_1;

          if (update_x2)
          {
            J_acc(ix_min + 3, iy_min,     iz_min,     cur::jx3) += jz_3_0_0;
            J_acc(ix_min + 3, iy_min,     iz_min + 1, cur::jx3) += jz_3_0_1;
            J_acc(ix_min + 3, iy_min + 1, iz_min,     cur::jx3) += jz_3_1_0;
            J_acc(ix_min + 3, iy_min + 1, iz_min + 1, cur::jx3) += jz_3_1_1;
            J_acc(ix_min + 3, iy_min + 2, iz_min,     cur::jx3) += jz_3_2_0;
            J_acc(ix_min + 3, iy_min + 2, iz_min + 1, cur::jx3) += jz_3_2_1;
            J_acc(ix_min + 3, iy_min + 3, iz_min,     cur::jx3) += jz_3_3_0;
            J_acc(ix_min + 3, iy_min + 3, iz_min + 1, cur::jx3) += jz_3_3_1;
          }

          if (update_y2)
          {
            J_acc(ix_min,     iy_min + 3, iz_min,     cur::jx3) += jz_0_3_0;
            J_acc(ix_min,     iy_min + 3, iz_min + 1, cur::jx3) += jz_0_3_1;
            J_acc(ix_min + 1, iy_min + 3, iz_min,     cur::jx3) += jz_1_3_0;
            J_acc(ix_min + 1, iy_min + 3, iz_min + 1, cur::jx3) += jz_1_3_1;
            J_acc(ix_min + 2, iy_min + 3, iz_min,     cur::jx3) += jz_2_3_0;
            J_acc(ix_min + 2, iy_min + 3, iz_min + 1, cur::jx3) += jz_2_3_1;
          }

          if (update_z2)
          {
            J_acc(ix_min,     iy_min,     iz_min + 2, cur::jx3) += jz_0_0_2;
            J_acc(ix_min,     iy_min + 1, iz_min + 2, cur::jx3) += jz_0_1_2;
            J_acc(ix_min,     iy_min + 2, iz_min + 2, cur::jx3) += jz_0_2_2;
            J_acc(ix_min + 1, iy_min,     iz_min + 2, cur::jx3) += jz_1_0_2;
            J_acc(ix_min + 1, iy_min + 1, iz_min + 2, cur::jx3) += jz_1_1_2;
            J_acc(ix_min + 1, iy_min + 2, iz_min + 2, cur::jx3) += jz_1_2_2;
            J_acc(ix_min + 2, iy_min,     iz_min + 2, cur::jx3) += jz_2_0_2;          
            J_acc(ix_min + 2, iy_min + 1, iz_min + 2, cur::jx3) += jz_2_1_2;
            J_acc(ix_min + 2, iy_min + 2, iz_min + 2, cur::jx3) += jz_2_2_2;

            if (update_x2)
            {
              J_acc(ix_min + 3, iy_min,     iz_min + 2, cur::jx3) += jz_3_0_2;
              J_acc(ix_min + 3, iy_min + 1, iz_min + 2, cur::jx3) += jz_3_1_2;
              J_acc(ix_min + 3, iy_min + 2, iz_min + 2, cur::jx3) += jz_3_2_2;

              if (update_y2)
              {
                J_acc(ix_min + 3, iy_min + 3, iz_min + 2, cur::jx3) += jz_3_3_2;
              }
            }

            if (update_y2)
            {
              J_acc(ix_min,     iy_min + 3, iz_min + 2, cur::jx3) += jz_0_3_2;
              J_acc(ix_min + 1, iy_min + 3, iz_min + 2, cur::jx3) += jz_1_3_2;
              J_acc(ix_min + 2, iy_min + 3, iz_min + 2, cur::jx3) += jz_2_3_2;
            }
          }
          // clang-format on
        } // dimension

      } else if constexpr (O == 3u) {
        /*
          Higher order charge conserving current deposition based on
          Esirkepov (2001) https://ui.adsabs.harvard.edu/abs/2001CoPhC.135..144E/abstract

          We need to define the follwowing variable:
          - Shape functions in spatial directions for the particle position
            before and after the current timestep.
            S0_*, S1_*
          - Density composition matrix
            Wx_*, Wy_*, Wz_*
        */

        /*
            x - direction
        */

        // shape function at previous timestep
        real_t   S0x_0, S0x_1, S0x_2, S0x_3, S0x_4;
        // shape function at current timestep
        real_t   S1x_0, S1x_1, S1x_2, S1x_3, S1x_4;
        // indices of the shape function
        ncells_t ix_min;
        bool     update_x3;
        // find indices and define shape function
        // clang-format off
        shape_function_3rd(S0x_0, S0x_1, S0x_2, S0x_3, S0x_4,
                           S1x_0, S1x_1, S1x_2, S1x_3, S1x_4,
                           ix_min, update_x3,
                           i1(p), dx1(p),
                           i1_prev(p), dx1_prev(p));
        // clang-format on

        if constexpr (D == Dim::_1D) {
          // ToDo
        } else if constexpr (D == Dim::_2D) {

          /*
            y - direction
          */

          // shape function at previous timestep
          real_t   S0y_0, S0y_1, S0y_2, S0y_3, S0y_4;
          // shape function at current timestep
          real_t   S1y_0, S1y_1, S1y_2, S1y_3, S1y_4;
          // indices of the shape function
          ncells_t iy_min;
          bool     update_y3;
          // find indices and define shape function
          // clang-format off
          shape_function_3rd(S0y_0, S0y_1, S0y_2, S0y_3, S0y_4,
                             S1y_0, S1y_1, S1y_2, S1y_3, S1y_4,
                             iy_min, update_y3,
                             i2(p), dx2(p),
                             i2_prev(p), dx2_prev(p));
          // clang-format on

          // Esirkepov 2001, Eq. 38
          /*
              x - component
          */
          // Calculate weight function - unrolled
          const auto Wx_0_0 = HALF * (S1x_0 - S0x_0) * (S0y_0 + S1y_0);
          const auto Wx_0_1 = HALF * (S1x_0 - S0x_0) * (S0y_1 + S1y_1);
          const auto Wx_0_2 = HALF * (S1x_0 - S0x_0) * (S0y_2 + S1y_2);
          const auto Wx_0_3 = HALF * (S1x_0 - S0x_0) * (S0y_3 + S1y_3);
          const auto Wx_0_4 = HALF * (S1x_0 - S0x_0) * (S0y_4 + S1y_4);

          const auto Wx_1_0 = HALF * (S1x_1 - S0x_1) * (S0y_0 + S1y_0);
          const auto Wx_1_1 = HALF * (S1x_1 - S0x_1) * (S0y_1 + S1y_1);
          const auto Wx_1_2 = HALF * (S1x_1 - S0x_1) * (S0y_2 + S1y_2);
          const auto Wx_1_3 = HALF * (S1x_1 - S0x_1) * (S0y_3 + S1y_3);
          const auto Wx_1_4 = HALF * (S1x_1 - S0x_1) * (S0y_4 + S1y_4);

          const auto Wx_2_0 = HALF * (S1x_2 - S0x_2) * (S0y_0 + S1y_0);
          const auto Wx_2_1 = HALF * (S1x_2 - S0x_2) * (S0y_1 + S1y_1);
          const auto Wx_2_2 = HALF * (S1x_2 - S0x_2) * (S0y_2 + S1y_2);
          const auto Wx_2_3 = HALF * (S1x_2 - S0x_2) * (S0y_3 + S1y_3);
          const auto Wx_2_4 = HALF * (S1x_2 - S0x_2) * (S0y_4 + S1y_4);

          const auto Wx_3_0 = HALF * (S1x_3 - S0x_3) * (S0y_0 + S1y_0);
          const auto Wx_3_1 = HALF * (S1x_3 - S0x_3) * (S0y_1 + S1y_1);
          const auto Wx_3_2 = HALF * (S1x_3 - S0x_3) * (S0y_2 + S1y_2);
          const auto Wx_3_3 = HALF * (S1x_3 - S0x_3) * (S0y_3 + S1y_3);
          const auto Wx_3_4 = HALF * (S1x_3 - S0x_3) * (S0y_4 + S1y_4);

          // Unrolled calculations for Wy
          const auto Wy_0_0 = HALF * (S1x_0 + S0x_0) * (S1y_0 - S0y_0);
          const auto Wy_0_1 = HALF * (S1x_0 + S0x_0) * (S1y_1 - S0y_1);
          const auto Wy_0_2 = HALF * (S1x_0 + S0x_0) * (S1y_2 - S0y_2);
          const auto Wy_0_3 = HALF * (S1x_0 + S0x_0) * (S1y_3 - S0y_3);

          const auto Wy_1_0 = HALF * (S1x_1 + S0x_1) * (S1y_0 - S0y_0);
          const auto Wy_1_1 = HALF * (S1x_1 + S0x_1) * (S1y_1 - S0y_1);
          const auto Wy_1_2 = HALF * (S1x_1 + S0x_1) * (S1y_2 - S0y_2);
          const auto Wy_1_3 = HALF * (S1x_1 + S0x_1) * (S1y_3 - S0y_3);

          const auto Wy_2_0 = HALF * (S1x_2 + S0x_2) * (S1y_0 - S0y_0);
          const auto Wy_2_1 = HALF * (S1x_2 + S0x_2) * (S1y_1 - S0y_1);
          const auto Wy_2_2 = HALF * (S1x_2 + S0x_2) * (S1y_2 - S0y_2);
          const auto Wy_2_3 = HALF * (S1x_2 + S0x_2) * (S1y_3 - S0y_3);

          const auto Wy_3_0 = HALF * (S1x_3 + S0x_3) * (S1y_0 - S0y_0);
          const auto Wy_3_1 = HALF * (S1x_3 + S0x_3) * (S1y_1 - S0y_1);
          const auto Wy_3_2 = HALF * (S1x_3 + S0x_3) * (S1y_2 - S0y_2);
          const auto Wy_3_3 = HALF * (S1x_3 + S0x_3) * (S1y_3 - S0y_3);

          const auto Wy_4_0 = HALF * (S1x_4 + S0x_4) * (S1y_0 - S0y_0);
          const auto Wy_4_1 = HALF * (S1x_4 + S0x_4) * (S1y_1 - S0y_1);
          const auto Wy_4_2 = HALF * (S1x_4 + S0x_4) * (S1y_2 - S0y_2);
          const auto Wy_4_3 = HALF * (S1x_4 + S0x_4) * (S1y_3 - S0y_3);

          // Unrolled calculations for Wz
          const auto Wz_0_0 = THIRD * (S1y_0 * (HALF * S0x_0 + S1x_0) +
                                       S0y_0 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_1 = THIRD * (S1y_1 * (HALF * S0x_0 + S1x_0) +
                                       S0y_1 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_2 = THIRD * (S1y_2 * (HALF * S0x_0 + S1x_0) +
                                       S0y_2 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_3 = THIRD * (S1y_3 * (HALF * S0x_0 + S1x_0) +
                                       S0y_3 * (HALF * S1x_0 + S0x_0));
          const auto Wz_0_4 = THIRD * (S1y_4 * (HALF * S0x_0 + S1x_0) +
                                       S0y_4 * (HALF * S1x_0 + S0x_0));

          const auto Wz_1_0 = THIRD * (S1y_0 * (HALF * S0x_1 + S1x_1) +
                                       S0y_0 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_1 = THIRD * (S1y_1 * (HALF * S0x_1 + S1x_1) +
                                       S0y_1 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_2 = THIRD * (S1y_2 * (HALF * S0x_1 + S1x_1) +
                                       S0y_2 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_3 = THIRD * (S1y_3 * (HALF * S0x_1 + S1x_1) +
                                       S0y_3 * (HALF * S1x_1 + S0x_1));
          const auto Wz_1_4 = THIRD * (S1y_4 * (HALF * S0x_1 + S1x_1) +
                                       S0y_4 * (HALF * S1x_1 + S0x_1));

          const auto Wz_2_0 = THIRD * (S1y_0 * (HALF * S0x_2 + S1x_2) +
                                       S0y_0 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_1 = THIRD * (S1y_1 * (HALF * S0x_2 + S1x_2) +
                                       S0y_1 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_2 = THIRD * (S1y_2 * (HALF * S0x_2 + S1x_2) +
                                       S0y_2 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_3 = THIRD * (S1y_3 * (HALF * S0x_2 + S1x_2) +
                                       S0y_3 * (HALF * S1x_2 + S0x_2));
          const auto Wz_2_4 = THIRD * (S1y_4 * (HALF * S0x_2 + S1x_2) +
                                       S0y_4 * (HALF * S1x_2 + S0x_2));

          const auto Wz_3_0 = THIRD * (S1y_0 * (HALF * S0x_3 + S1x_3) +
                                       S0y_0 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_1 = THIRD * (S1y_1 * (HALF * S0x_3 + S1x_3) +
                                       S0y_1 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_2 = THIRD * (S1y_2 * (HALF * S0x_3 + S1x_3) +
                                       S0y_2 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_3 = THIRD * (S1y_3 * (HALF * S0x_3 + S1x_3) +
                                       S0y_3 * (HALF * S1x_3 + S0x_3));
          const auto Wz_3_4 = THIRD * (S1y_4 * (HALF * S0x_3 + S1x_3) +
                                       S0y_4 * (HALF * S1x_3 + S0x_3));

          const auto Wz_4_0 = THIRD * (S1y_0 * (HALF * S0x_4 + S1x_4) +
                                       S0y_0 * (HALF * S1x_4 + S0x_4));
          const auto Wz_4_1 = THIRD * (S1y_1 * (HALF * S0x_4 + S1x_4) +
                                       S0y_1 * (HALF * S1x_4 + S0x_4));
          const auto Wz_4_2 = THIRD * (S1y_2 * (HALF * S0x_4 + S1x_4) +
                                       S0y_2 * (HALF * S1x_4 + S0x_4));
          const auto Wz_4_3 = THIRD * (S1y_3 * (HALF * S0x_4 + S1x_4) +
                                       S0y_3 * (HALF * S1x_4 + S0x_4));
          const auto Wz_4_4 = THIRD * (S1y_4 * (HALF * S0x_4 + S1x_4) +
                                       S0y_4 * (HALF * S1x_4 + S0x_4));

          const real_t Qdxdt = coeff * inv_dt;
          const real_t Qdydt = coeff * inv_dt;
          const real_t QVz   = coeff * inv_dt * vp[2];

          // Esirkepov - Eq. 39
          // x-component
          const auto jx_0_0 = -Qdxdt * Wx_0_0;
          const auto jx_1_0 = jx_0_0 - Qdxdt * Wx_1_0;
          const auto jx_2_0 = jx_1_0 - Qdxdt * Wx_2_0;
          const auto jx_3_0 = jx_2_0 - Qdxdt * Wx_3_0;

          const auto jx_0_1 = -Qdxdt * Wx_0_1;
          const auto jx_1_1 = jx_0_1 - Qdxdt * Wx_1_1;
          const auto jx_2_1 = jx_1_1 - Qdxdt * Wx_2_1;
          const auto jx_3_1 = jx_2_1 - Qdxdt * Wx_3_1;

          const auto jx_0_2 = -Qdxdt * Wx_0_2;
          const auto jx_1_2 = jx_0_2 - Qdxdt * Wx_1_2;
          const auto jx_2_2 = jx_1_2 - Qdxdt * Wx_2_2;
          const auto jx_3_2 = jx_2_2 - Qdxdt * Wx_3_2;

          const auto jx_0_3 = -Qdxdt * Wx_0_3;
          const auto jx_1_3 = jx_0_3 - Qdxdt * Wx_1_3;
          const auto jx_2_3 = jx_1_3 - Qdxdt * Wx_2_3;
          const auto jx_3_3 = jx_2_3 - Qdxdt * Wx_3_3;

          const auto jx_0_4 = -Qdxdt * Wx_0_4;
          const auto jx_1_4 = jx_0_4 - Qdxdt * Wx_1_4;
          const auto jx_2_4 = jx_1_4 - Qdxdt * Wx_2_4;
          const auto jx_3_4 = jx_2_4 - Qdxdt * Wx_3_4;

          // y-component
          const auto jy_0_0 = -Qdydt * Wy_0_0;
          const auto jy_0_1 = jy_0_0 - Qdydt * Wy_0_1;
          const auto jy_0_2 = jy_0_1 - Qdydt * Wy_0_2;
          const auto jy_0_3 = jy_0_2 - Qdydt * Wy_0_3;

          const auto jy_1_0 = -Qdydt * Wy_1_0;
          const auto jy_1_1 = jy_1_0 - Qdydt * Wy_1_1;
          const auto jy_1_2 = jy_1_1 - Qdydt * Wy_1_2;
          const auto jy_1_3 = jy_1_2 - Qdydt * Wy_1_3;

          const auto jy_2_0 = -Qdydt * Wy_2_0;
          const auto jy_2_1 = jy_2_0 - Qdydt * Wy_2_1;
          const auto jy_2_2 = jy_2_1 - Qdydt * Wy_2_2;
          const auto jy_2_3 = jy_2_2 - Qdydt * Wy_2_3;

          const auto jy_3_0 = -Qdydt * Wy_3_0;
          const auto jy_3_1 = jy_3_0 - Qdydt * Wy_3_1;
          const auto jy_3_2 = jy_3_1 - Qdydt * Wy_3_2;
          const auto jy_3_3 = jy_3_2 - Qdydt * Wy_3_3;

          const auto jy_4_0 = -Qdydt * Wy_4_0;
          const auto jy_4_1 = jy_4_0 - Qdydt * Wy_4_1;
          const auto jy_4_2 = jy_4_1 - Qdydt * Wy_4_2;
          const auto jy_4_3 = jy_4_2 - Qdydt * Wy_4_3;

          /*
            Current update
          */
          auto J_acc = J.access();

          /*
              x - component
          */
          J_acc(ix_min, iy_min, cur::jx1)     += jx_0_0;
          J_acc(ix_min, iy_min + 1, cur::jx1) += jx_0_1;
          J_acc(ix_min, iy_min + 2, cur::jx1) += jx_0_2;
          J_acc(ix_min, iy_min + 3, cur::jx1) += jx_0_3;

          J_acc(ix_min + 1, iy_min, cur::jx1)     += jx_1_0;
          J_acc(ix_min + 1, iy_min + 1, cur::jx1) += jx_1_1;
          J_acc(ix_min + 1, iy_min + 2, cur::jx1) += jx_1_2;
          J_acc(ix_min + 1, iy_min + 3, cur::jx1) += jx_1_3;

          J_acc(ix_min + 2, iy_min, cur::jx1)     += jx_2_0;
          J_acc(ix_min + 2, iy_min + 1, cur::jx1) += jx_2_1;
          J_acc(ix_min + 2, iy_min + 2, cur::jx1) += jx_2_2;
          J_acc(ix_min + 2, iy_min + 3, cur::jx1) += jx_2_3;

          if (update_x3) {
            J_acc(ix_min + 3, iy_min, cur::jx1)     += jx_3_0;
            J_acc(ix_min + 3, iy_min + 1, cur::jx1) += jx_3_1;
            J_acc(ix_min + 3, iy_min + 2, cur::jx1) += jx_3_2;
            J_acc(ix_min + 3, iy_min + 3, cur::jx1) += jx_3_3;
          }

          if (update_y3) {
            J_acc(ix_min, iy_min + 4, cur::jx1)     += jx_0_4;
            J_acc(ix_min + 1, iy_min + 4, cur::jx1) += jx_1_4;
            J_acc(ix_min + 2, iy_min + 4, cur::jx1) += jx_2_4;
          }

          if (update_x3 && update_y3) {
            J_acc(ix_min + 3, iy_min + 4, cur::jx1) += jx_3_4;
          }

          /*
              y - component
          */
          J_acc(ix_min, iy_min, cur::jx2)     += jy_0_0;
          J_acc(ix_min + 1, iy_min, cur::jx2) += jy_1_0;
          J_acc(ix_min + 2, iy_min, cur::jx2) += jy_2_0;
          J_acc(ix_min + 3, iy_min, cur::jx2) += jy_3_0;

          J_acc(ix_min, iy_min + 1, cur::jx2)     += jy_0_1;
          J_acc(ix_min + 1, iy_min + 1, cur::jx2) += jy_1_1;
          J_acc(ix_min + 2, iy_min + 1, cur::jx2) += jy_2_1;
          J_acc(ix_min + 3, iy_min + 1, cur::jx2) += jy_3_1;

          J_acc(ix_min, iy_min + 2, cur::jx2)     += jy_0_2;
          J_acc(ix_min + 1, iy_min + 2, cur::jx2) += jy_1_2;
          J_acc(ix_min + 2, iy_min + 2, cur::jx2) += jy_2_2;
          J_acc(ix_min + 3, iy_min + 2, cur::jx2) += jy_3_2;

          if (update_x3) {
            J_acc(ix_min + 4, iy_min, cur::jx2)     += jy_4_0;
            J_acc(ix_min + 4, iy_min + 1, cur::jx2) += jy_4_1;
            J_acc(ix_min + 4, iy_min + 2, cur::jx2) += jy_4_2;
          }

          if (update_y3) {
            J_acc(ix_min, iy_min + 3, cur::jx2)     += jy_0_3;
            J_acc(ix_min + 1, iy_min + 3, cur::jx2) += jy_1_3;
            J_acc(ix_min + 2, iy_min + 3, cur::jx2) += jy_2_3;
            J_acc(ix_min + 3, iy_min + 3, cur::jx2) += jy_3_3;
          }

          if (update_x3 && update_y3) {
            J_acc(ix_min + 4, iy_min + 3, cur::jx2) += jy_4_3;
          }
          /*
              z - component, simulated direction
          */
          J_acc(ix_min, iy_min, cur::jx3)     += QVz * Wz_0_0;
          J_acc(ix_min, iy_min + 1, cur::jx3) += QVz * Wz_0_1;
          J_acc(ix_min, iy_min + 2, cur::jx3) += QVz * Wz_0_2;
          J_acc(ix_min, iy_min + 3, cur::jx3) += QVz * Wz_0_3;

          J_acc(ix_min + 1, iy_min, cur::jx3)     += QVz * Wz_1_0;
          J_acc(ix_min + 1, iy_min + 1, cur::jx3) += QVz * Wz_1_1;
          J_acc(ix_min + 1, iy_min + 2, cur::jx3) += QVz * Wz_1_2;
          J_acc(ix_min + 1, iy_min + 3, cur::jx3) += QVz * Wz_1_3;

          J_acc(ix_min + 2, iy_min, cur::jx3)     += QVz * Wz_2_0;
          J_acc(ix_min + 2, iy_min + 1, cur::jx3) += QVz * Wz_2_1;
          J_acc(ix_min + 2, iy_min + 2, cur::jx3) += QVz * Wz_2_2;
          J_acc(ix_min + 2, iy_min + 3, cur::jx3) += QVz * Wz_2_3;

          J_acc(ix_min + 3, iy_min, cur::jx3)     += QVz * Wz_3_0;
          J_acc(ix_min + 3, iy_min + 1, cur::jx3) += QVz * Wz_3_1;
          J_acc(ix_min + 3, iy_min + 2, cur::jx3) += QVz * Wz_3_2;
          J_acc(ix_min + 3, iy_min + 3, cur::jx3) += QVz * Wz_3_3;

          if (update_x3) {
            J_acc(ix_min + 4, iy_min, cur::jx3)     += QVz * Wz_4_0;
            J_acc(ix_min + 4, iy_min + 1, cur::jx3) += QVz * Wz_4_1;
            J_acc(ix_min + 4, iy_min + 2, cur::jx3) += QVz * Wz_4_2;
            J_acc(ix_min + 4, iy_min + 3, cur::jx3) += QVz * Wz_4_3;
          }

          if (update_y3) {
            J_acc(ix_min, iy_min + 4, cur::jx3)     += QVz * Wz_0_4;
            J_acc(ix_min + 1, iy_min + 4, cur::jx3) += QVz * Wz_1_4;
            J_acc(ix_min + 2, iy_min + 4, cur::jx3) += QVz * Wz_2_4;
            J_acc(ix_min + 3, iy_min + 4, cur::jx3) += QVz * Wz_3_4;
          }
          if (update_x3 && update_y3) {
            J_acc(ix_min + 4, iy_min + 4, cur::jx3) += QVz * Wz_4_4;
          }

        } // dim -> ToDo: 3D!

      } else if constexpr (O > 3u) {

        // shape function in dim1 -> always required
        real_t   S0x[O + 2], S1x[O + 2];
        // indices of the shape function
        ncells_t ix_min;

        // ToDo: Call shape function

        if constexpr (D == Dim::_1D) {
          // ToDo
        } else if constexpr (D == Dim::_2D) {

          // shape function in dim2
          real_t   S0y[O + 2], S1y[O + 2];
          // indices of the shape function
          ncells_t iy_min;

          // ToDo: Call shape function

          // define weight tensors
          real_t Wx[O + 1][O + 1];
          real_t Wy[O + 1][O + 1];
          real_t Wz[O + 1][O + 1];

// Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              // Esirkepov 2001, Eq. 38
              Wx[i][j] = (S1x[i] - S0x[i]) * (S0y[j] + HALF * (S1y[j] - S0y[j]));

              Wy[i][j] = (S1y[i] - S0y[i]) * (S0y[j] + HALF * (S1x[j] - S0x[j]));

              Wz[i][j] = S0x[i] * S0y[j] + HALF * (S1x[i] - S1x[i]) * S0y[j] +
                         HALF * S0x[i] * (S1y[j] - S0y[j]) +
                         THIRD * (S1x[i] - S0x[i]) * (S1y[j] - S0y[j]);
            }
          }

          // contribution within the shape function stencil
          real_t jx[O + 2][O + 2], jy[O + 2][O + 2], jz[O + 2][O + 2];

          // prefactors to j update
          const real_t Qdxdt = coeff * inv_dt;
          const real_t Qdydt = coeff * inv_dt;
          const real_t QVz   = coeff * inv_dt * vp[2];

          // Calculate current contribution

          // jx
#pragma unroll
          for (int j = 0; j < O + 2; ++j) {
            jx[0][j] = -Qdxdt * Wx[0][j];
          }

#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jx[i][j] = jx[i - 1][j] - Qdxdt * Wx[i][j];
            }
          }

          // jy
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
            jy[i][0] = -Qdydt * Wy[i][0];
          }

#pragma unroll
          for (int j = 1; j < O + 2; ++j) {
#pragma unroll
            for (int i = 0; i < O + 2; ++i) {
              jy[i][j] = jy[i][j - 1] - Qdydt * Wy[i][j];
            }
          }

          // jz
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jz[i][j] = QVz * Wz[i][j];
            }
          }

          /*
              Current update
            */
          auto J_acc = J.access();

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              J_acc(ix_min + i, iy_min + j, cur::jx1) += jx[i][j];
              J_acc(ix_min + i, iy_min + j, cur::jx2) += jy[i][j];
              J_acc(ix_min + i, iy_min + j, cur::jx3) += jz[i][j];
            }
          }

        } else if constexpr (D == Dim::_3D) {
          // shape function in dim2
          real_t   S0y[O + 2], S1y[O + 2];
          // indices of the shape function
          ncells_t iy_min;

          // ToDo: Call shape function

          // shape function in dim3
          real_t   S0z[O + 2], S1z[O + 2];
          // indices of the shape function
          ncells_t iz_min;

          // ToDo: Call shape function

          // define weight tensors
          real_t Wx[O + 1][O + 1][O + 1];
          real_t Wy[O + 1][O + 1][O + 1];
          real_t Wz[O + 1][O + 1][O + 1];

// Calculate weight function
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                // Esirkepov 2001, Eq. 31
                Wx[i][j][k] = THIRD * (S1x[i] - S0x[i]) *
                              ((S0y[j] * S0z[k] + S1y[j] * S1z[k]) +
                               HALF * (S0z[k] * S1y[j] + S0y[j] * S1z[k]));

                Wy[i][j][k] = THIRD * (S1y[j] - S0y[j]) *
                              (S0x[i] * S0z[k] + S1x[i] * S1z[k] +
                               HALF * (S0z[k] * S1x[i] + S0x[i] * S1z[k]));

                Wz[i][j][k] = THIRD * (S1z[k] - S0z[k]) *
                              (S0x[i] * S0y[j] + S1x[i] * S1y[j] +
                               HALF * (S0x[i] * S1y[j] + S0y[j] * S1x[i]));
              }
            }
          }

          // contribution within the shape function stencil
          real_t jx[O + 2][O + 2][O + 2], jy[O + 2][O + 2][O + 2],
            jz[O + 2][O + 2][O + 2];

          // prefactors to j update
          const real_t Qdxdt = coeff * inv_dt;
          const real_t Qdydt = coeff * inv_dt;
          const real_t Qdzdt = coeff * inv_dt;

          // Calculate current contribution

          // jx
#pragma unroll
          for (int j = 0; j < O + 2; ++j) {
#pragma unroll
            for (int k = 0; k < O + 2; ++k) {
              jx[0][j][k] = -Qdxdt * Wx[0][j][k];
            }
          }

#pragma unroll
          for (int i = 1; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; j < O + 2; ++k) {
                jx[i][j][k] = jx[i - 1][j][k] - Qdxdt * Wx[i][j][k];
              }
            }
          }

          // jy
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int k = 0; k < O + 2; ++k) {
              jy[i][0][k] = -Qdydt * Wy[i][0][k];
            }
          }

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 1; j < O + 2; ++j) {
#pragma unroll
              for (int k = 0; k < O + 2; ++k) {
                jy[i][j][k] = jy[i][j - 1][k] - Qdydt * Wy[i][j][k];
              }
            }
          }

          // jz
#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
              jy[i][j][0] = -Qdydt * Wy[i][j][0];
            }
          }

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 1; k < O + 2; ++k) {
                jz[i][j][k] = jz[i][j][k - 1] - Qdzdt * Wz[i][j][k];
              }
            }
          }

          /*
            Current update
          */
          auto J_acc = J.access();

#pragma unroll
          for (int i = 0; i < O + 2; ++i) {
#pragma unroll
            for (int j = 0; j < O + 2; ++j) {
#pragma unroll
              for (int k = 1; k < O + 2; ++k) {
                J_acc(ix_min + i, iy_min + j, iz_min, cur::jx1) += jx[i][j][k];
                J_acc(ix_min + i, iy_min + j, iz_min, cur::jx2) += jy[i][j][k];
                J_acc(ix_min + i, iy_min + j, iz_min, cur::jx3) += jz[i][j][k];
              }
            }
          }
        }
        
        } else { // order
          raise::KernelError(HERE, "Unsupported interpolation order");
        }
      }
    };
  } // namespace kernel

#undef i_di_to_Xi

#endif // KERNELS_CURRENTS_DEPOSIT_HPP
