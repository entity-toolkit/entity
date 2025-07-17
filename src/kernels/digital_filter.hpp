/**
 * @file kernels/digital_filter.hpp
 * @brief Algorithms for covariant digital filtering
 * @implements
 *   - kernel::DigitalFilter_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_DIGITAL_FILTER_HPP
#define KERNELS_DIGITAL_FILTER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#define FILTER2D_IN_I1(ARR, COMP, I, J)                                        \
  INV_2*(ARR)((I), (J), (COMP)) +                                              \
    INV_4*((ARR)((I) - 1, (J), (COMP)) + (ARR)((I) + 1, (J), (COMP)))

#define FILTER2D_IN_I2(ARR, COMP, I, J)                                        \
  INV_2*(ARR)((I), (J), (COMP)) +                                              \
    INV_4*((ARR)((I), (J) - 1, (COMP)) + (ARR)((I), (J) + 1, (COMP)))

#define FILTER3D_IN_I1_I2(ARR, COMP, I, J, K)                                     \
  INV_4*(ARR)(I, J, K, (COMP)) +                                                  \
    INV_8*((ARR)((I) - 1, (J), (K), (COMP)) + (ARR)((I) + 1, (J), (K), (COMP)) +  \
           (ARR)((I), (J) - 1, (K), (COMP)) + (ARR)((I), (J) + 1, (K), (COMP))) + \
    INV_16*((ARR)((I) - 1, (J) - 1, (K), (COMP)) +                                \
            (ARR)((I) + 1, (J) + 1, (K), (COMP)) +                                \
            (ARR)((I) - 1, (J) + 1, (K), (COMP)) +                                \
            (ARR)((I) + 1, (J) - 1, (K), (COMP)))

#define FILTER3D_IN_I2_I3(ARR, COMP, I, J, K)                                     \
  INV_4*(ARR)(I, J, K, (COMP)) +                                                  \
    INV_8*((ARR)((I), (J) - 1, (K), (COMP)) + (ARR)((I), (J) + 1, (K), (COMP)) +  \
           (ARR)((I), (J), (K) - 1, (COMP)) + (ARR)((I), (J), (K) + 1, (COMP))) + \
    INV_16*((ARR)((I), (J) - 1, (K) - 1, (COMP)) +                                \
            (ARR)((I), (J) + 1, (K) + 1, (COMP)) +                                \
            (ARR)((I), (J) - 1, (K) + 1, (COMP)) +                                \
            (ARR)((I), (J) + 1, (K) - 1, (COMP)))

#define FILTER3D_IN_I1_I3(ARR, COMP, I, J, K)                                     \
  INV_4*(ARR)(I, J, K, (COMP)) +                                                  \
    INV_8*((ARR)((I) - 1, (J), (K), (COMP)) + (ARR)((I) + 1, (J), (K), (COMP)) +  \
           (ARR)((I), (J), (K) - 1, (COMP)) + (ARR)((I), (J), (K) + 1, (COMP))) + \
    INV_16*((ARR)((I) - 1, (J), (K) - 1, (COMP)) +                                \
            (ARR)((I) + 1, (J), (K) + 1, (COMP)) +                                \
            (ARR)((I) - 1, (J), (K) + 1, (COMP)) +                                \
            (ARR)((I) + 1, (J), (K) - 1, (COMP)))

namespace kernel {
  using namespace ntt;

  template <Dimension D, Coord::type C>
  class DigitalFilter_kernel {
    ndfield_t<D, 3>       array;
    const ndfield_t<D, 3> buffer;
    const bool            is_axis_i2min, is_axis_i2max;
    const bool            is_conductor_i1min, is_conductor_i1max;
    const bool            is_conductor_i2min, is_conductor_i2max;
    const bool            is_conductor_i3min, is_conductor_i3max;
    static constexpr auto i1_min = N_GHOSTS, i2_min = N_GHOSTS, i3_min = N_GHOSTS;
    const ncells_t i1_max, i2_max, i3_max;

    // @TODO: Current implementation might have issues
    // ... at the corners between two conductors

  public:
    DigitalFilter_kernel(ndfield_t<D, 3>&       array,
                         const ndfield_t<D, 3>& buffer,
                         const ncells_t (&size_)[D],
                         const boundaries_t<FldsBC>& boundaries)
      : array { array }
      , buffer { buffer }
      , is_axis_i2min { (D == Dim::_2D) and (boundaries[1].first == FldsBC::AXIS) }
      , is_axis_i2max { (D == Dim::_2D) and (boundaries[1].second == FldsBC::AXIS) }
      , is_conductor_i1min { boundaries[0].first == FldsBC::CONDUCTOR }
      , is_conductor_i1max { boundaries[0].second == FldsBC::CONDUCTOR }
      , is_conductor_i2min { (short)D > 1
                               ? (boundaries[1].first == FldsBC::CONDUCTOR)
                               : false }
      , is_conductor_i2max { (short)D > 1
                               ? (boundaries[1].second == FldsBC::CONDUCTOR)
                               : false }
      , is_conductor_i3min { (short)D > 2
                               ? (boundaries[2].first == FldsBC::CONDUCTOR)
                               : false }
      , is_conductor_i3max { (short)D > 2
                               ? (boundaries[2].second == FldsBC::CONDUCTOR)
                               : false }
      , i1_max { size_[0] + N_GHOSTS }
      , i2_max { (short)D > 1 ? (size_[1] + N_GHOSTS) : 0 }
      , i3_max { (short)D > 2 ? (size_[2] + N_GHOSTS) : 0 } {}

    Inline void operator()(index_t i1) const {
      if constexpr ((D == Dim::_1D) && (C == Coord::Cart)) {
        if ((is_conductor_i1min and i1 == i1_min) or
            (is_conductor_i1max and i1 == i1_max - 1)) {
          const auto i1side   = is_conductor_i1min ? (i1 + 1) : (i1 - 1);
          array(i1, cur::jx1) = (THREE * INV_4) * buffer(i1, cur::jx1) +
                                (INV_4)*buffer(i1side, cur::jx1);
        } else if ((is_conductor_i1min and i1 == i1_min + 1) or
                   (is_conductor_i1max and i1 == i1_max - 2)) {
          const auto i1side   = is_conductor_i1min ? (i1 + 1) : (i1 - 1);
          array(i1, cur::jx1) = INV_2 * buffer(i1, cur::jx1) +
                                INV_4 * (buffer(i1 - 1, cur::jx1) +
                                         buffer(i1 + 1, cur::jx1));
          array(i1, cur::jx2) = (INV_2)*buffer(i1, cur::jx2) +
                                (INV_4)*buffer(i1side, cur::jx2);
          array(i1, cur::jx3) = (INV_2)*buffer(i1, cur::jx3) +
                                (INV_4)*buffer(i1side, cur::jx3);
        } else {
#pragma unroll
          for (const auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
            array(i1, comp) = INV_2 * buffer(i1, comp) +
                              INV_4 *
                                (buffer(i1 - 1, comp) + buffer(i1 + 1, comp));
          }
        }
      } else {
        raise::KernelError(HERE, "DigitalFilter_kernel: 1D implementation called for D != 1 or for non-Cartesian metric");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        if constexpr (C == Coord::Cart) {
          if ((is_conductor_i1min and i1 == i1_min) or
              (is_conductor_i1max and i1 == i1_max - 1)) {
            const auto i1side = is_conductor_i1min ? (i1 + 1) : (i1 - 1);
            array(i1, i2, cur::jx1) =
              (THREE * INV_4) * (FILTER2D_IN_I2(buffer, cur::jx1, i1, i2)) +
              (INV_4) * (FILTER2D_IN_I2(buffer, cur::jx1, i1side, i2));
          } else if ((is_conductor_i1min and i1 == i1_min + 1) or
                     (is_conductor_i1max and i1 == i1_max - 2)) {
            const auto i1side = is_conductor_i1min ? (i1 + 1) : (i1 - 1);
            array(i1,
                  i2,
                  cur::jx1) = INV_2 * (FILTER2D_IN_I2(buffer, cur::jx1, i1, i2)) +
                              INV_4 *
                                ((FILTER2D_IN_I2(buffer, cur::jx1, i1 - 1, i2)) +
                                 (FILTER2D_IN_I2(buffer, cur::jx1, i1 + 1, i2)));
            array(i1,
                  i2,
                  cur::jx2) = INV_2 * (FILTER2D_IN_I2(buffer, cur::jx2, i1, i2)) +
                              INV_4 *
                                (FILTER2D_IN_I2(buffer, cur::jx2, i1side, i2));
            array(i1,
                  i2,
                  cur::jx3) = INV_2 * (FILTER2D_IN_I2(buffer, cur::jx3, i1, i2)) +
                              INV_4 *
                                (FILTER2D_IN_I2(buffer, cur::jx3, i1side, i2));
          } else if ((is_conductor_i2min and i2 == i2_min) or
                     (is_conductor_i2max and i2 == i2_max - 1)) {
            const auto i2side = is_conductor_i2min ? (i2 + 1) : (i2 - 1);
            array(i1, i2, cur::jx2) =
              (THREE * INV_4) * (FILTER2D_IN_I1(buffer, cur::jx2, i1, i2)) +
              (INV_4) * (FILTER2D_IN_I1(buffer, cur::jx2, i1, i2side));
          } else if ((is_conductor_i2min and i2 == i2_min + 1) or
                     (is_conductor_i2max and i2 == i2_max - 2)) {
            const auto i2side = is_conductor_i2min ? (i2 + 1) : (i2 - 1);
            array(i1,
                  i2,
                  cur::jx1) = INV_2 * (FILTER2D_IN_I1(buffer, cur::jx1, i1, i2)) +
                              INV_4 *
                                (FILTER2D_IN_I1(buffer, cur::jx1, i1, i2side));
            array(i1,
                  i2,
                  cur::jx2) = INV_2 * (FILTER2D_IN_I1(buffer, cur::jx2, i1, i2)) +
                              INV_4 *
                                ((FILTER2D_IN_I1(buffer, cur::jx2, i1, i2 - 1)) +
                                 (FILTER2D_IN_I1(buffer, cur::jx2, i1, i2 + 1)));
            array(i1,
                  i2,
                  cur::jx3) = INV_2 * (FILTER2D_IN_I1(buffer, cur::jx3, i1, i2)) +
                              INV_4 *
                                (FILTER2D_IN_I1(buffer, cur::jx3, i1, i2side));
          } else {
#pragma unroll
            for (const auto comp : { cur::jx1, cur::jx2, cur::jx3 }) {
              array(i1, i2, comp) = INV_4 * buffer(i1, i2, comp) +
                                    INV_8 * (buffer(i1 - 1, i2, comp) +
                                             buffer(i1 + 1, i2, comp) +
                                             buffer(i1, i2 - 1, comp) +
                                             buffer(i1, i2 + 1, comp)) +
                                    INV_16 * (buffer(i1 - 1, i2 - 1, comp) +
                                              buffer(i1 + 1, i2 + 1, comp) +
                                              buffer(i1 - 1, i2 + 1, comp) +
                                              buffer(i1 + 1, i2 - 1, comp));
            }
          }
        } else { // spherical
          // @TODO: get rid of temporary variables
          real_t cur_00, cur_0p1, cur_0m1;
          if (is_axis_i2min && (i2 == i2_min)) {
            /* --------------------------------- r, phi --------------------------------- */
            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2);
            cur_0p1 = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2 + 1);
            // ... filter in theta
            array(i1, i2, cur::jx1) = INV_2 * cur_00 + INV_2 * cur_0p1;

            array(i1, i2, cur::jx3) = ZERO;

            /* ---------------------------------- theta --------------------------------- */
            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx2, i1, i2);
            cur_0p1 = FILTER2D_IN_I1(buffer, cur::jx2, i1, i2 + 1);
            // ... filter in theta
            array(i1, i2, cur::jx2) = INV_4 * (cur_00 + cur_0p1);
          } else if (is_axis_i2min && (i2 == i2_min + 1)) {
            /* --------------------------------- r, phi --------------------------------- */
            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2);
            cur_0p1 = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2 + 1);
            cur_0m1 = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2 - 1);
            // ... filter in theta
            array(i1, i2, cur::jx1) = INV_2 * cur_00 + INV_4 * (cur_0p1 + cur_0m1);

            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx3, i1, i2);
            cur_0p1 = FILTER2D_IN_I1(buffer, cur::jx3, i1, i2 + 1);
            // ... filter in theta
            array(i1, i2, cur::jx3) = INV_2 * cur_00 + INV_4 * cur_0p1;

            /* ---------------------------------- theta --------------------------------- */
            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx2, i1, i2);
            cur_0p1 = FILTER2D_IN_I1(buffer, cur::jx2, i1, i2 + 1);
            cur_0m1 = FILTER2D_IN_I1(buffer, cur::jx2, i1, i2 - 1);
            // ... filter in theta
            array(i1, i2, cur::jx2) = INV_2 * cur_00 + INV_4 * (cur_0m1 + cur_0p1);
          } else if (is_axis_i2max && (i2 == i2_max - 1)) {
            /* --------------------------------- r, phi --------------------------------- */
            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2);
            cur_0p1 = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2 + 1);
            cur_0m1 = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2 - 1);
            // ... filter in theta
            array(i1, i2, cur::jx1) = INV_2 * cur_00 + INV_4 * (cur_0m1 + cur_0p1);

            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx3, i1, i2);
            cur_0m1 = FILTER2D_IN_I1(buffer, cur::jx3, i1, i2 - 1);
            // ... filter in theta
            array(i1, i2, cur::jx3) = INV_2 * cur_00 + INV_4 * cur_0m1;

            /* ---------------------------------- theta --------------------------------- */
            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx2, i1, i2);
            cur_0m1 = FILTER2D_IN_I1(buffer, cur::jx2, i1, i2 - 1);
            // ... filter in theta
            array(i1, i2, cur::jx2) = INV_4 * (cur_00 + cur_0m1);
          } else if (is_axis_i2max && (i2 == i2_max)) {
            /* --------------------------------- r, phi --------------------------------- */
            // ... filter in r
            cur_00  = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2);
            cur_0m1 = FILTER2D_IN_I1(buffer, cur::jx1, i1, i2 - 1);
            // ... filter in theta
            array(i1, i2, cur::jx1) = INV_2 * cur_00 + INV_2 * cur_0m1;

            array(i1, i2, cur::jx3) = ZERO;
          } else {
#pragma unroll
            for (const auto comp : { cur::jx1, cur::jx2, cur::jx3 }) {
              array(i1, i2, comp) = INV_4 * buffer(i1, i2, comp) +
                                    INV_8 * (buffer(i1 - 1, i2, comp) +
                                             buffer(i1 + 1, i2, comp) +
                                             buffer(i1, i2 - 1, comp) +
                                             buffer(i1, i2 + 1, comp)) +
                                    INV_16 * (buffer(i1 - 1, i2 - 1, comp) +
                                              buffer(i1 + 1, i2 + 1, comp) +
                                              buffer(i1 - 1, i2 + 1, comp) +
                                              buffer(i1 + 1, i2 - 1, comp));
            }
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "DigitalFilter_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        if constexpr (C == Coord::Cart) {
          if ((is_conductor_i1min and i1 == i1_min) or
              (is_conductor_i1max and i1 == i1_max - 1)) {
            const auto i1side = is_conductor_i1min ? (i1 + 1) : (i1 - 1);
            array(i1, i2, i3, cur::jx1) =
              (THREE * INV_4) * (FILTER3D_IN_I2_I3(buffer, cur::jx1, i1, i2, i3)) +
              (INV_4) * (FILTER3D_IN_I2_I3(buffer, cur::jx1, i1side, i2, i3));
          } else if ((is_conductor_i1min and i1 == i1_min + 1) or
                     (is_conductor_i1max and i1 == i1_max - 2)) {
            const auto i1side = is_conductor_i1min ? (i1 + 1) : (i1 - 1);
            array(i1, i2, i3, cur::jx1) =
              INV_2 * (FILTER3D_IN_I2_I3(buffer, cur::jx1, i1, i2, i3)) +
              INV_4 * ((FILTER3D_IN_I2_I3(buffer, cur::jx1, i1 - 1, i2, i3)) +
                       (FILTER3D_IN_I2_I3(buffer, cur::jx1, i1 + 1, i2, i3)));
            array(i1, i2, i3, cur::jx2) =
              INV_2 * (FILTER3D_IN_I2_I3(buffer, cur::jx2, i1, i2, i3)) +
              INV_4 * (FILTER3D_IN_I2_I3(buffer, cur::jx2, i1side, i2, i3));
            array(i1, i2, i3, cur::jx3) =
              INV_2 * (FILTER3D_IN_I2_I3(buffer, cur::jx3, i1, i2, i3)) +
              INV_4 * (FILTER3D_IN_I2_I3(buffer, cur::jx3, i1side, i2, i3));
          } else if ((is_conductor_i2min and i2 == i2_min) or
                     (is_conductor_i2max and i2 == i2_max - 1)) {
            const auto i2side = is_conductor_i2min ? (i2 + 1) : (i2 - 1);
            array(i1, i2, i3, cur::jx2) =
              (THREE * INV_4) * (FILTER3D_IN_I1_I3(buffer, cur::jx2, i1, i2, i3)) +
              (INV_4) * (FILTER3D_IN_I1_I3(buffer, cur::jx2, i1, i2side, i3));
          } else if ((is_conductor_i2min and i2 == i2_min + 1) or
                     (is_conductor_i2max and i2 == i2_max - 2)) {
            const auto i2side = is_conductor_i2min ? (i2 + 1) : (i2 - 1);
            array(i1, i2, i3, cur::jx1) =
              INV_2 * (FILTER3D_IN_I1_I3(buffer, cur::jx1, i1, i2, i3)) +
              INV_4 * (FILTER3D_IN_I1_I3(buffer, cur::jx1, i1, i2side, i3));
            array(i1, i2, i3, cur::jx2) =
              INV_2 * (FILTER3D_IN_I1_I3(buffer, cur::jx2, i1, i2, i3)) +
              INV_4 * ((FILTER3D_IN_I1_I3(buffer, cur::jx2, i1, i2 - 1, i3)) +
                       (FILTER3D_IN_I1_I3(buffer, cur::jx2, i1, i2 + 1, i3)));
            array(i1, i2, i3, cur::jx3) =
              INV_2 * (FILTER3D_IN_I1_I3(buffer, cur::jx3, i1, i2, i3)) +
              INV_4 * (FILTER3D_IN_I1_I3(buffer, cur::jx3, i1, i2side, i3));
          } else if ((is_conductor_i3min and i3 == i3_min) or
                     (is_conductor_i3max and i3 == i3_max - 1)) {
            const auto i3side = is_conductor_i3min ? (i3 + 1) : (i3 - 1);
            array(i1, i2, i3, cur::jx3) =
              (THREE * INV_4) * (FILTER3D_IN_I1_I2(buffer, cur::jx3, i1, i2, i3)) +
              (INV_4) * (FILTER3D_IN_I1_I2(buffer, cur::jx3, i1, i2, i3side));
          } else if ((is_conductor_i3min and i3 == i3_min + 1) or
                     (is_conductor_i3max and i3 == i3_max - 2)) {
            const auto i3side = is_conductor_i3min ? (i3 + 1) : (i3 - 1);
            array(i1, i2, i3, cur::jx1) =
              INV_2 * (FILTER3D_IN_I1_I2(buffer, cur::jx1, i1, i2, i3)) +
              INV_4 * (FILTER3D_IN_I1_I2(buffer, cur::jx1, i1, i2, i3side));
            array(i1, i2, i3, cur::jx2) =
              INV_2 * (FILTER3D_IN_I1_I2(buffer, cur::jx2, i1, i2, i3)) +
              INV_4 * (FILTER3D_IN_I1_I2(buffer, cur::jx2, i1, i2, i3side));
            array(i1, i2, i3, cur::jx3) =
              INV_2 * (FILTER3D_IN_I1_I2(buffer, cur::jx3, i1, i2, i3)) +
              INV_4 * ((FILTER3D_IN_I1_I2(buffer, cur::jx3, i1, i2, i3 - 1)) +
                       (FILTER3D_IN_I1_I2(buffer, cur::jx3, i1, i2, i3 + 1)));
          } else {
#pragma unroll
            for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
              array(i1, i2, i3, comp) =
                INV_8 * buffer(i1, i2, i3, comp) +
                INV_16 *
                  (buffer(i1 - 1, i2, i3, comp) + buffer(i1 + 1, i2, i3, comp) +
                   buffer(i1, i2 - 1, i3, comp) + buffer(i1, i2 + 1, i3, comp) +
                   buffer(i1, i2, i3 - 1, comp) + buffer(i1, i2, i3 + 1, comp)) +
                INV_32 *
                  (buffer(i1 - 1, i2 - 1, i3, comp) +
                   buffer(i1 + 1, i2 + 1, i3, comp) +
                   buffer(i1 - 1, i2 + 1, i3, comp) +
                   buffer(i1 + 1, i2 - 1, i3, comp) +
                   buffer(i1, i2 - 1, i3 - 1, comp) +
                   buffer(i1, i2 + 1, i3 + 1, comp) +
                   buffer(i1, i2, i3 - 1, comp) + buffer(i1, i2, i3 + 1, comp) +
                   buffer(i1 - 1, i2, i3 - 1, comp) +
                   buffer(i1 + 1, i2, i3 + 1, comp) +
                   buffer(i1 - 1, i2, i3 + 1, comp) +
                   buffer(i1 + 1, i2, i3 - 1, comp)) +
                INV_64 * (buffer(i1 - 1, i2 - 1, i3 - 1, comp) +
                          buffer(i1 + 1, i2 + 1, i3 + 1, comp) +
                          buffer(i1 - 1, i2 + 1, i3 + 1, comp) +
                          buffer(i1 + 1, i2 - 1, i3 - 1, comp) +
                          buffer(i1 - 1, i2 - 1, i3 + 1, comp) +
                          buffer(i1 + 1, i2 + 1, i3 - 1, comp) +
                          buffer(i1 - 1, i2 + 1, i3 - 1, comp) +
                          buffer(i1 + 1, i2 - 1, i3 + 1, comp));
            }
          }
        } else {
          raise::KernelNotImplementedError(HERE);
        }
      } else {
        raise::KernelError(
          HERE,
          "DigitalFilter_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel

#undef FILTER3D_IN_I1_I3
#undef FILTER3D_IN_I2_I3
#undef FILTER3D_IN_I1_I2

#undef FILTER2D_IN_I2
#undef FILTER2D_IN_I1

#endif // DIGITAL_FILTER_HPP
