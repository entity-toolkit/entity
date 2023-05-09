#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "pic.h"

namespace ntt {
#ifdef MINKOWSKI_METRIC
  /**
   * @brief 1d periodic field bc (minkowski).
   */
  template <>
  void PIC<Dim1>::FieldsExchange() {
    auto&         mblock = this->meshblock;
    const auto    i1min = mblock.i1_min(), i1max = mblock.i1_max();
    range_tuple_t i1_range_to;
    range_tuple_t i1_range_from;
    NTTHostErrorIf((mblock.boundaries[0][0] != BoundaryCondition::PERIODIC),
                   "1d minkowski only supports periodic boundaries");

    for (auto dir1 { -1 }; dir1 < 2; ++dir1) {
      if (dir1 == -1) {
        i1_range_to   = range_tuple_t(i1min - N_GHOSTS, i1min);
        i1_range_from = range_tuple_t(i1max - N_GHOSTS, i1max);
      } else if (dir1 == 0) {
        continue;
      } else {
        i1_range_to   = range_tuple_t(i1max, i1max + N_GHOSTS);
        i1_range_from = range_tuple_t(i1min, i1min + N_GHOSTS);
      }
      Kokkos::deep_copy(Kokkos::subview(mblock.em, i1_range_to, Kokkos::ALL()),
                        Kokkos::subview(mblock.em, i1_range_from, Kokkos::ALL()));
    }
  }

  /**
   * @brief 2d periodic field bc.
   *
   */
  template <>
  void PIC<Dim2>::FieldsExchange() {
    auto&         mblock = this->meshblock;
    const auto    i1min = mblock.i1_min(), i1max = mblock.i1_max();
    const auto    i2min = mblock.i2_min(), i2max = mblock.i2_max();
    range_tuple_t i1_range_to, i2_range_to;
    range_tuple_t i1_range_from, i2_range_from;
    NTTHostErrorIf((mblock.boundaries[0][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[1][0] != BoundaryCondition::PERIODIC),
                   "2d minkowski only supports periodic boundaries");

    for (auto dir1 { -1 }; dir1 < 2; ++dir1) {
      if (dir1 == -1) {
        i1_range_to   = range_tuple_t(i1min - N_GHOSTS, i1min);
        i1_range_from = range_tuple_t(i1max - N_GHOSTS, i1max);
      } else if (dir1 == 0) {
        i1_range_to   = range_tuple_t(i1min, i1max);
        i1_range_from = range_tuple_t(i1min, i1max);
      } else {
        i1_range_to   = range_tuple_t(i1max, i1max + N_GHOSTS);
        i1_range_from = range_tuple_t(i1min, i1min + N_GHOSTS);
      }
      for (auto dir2 { -1 }; dir2 < 2; ++dir2) {
        if (dir2 == -1) {
          i2_range_to   = range_tuple_t(i2min - N_GHOSTS, i2min);
          i2_range_from = range_tuple_t(i2max - N_GHOSTS, i2max);
        } else if (dir2 == 0) {
          i2_range_to   = range_tuple_t(i2min, i2max);
          i2_range_from = range_tuple_t(i2min, i2max);
        } else {
          i2_range_to   = range_tuple_t(i2max, i2max + N_GHOSTS);
          i2_range_from = range_tuple_t(i2min, i2min + N_GHOSTS);
        }
        if (dir1 == 0 && dir2 == 0) {
          continue;
        }
        Kokkos::deep_copy(
          Kokkos::subview(mblock.em, i1_range_to, i2_range_to, Kokkos::ALL()),
          Kokkos::subview(mblock.em, i1_range_from, i2_range_from, Kokkos::ALL()));
      }
    }
  }

  /**
   * @brief 3d periodic field bc.
   *
   */
  template <>
  void PIC<Dim3>::FieldsExchange() {
    auto&         mblock = this->meshblock;
    const auto    i1min = mblock.i1_min(), i1max = mblock.i1_max();
    const auto    i2min = mblock.i2_min(), i2max = mblock.i2_max();
    const auto    i3min = mblock.i3_min(), i3max = mblock.i3_max();
    range_tuple_t i1_range_to, i2_range_to, i3_range_to;
    range_tuple_t i1_range_from, i2_range_from, i3_range_from;
    NTTHostErrorIf((mblock.boundaries[0][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[1][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[2][0] != BoundaryCondition::PERIODIC),
                   "3d minkowski only supports periodic boundaries");

    for (auto dir1 { -1 }; dir1 < 2; ++dir1) {
      if (dir1 == -1) {
        i1_range_to   = range_tuple_t(i1min - N_GHOSTS, i1min);
        i1_range_from = range_tuple_t(i1max - N_GHOSTS, i1max);
      } else if (dir1 == 0) {
        i1_range_to   = range_tuple_t(i1min, i1max);
        i1_range_from = range_tuple_t(i1min, i1max);
      } else {
        i1_range_to   = range_tuple_t(i1max, i1max + N_GHOSTS);
        i1_range_from = range_tuple_t(i1min, i1min + N_GHOSTS);
      }
      for (auto dir2 { -1 }; dir2 < 2; ++dir2) {
        if (dir2 == -1) {
          i2_range_to   = range_tuple_t(i2min - N_GHOSTS, i2min);
          i2_range_from = range_tuple_t(i2max - N_GHOSTS, i2max);
        } else if (dir2 == 0) {
          i2_range_to   = range_tuple_t(i2min, i2max);
          i2_range_from = range_tuple_t(i2min, i2max);
        } else {
          i2_range_to   = range_tuple_t(i2max, i2max + N_GHOSTS);
          i2_range_from = range_tuple_t(i2min, i2min + N_GHOSTS);
        }
        for (auto dir3 { -1 }; dir3 < 2; ++dir3) {
          if (dir3 == -1) {
            i3_range_to   = range_tuple_t(i3min - N_GHOSTS, i3min);
            i3_range_from = range_tuple_t(i3max - N_GHOSTS, i3max);
          } else if (dir2 == 0) {
            i3_range_to   = range_tuple_t(i3min, i3max);
            i3_range_from = range_tuple_t(i3min, i3max);
          } else {
            i3_range_to   = range_tuple_t(i3max, i3max + N_GHOSTS);
            i3_range_from = range_tuple_t(i3min, i3min + N_GHOSTS);
          }
          if (dir1 == 0 && dir2 == 0 && dir3 == 0) {
            continue;
          }
          Kokkos::deep_copy(
            Kokkos::subview(mblock.em, i1_range_to, i2_range_to, i3_range_to, Kokkos::ALL()),
            Kokkos::subview(
              mblock.em, i1_range_from, i2_range_from, i3_range_from, Kokkos::ALL()));
        }
      }
    }
  }
#else
  template <Dimension D>
  void PIC<D>::FieldsExchange() {}
#endif

}    // namespace ntt

#ifndef MINKOWSKI_METRIC

template void ntt::PIC<ntt::Dim1>::FieldsExchange();
template void ntt::PIC<ntt::Dim2>::FieldsExchange();
template void ntt::PIC<ntt::Dim3>::FieldsExchange();

#endif