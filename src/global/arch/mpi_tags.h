/**
 * @file arch/mpi_tags.h
 * @brief MPI tags for particle communication
 * @implements
 *   - mpi::PrtlSendTag<>
 *   - mpi::SendTag<> -> short
 * @namespaces:
 *   - mpi::
 */
#ifndef GLOBAL_ARCH_MPI_TAGS_H
#define GLOBAL_ARCH_MPI_TAGS_H

#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace mpi {
  using namespace dir;

  template <Dimension D>
  struct PrtlSendTag {
    static auto tag2dir(short tag) -> direction_t<D> {
      raise::ErrorIf((tag - 2 < 0) || (tag - 2 >= Directions<D>::all.size()),
                     "Invalid tag",
                     HERE);
      return Directions<D>::all[tag - 2];
    }

    inline static auto dir2tag(const direction_t<D>&) -> short;
  };

  template <>
  struct PrtlSendTag<Dim::_1D> {
    static constexpr short im1 { 2 };
    static constexpr short ip1 { 3 };

    static auto dir2tag(const direction_t<Dim::_1D>& dir) -> short {
      if (dir == direction_t<Dim::_1D>({ -1 })) {
        return im1;
      } else if (dir == direction_t<Dim::_1D>({ 1 })) {
        return ip1;
      } else {
        raise::Error("Invalid direction", HERE);
        throw;
      }
    }
  };

  template <>
  struct PrtlSendTag<Dim::_2D> {
    static constexpr short im1_jm1 { 2 };
    static constexpr short im1_j_0 { 3 };
    static constexpr short im1_jp1 { 4 };
    static constexpr short i_0_jm1 { 5 };
    static constexpr short i_0_jp1 { 6 };
    static constexpr short ip1_jm1 { 7 };
    static constexpr short ip1_j_0 { 8 };
    static constexpr short ip1_jp1 { 9 };

    static auto dir2tag(const direction_t<Dim::_2D>& dir) -> short {
      if (dir == direction_t<Dim::_2D>({ -1, -1 })) {
        return im1_jm1;
      } else if (dir == direction_t<Dim::_2D>({ -1, 0 })) {
        return im1_j_0;
      } else if (dir == direction_t<Dim::_2D>({ -1, 1 })) {
        return im1_jp1;
      } else if (dir == direction_t<Dim::_2D>({ 0, -1 })) {
        return i_0_jm1;
      } else if (dir == direction_t<Dim::_2D>({ 0, 1 })) {
        return i_0_jp1;
      } else if (dir == direction_t<Dim::_2D>({ 1, -1 })) {
        return ip1_jm1;
      } else if (dir == direction_t<Dim::_2D>({ 1, 0 })) {
        return ip1_j_0;
      } else if (dir == direction_t<Dim::_2D>({ 1, 1 })) {
        return ip1_jp1;
      } else {
        raise::Error("Invalid direction", HERE);
        throw;
      }
    }
  };

  template <>
  struct PrtlSendTag<Dim::_3D> {
    static constexpr short im1_jm1_km1 { 2 };
    static constexpr short im1_jm1_k_0 { 3 };
    static constexpr short im1_jm1_kp1 { 4 };
    static constexpr short im1_j_0_km1 { 5 };
    static constexpr short im1_j_0_k_0 { 6 };
    static constexpr short im1_j_0_kp1 { 7 };
    static constexpr short im1_jp1_km1 { 8 };
    static constexpr short im1_jp1_k_0 { 9 };
    static constexpr short im1_jp1_kp1 { 10 };
    static constexpr short i_0_jm1_km1 { 11 };
    static constexpr short i_0_jm1_k_0 { 12 };
    static constexpr short i_0_jm1_kp1 { 13 };
    static constexpr short i_0_j_0_km1 { 14 };
    static constexpr short i_0_j_0_kp1 { 15 };
    static constexpr short i_0_jp1_km1 { 16 };
    static constexpr short i_0_jp1_k_0 { 17 };
    static constexpr short i_0_jp1_kp1 { 18 };
    static constexpr short ip1_jm1_km1 { 19 };
    static constexpr short ip1_jm1_k_0 { 20 };
    static constexpr short ip1_jm1_kp1 { 21 };
    static constexpr short ip1_j_0_km1 { 22 };
    static constexpr short ip1_j_0_k_0 { 23 };
    static constexpr short ip1_j_0_kp1 { 24 };
    static constexpr short ip1_jp1_km1 { 25 };
    static constexpr short ip1_jp1_k_0 { 26 };
    static constexpr short ip1_jp1_kp1 { 27 };

    static auto dir2tag(const direction_t<Dim::_3D>& dir) -> short {
      if (dir == direction_t<Dim::_3D>({ -1, -1, -1 })) {
        return im1_jm1_km1;
      } else if (dir == direction_t<Dim::_3D>({ -1, -1, 0 })) {
        return im1_jm1_k_0;
      } else if (dir == direction_t<Dim::_3D>({ -1, -1, 1 })) {
        return im1_jm1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ -1, 0, -1 })) {
        return im1_j_0_km1;
      } else if (dir == direction_t<Dim::_3D>({ -1, 0, 0 })) {
        return im1_j_0_k_0;
      } else if (dir == direction_t<Dim::_3D>({ -1, 0, 1 })) {
        return im1_j_0_kp1;
      } else if (dir == direction_t<Dim::_3D>({ -1, 1, -1 })) {
        return im1_jp1_km1;
      } else if (dir == direction_t<Dim::_3D>({ -1, 1, 0 })) {
        return im1_jp1_k_0;
      } else if (dir == direction_t<Dim::_3D>({ -1, 1, 1 })) {
        return im1_jp1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 0, -1, -1 })) {
        return i_0_jm1_km1;
      } else if (dir == direction_t<Dim::_3D>({ 0, -1, 0 })) {
        return i_0_jm1_k_0;
      } else if (dir == direction_t<Dim::_3D>({ 0, -1, 1 })) {
        return i_0_jm1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 0, 0, -1 })) {
        return i_0_j_0_km1;
      } else if (dir == direction_t<Dim::_3D>({ 0, 0, 1 })) {
        return i_0_j_0_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 0, 1, -1 })) {
        return i_0_jp1_km1;
      } else if (dir == direction_t<Dim::_3D>({ 0, 1, 0 })) {
        return i_0_jp1_k_0;
      } else if (dir == direction_t<Dim::_3D>({ 0, 1, 1 })) {
        return i_0_jp1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 1, -1, -1 })) {
        return ip1_jm1_km1;
      } else if (dir == direction_t<Dim::_3D>({ 1, -1, 0 })) {
        return ip1_jm1_k_0;
      } else if (dir == direction_t<Dim::_3D>({ 1, -1, 1 })) {
        return ip1_jm1_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 1, 0, -1 })) {
        return ip1_j_0_km1;
      } else if (dir == direction_t<Dim::_3D>({ 1, 0, 0 })) {
        return ip1_j_0_k_0;
      } else if (dir == direction_t<Dim::_3D>({ 1, 0, 1 })) {
        return ip1_j_0_kp1;
      } else if (dir == direction_t<Dim::_3D>({ 1, 1, -1 })) {
        return ip1_jp1_km1;
      } else if (dir == direction_t<Dim::_3D>({ 1, 1, 0 })) {
        return ip1_jp1_k_0;
      } else if (dir == direction_t<Dim::_3D>({ 1, 1, 1 })) {
        return ip1_jp1_kp1;
      } else {
        raise::Error("Invalid direction", HERE);
        throw;
      }
    }
  };

  Inline auto SendTag(short tag, bool im1, bool ip1) -> short {
    return static_cast<short>(
      ((im1) * (PrtlSendTag<Dim::_1D>::im1 - 1) +
       (ip1) * (PrtlSendTag<Dim::_1D>::ip1 - 1) + static_cast<short>(1)) *
      tag);
  }

  Inline auto SendTag(short tag, bool im1, bool ip1, bool jm1, bool jp1) -> short {
    return static_cast<short>(
      ((im1 && jm1) * (PrtlSendTag<Dim::_2D>::im1_jm1 - 1) +
       (im1 && jp1) * (PrtlSendTag<Dim::_2D>::im1_jp1 - 1) +
       (ip1 && jm1) * (PrtlSendTag<Dim::_2D>::ip1_jm1 - 1) +
       (ip1 && jp1) * (PrtlSendTag<Dim::_2D>::ip1_jp1 - 1) +
       (im1 && !jp1 && !jm1) * (PrtlSendTag<Dim::_2D>::im1_j_0 - 1) +
       (ip1 && !jp1 && !jm1) * (PrtlSendTag<Dim::_2D>::ip1_j_0 - 1) +
       (jm1 && !ip1 && !im1) * (PrtlSendTag<Dim::_2D>::i_0_jm1 - 1) +
       (jp1 && !ip1 && !im1) * (PrtlSendTag<Dim::_2D>::i_0_jp1 - 1) + 1) *
      tag);
  }

  Inline auto SendTag(short tag, bool im1, bool ip1, bool jm1, bool jp1, bool km1, bool kp1)
    -> short {
    return static_cast<short>(
      ((im1 && jm1 && km1) * (PrtlSendTag<Dim::_3D>::im1_jm1_km1 - 1) +
       (im1 && jm1 && kp1) * (PrtlSendTag<Dim::_3D>::im1_jm1_kp1 - 1) +
       (im1 && jp1 && km1) * (PrtlSendTag<Dim::_3D>::im1_jp1_km1 - 1) +
       (im1 && jp1 && kp1) * (PrtlSendTag<Dim::_3D>::im1_jp1_kp1 - 1) +
       (ip1 && jm1 && km1) * (PrtlSendTag<Dim::_3D>::ip1_jm1_km1 - 1) +
       (ip1 && jm1 && kp1) * (PrtlSendTag<Dim::_3D>::ip1_jm1_kp1 - 1) +
       (ip1 && jp1 && km1) * (PrtlSendTag<Dim::_3D>::ip1_jp1_km1 - 1) +
       (ip1 && jp1 && kp1) * (PrtlSendTag<Dim::_3D>::ip1_jp1_kp1 - 1) +
       (im1 && jm1 && !km1 && !kp1) * (PrtlSendTag<Dim::_3D>::im1_jm1_k_0 - 1) +
       (im1 && jp1 && !km1 && !kp1) * (PrtlSendTag<Dim::_3D>::im1_jp1_k_0 - 1) +
       (ip1 && jm1 && !km1 && !kp1) * (PrtlSendTag<Dim::_3D>::ip1_jm1_k_0 - 1) +
       (ip1 && jp1 && !km1 && !kp1) * (PrtlSendTag<Dim::_3D>::ip1_jp1_k_0 - 1) +
       (im1 && !jm1 && !jp1 && km1) * (PrtlSendTag<Dim::_3D>::im1_j_0_km1 - 1) +
       (im1 && !jm1 && !jp1 && kp1) * (PrtlSendTag<Dim::_3D>::im1_j_0_kp1 - 1) +
       (ip1 && !jm1 && !jp1 && km1) * (PrtlSendTag<Dim::_3D>::ip1_j_0_km1 - 1) +
       (ip1 && !jm1 && !jp1 && kp1) * (PrtlSendTag<Dim::_3D>::ip1_j_0_kp1 - 1) +
       (!im1 && !ip1 && jm1 && km1) * (PrtlSendTag<Dim::_3D>::i_0_jm1_km1 - 1) +
       (!im1 && !ip1 && jm1 && kp1) * (PrtlSendTag<Dim::_3D>::i_0_jm1_kp1 - 1) +
       (!im1 && !ip1 && jp1 && km1) * (PrtlSendTag<Dim::_3D>::i_0_jp1_km1 - 1) +
       (!im1 && !ip1 && jp1 && kp1) * (PrtlSendTag<Dim::_3D>::i_0_jp1_kp1 - 1) +
       (!im1 && !ip1 && !jm1 && !jp1 && km1) *
         (PrtlSendTag<Dim::_3D>::i_0_j_0_km1 - 1) +
       (!im1 && !ip1 && !jm1 && !jp1 && kp1) *
         (PrtlSendTag<Dim::_3D>::i_0_j_0_kp1 - 1) +
       (!im1 && !ip1 && jm1 && !km1 && !kp1) *
         (PrtlSendTag<Dim::_3D>::i_0_jm1_k_0 - 1) +
       (!im1 && !ip1 && jp1 && !km1 && !kp1) *
         (PrtlSendTag<Dim::_3D>::i_0_jp1_k_0 - 1) +
       (im1 && !jm1 && !jp1 && !km1 && !kp1) *
         (PrtlSendTag<Dim::_3D>::im1_j_0_k_0 - 1) +
       (ip1 && !jm1 && !jp1 && !km1 && !kp1) *
         (PrtlSendTag<Dim::_3D>::ip1_j_0_k_0 - 1) +
       1) *
      tag);
  }
} // namespace mpi

#endif // GLOBAL_ARCH_MPI_TAGS_H
