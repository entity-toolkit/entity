#ifndef DIRECTIONS_H
#define DIRECTIONS_H

#include "definitions.h"

#include <vector>

namespace ntt {
  

#ifdef MPI_ENABLED
  template <Dimension D>
  struct PrtlSendTag {
    inline static auto tag2dir(short tag) -> direction_t<D> {
      if ((tag - 2 < 0) || (tag - 2 >= Directions<D>::all.size())) {
        NTTHostError("Invalid tag.");
      }
      return Directions<D>::all[tag - 2];
    }

    inline static auto dir2tag(const direction_t<D>&) -> short;
  };

  template <>
  struct PrtlSendTag<Dim1> {
    inline static constexpr short im1 { 2 };
    inline static constexpr short ip1 { 3 };

    inline static auto dir2tag(const direction_t<Dim1>& dir) -> short {
      if (dir == direction_t<Dim1>({ -1 })) {
        return im1;
      } else if (dir == direction_t<Dim1>({ 1 })) {
        return ip1;
      } else {
        NTTHostError("Invalid direction.");
      }
    }
  };

  template <>
  struct PrtlSendTag<Dim2> {
    inline static constexpr short im1_jm1 { 2 };
    inline static constexpr short im1__j0 { 3 };
    inline static constexpr short im1_jp1 { 4 };
    inline static constexpr short i0__jm1 { 5 };
    inline static constexpr short i0__jp1 { 6 };
    inline static constexpr short ip1_jm1 { 7 };
    inline static constexpr short ip1__j0 { 8 };
    inline static constexpr short ip1_jp1 { 9 };

    inline static auto dir2tag(const direction_t<Dim2>& dir) -> short {
      if (dir == direction_t<Dim2>({ -1, -1 })) {
        return im1_jm1;
      } else if (dir == direction_t<Dim2>({ -1, 0 })) {
        return im1__j0;
      } else if (dir == direction_t<Dim2>({ -1, 1 })) {
        return im1_jp1;
      } else if (dir == direction_t<Dim2>({ 0, -1 })) {
        return i0__jm1;
      } else if (dir == direction_t<Dim2>({ 0, 1 })) {
        return i0__jp1;
      } else if (dir == direction_t<Dim2>({ 1, -1 })) {
        return ip1_jm1;
      } else if (dir == direction_t<Dim2>({ 1, 0 })) {
        return ip1__j0;
      } else if (dir == direction_t<Dim2>({ 1, 1 })) {
        return ip1_jp1;
      } else {
        NTTHostError("Invalid direction.");
      }
    }
  };

  template <>
  struct PrtlSendTag<Dim3> {
    inline static constexpr short im1_jm1_km1 { 2 };
    inline static constexpr short im1_jm1__k0 { 3 };
    inline static constexpr short im1_jm1_kp1 { 4 };
    inline static constexpr short im1__j0_km1 { 5 };
    inline static constexpr short im1__j0__k0 { 6 };
    inline static constexpr short im1__j0_kp1 { 7 };
    inline static constexpr short im1_jp1_km1 { 8 };
    inline static constexpr short im1_jp1__k0 { 9 };
    inline static constexpr short im1_jp1_kp1 { 10 };
    inline static constexpr short i0__jm1_km1 { 11 };
    inline static constexpr short i0__jm1__k0 { 12 };
    inline static constexpr short i0__jm1_kp1 { 13 };
    inline static constexpr short i0___j0_km1 { 14 };
    inline static constexpr short i0___j0_kp1 { 15 };
    inline static constexpr short i0__jp1_km1 { 16 };
    inline static constexpr short i0__jp1__k0 { 17 };
    inline static constexpr short i0__jp1_kp1 { 18 };
    inline static constexpr short ip1_jm1_km1 { 19 };
    inline static constexpr short ip1_jm1__k0 { 20 };
    inline static constexpr short ip1_jm1_kp1 { 21 };
    inline static constexpr short ip1__j0_km1 { 22 };
    inline static constexpr short ip1__j0__k0 { 23 };
    inline static constexpr short ip1__j0_kp1 { 24 };
    inline static constexpr short ip1_jp1_km1 { 25 };
    inline static constexpr short ip1_jp1__k0 { 26 };
    inline static constexpr short ip1_jp1_kp1 { 27 };

    inline static auto dir2tag(const direction_t<Dim3>& dir) -> short {
      if (dir == direction_t<Dim3>({ -1, -1, -1 })) {
        return im1_jm1_km1;
      } else if (dir == direction_t<Dim3>({ -1, -1, 0 })) {
        return im1_jm1__k0;
      } else if (dir == direction_t<Dim3>({ -1, -1, 1 })) {
        return im1_jm1_kp1;
      } else if (dir == direction_t<Dim3>({ -1, 0, -1 })) {
        return im1__j0_km1;
      } else if (dir == direction_t<Dim3>({ -1, 0, 0 })) {
        return im1__j0__k0;
      } else if (dir == direction_t<Dim3>({ -1, 0, 1 })) {
        return im1__j0_kp1;
      } else if (dir == direction_t<Dim3>({ -1, 1, -1 })) {
        return im1_jp1_km1;
      } else if (dir == direction_t<Dim3>({ -1, 1, 0 })) {
        return im1_jp1__k0;
      } else if (dir == direction_t<Dim3>({ -1, 1, 1 })) {
        return im1_jp1_kp1;
      } else if (dir == direction_t<Dim3>({ 0, -1, -1 })) {
        return i0__jm1_km1;
      } else if (dir == direction_t<Dim3>({ 0, -1, 0 })) {
        return i0__jm1__k0;
      } else if (dir == direction_t<Dim3>({ 0, -1, 1 })) {
        return i0__jm1_kp1;
      } else if (dir == direction_t<Dim3>({ 0, 0, -1 })) {
        return i0___j0_km1;
      } else if (dir == direction_t<Dim3>({ 0, 0, 1 })) {
        return i0___j0_kp1;
      } else if (dir == direction_t<Dim3>({ 0, 1, -1 })) {
        return i0__jp1_km1;
      } else if (dir == direction_t<Dim3>({ 0, 1, 0 })) {
        return i0__jp1__k0;
      } else if (dir == direction_t<Dim3>({ 0, 1, 1 })) {
        return i0__jp1_kp1;
      } else if (dir == direction_t<Dim3>({ 1, -1, -1 })) {
        return ip1_jm1_km1;
      } else if (dir == direction_t<Dim3>({ 1, -1, 0 })) {
        return ip1_jm1__k0;
      } else if (dir == direction_t<Dim3>({ 1, -1, 1 })) {
        return ip1_jm1_kp1;
      } else if (dir == direction_t<Dim3>({ 1, 0, -1 })) {
        return ip1__j0_km1;
      } else if (dir == direction_t<Dim3>({ 1, 0, 0 })) {
        return ip1__j0__k0;
      } else if (dir == direction_t<Dim3>({ 1, 0, 1 })) {
        return ip1__j0_kp1;
      } else if (dir == direction_t<Dim3>({ 1, 1, -1 })) {
        return ip1_jp1_km1;
      } else if (dir == direction_t<Dim3>({ 1, 1, 0 })) {
        return ip1_jp1__k0;
      } else if (dir == direction_t<Dim3>({ 1, 1, 1 })) {
        return ip1_jp1_kp1;
      } else {
        NTTHostError("Invalid direction.");
      }
    }
  };
#endif

} // namespace ntt

#endif // DIRECTIONS_H