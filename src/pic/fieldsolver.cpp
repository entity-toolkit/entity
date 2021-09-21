#include "global.h"
#include "picsim.h"

#ifdef KOKKOS
#  include <Kokkos_Core.hpp>
#endif

#include <plog/Log.h>

namespace ntt {

void PICSimulation1D::faradayHalfsubstep(const real_t &time) {
  PLOGD << time;
  
}

void PICSimulation2D::faradayHalfsubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation3D::faradayHalfsubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation1D::ampereSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation2D::ampereSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation3D::ampereSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation1D::addCurrentsSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation2D::addCurrentsSubstep(const real_t &time) {
  PLOGD << time;
}

void PICSimulation3D::addCurrentsSubstep(const real_t &time) {
  PLOGD << time;
}


}

//
// do i = 0, this_meshblock%ptr%sx - 1
//   ip1 = i + 1
//   by(i, j, k) = by(i, j, k) + const *&
//             & (ez(ip1, j, k) - ez(i, j, k))
//   bz(i, j, k) = bz(i, j, k) + const *&
//             & (-ey(ip1, j, k) + ey(i, j, k))
// enddo
// #elif twoD
// k = 0
// do j = 0, this_meshblock%ptr%sy - 1
//   jp1 = j + 1
//   do i = 0, this_meshblock%ptr%sx - 1
//     ip1 = i + 1
//     bx(i, j, k) = bx(i, j, k) + const *&
//               & (-ez(i, jp1, k) + ez(i, j, k))
//     by(i, j, k) = by(i, j, k) + const *&
//               & (ez(ip1, j, k) - ez(i, j, k))
//     bz(i, j, k) = bz(i, j, k) + const *&
//               & (ex(i, jp1, k) - ex(i, j, k) - ey(ip1, j, k) + ey(i, j, k))
//   enddo
// enddo
// #elif threeD
// do k = 0, this_meshblock%ptr%sz - 1
//   kp1 = k + 1
//   do j = 0, this_meshblock%ptr%sy - 1
//     jp1 = j + 1
//     do i = 0, this_meshblock%ptr%sx - 1
//       ip1 = i + 1
//       bx(i, j, k) = bx(i, j, k) + const *&
//                 & (ey(i, j, kp1) - ey(i, j, k) - ez(i, jp1, k) + ez(i, j, k))
//       by(i, j, k) = by(i, j, k) + const *&
//                 & (ez(ip1, j, k) - ez(i, j, k) - ex(i, j, kp1) + ex(i, j, k))
//       bz(i, j, k) = bz(i, j, k) + const *&
//                 & (ex(i, jp1, k) - ex(i, j, k) - ey(ip1, j, k) + ey(i, j, k))
//     enddo
