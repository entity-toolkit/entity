#include "global.h"
#include "simulation.h"

#include "boris.hpp"

namespace ntt {

void Simulation1D::pushParticlesSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "1D pusher";
  Kokkos::parallel_for("pusher", NTT1DRange(0, 1), Boris1D(m_meshblock, 0));
}
void Simulation2D::pushParticlesSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "2D pusher";
  Kokkos::parallel_for("pusher", NTT1DRange(0, 1), Boris2D(m_meshblock, 0));
}
void Simulation3D::pushParticlesSubstep(const real_t& time) {
  UNUSED(time);
  PLOGD << "3D pusher";
  Kokkos::parallel_for("pusher", NTT1DRange(0, 1), Boris3D(m_meshblock, 0));
}

} // namespace ntt
