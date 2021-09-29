#include "global.h"
#include "meshblock.h"
#include "simulation.h"
#include "faraday.hpp"

#include <plog/Log.h>

#include <Kokkos_Core.hpp>

namespace ntt {

template <>
void Simulation<One_D>::faradayHalfsubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "1D faraday";
  const real_t testval {-2.0};
  auto range = NTT1DRange({N_GHOSTS}, {m_sim_params.m_resolution[0] - N_GHOSTS});
  // if cartesian:
  Kokkos::parallel_for("faraday", range, Faraday1DHalfstep_Cartesian(m_meshblock, testval));
}
template <>
void Simulation<Two_D>::faradayHalfsubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "2D faraday";
  const real_t testval {-2.0};
  auto range = NTT2DRange({N_GHOSTS, N_GHOSTS}, {m_sim_params.m_resolution[0] - N_GHOSTS, m_sim_params.m_resolution[1] - N_GHOSTS});
  // if cartesian:
  Kokkos::parallel_for("faraday", range, Faraday2DHalfstep_Cartesian(m_meshblock, testval));
}
template <>
void Simulation<Three_D>::faradayHalfsubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "3D faraday";
  const real_t testval {-2.0};
  auto range = NTT3DRange({N_GHOSTS, N_GHOSTS, N_GHOSTS}, {m_sim_params.m_resolution[0] - N_GHOSTS, m_sim_params.m_resolution[1] - N_GHOSTS, m_sim_params.m_resolution[2] - N_GHOSTS});
  // if cartesian:
  Kokkos::parallel_for("faraday", range, Faraday3DHalfstep_Cartesian(m_meshblock, testval));
}

template <>
void Simulation<One_D>::ampereSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "1D ampere";
}
template <>
void Simulation<Two_D>::ampereSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "2D ampere";
}
template <>
void Simulation<Three_D>::ampereSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "3D ampere";
}


template <>
void Simulation<One_D>::addCurrentsSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "1D add current";
}
template <>
void Simulation<Two_D>::addCurrentsSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "2D add current";
}
template <>
void Simulation<Three_D>::addCurrentsSubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "3D add current";
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

}
