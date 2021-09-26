#include "global.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

template <>
void Simulation<One_D>::faradayHalfsubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "1D faraday";
  const real_t testval {2.0};
  Kokkos::parallel_for("faraday",
    NTTRange(N_GHOSTS, m_sim_params.m_resolution[0] - N_GHOSTS),
    Lambda (index_t i) {
      m_meshblock.ex1(i) = time;
      // update e,b
    }
  );
  real_t sum {0.0};
  Kokkos::parallel_reduce("faraday2",
    NTTRange(N_GHOSTS, m_sim_params.m_resolution[0] - N_GHOSTS),
    Lambda (index_t i, real_t & s) {
      s += m_meshblock.ex1(i);
      // update e,b
    }, sum
  );
  PLOGI << sum;
}
template <>
void Simulation<Two_D>::faradayHalfsubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "2D faraday";
}
template <>
void Simulation<Three_D>::faradayHalfsubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "3D faraday";
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
