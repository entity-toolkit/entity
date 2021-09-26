#include "global.h"
#include "simulation.h"

#include <plog/Log.h>

namespace ntt {

template <>
void Simulation<One_D>::faradayHalfsubstep(const real_t &time) {
  UNUSED(time);
  PLOGD << "1D faraday";
  Kokkos::parallel_for("faraday", m_sim_params.m_resolution[0],
    Lambda (index_t i) {
      // update e,b
    }
  );
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
