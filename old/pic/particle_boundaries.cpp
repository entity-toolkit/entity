#include "global.h"
#include "simulation.h"

#include "particle_periodic_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

  template <Dimension D>
  void Simulation<D>::particleBoundaryConditions(const real_t& time) {
    UNUSED(time);
    // for (auto& species : mblock.particles) {
    //   Kokkos::parallel_for(
    //       "prtl_bc",
    //       species.loopParticles(),
    //       PrtlBC_Periodic<D>(mblock.m_extent, species));
    // }
  }

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;
