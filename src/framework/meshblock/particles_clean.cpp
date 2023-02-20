#include "wrapper.h"

#include "meshblock.h"

#include <vector>

namespace ntt {
  template <Dimension D, SimulationEngine S>
  auto Meshblock<D, S>::RemoveDeadParticles(const double& max_dead_frac)
    -> std::vector<double> {
    std::vector<double> dead_fractions = {};
    for (auto& species : particles) {
      auto npart_alive   = species.CountLiving();
      auto dead_fraction = 1.0 - (double)(npart_alive) / (double)(species.npart());
      if ((species.npart() > 0) && dead_fraction >= (double)max_dead_frac) {
        species.ReshuffleDead();
        species.setNpart(npart_alive);
      }
      dead_fractions.push_back(dead_fraction);
    }
    return dead_fractions;
  }
}    // namespace ntt

#ifdef PIC_ENGINE
template std::vector<double> ntt::Meshblock<ntt::Dim1, ntt::PICEngine>::RemoveDeadParticles(
  const double&);
template std::vector<double> ntt::Meshblock<ntt::Dim2, ntt::PICEngine>::RemoveDeadParticles(
  const double&);
template std::vector<double> ntt::Meshblock<ntt::Dim3, ntt::PICEngine>::RemoveDeadParticles(
  const double&);
#elif defined(GRPIC_ENGINE)

#endif