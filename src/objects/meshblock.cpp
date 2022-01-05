#include "global.h"
#include "meshblock.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

  template <Dimension D>
  Meshblock<D>::Meshblock(std::vector<real_t> ext,
                          std::vector<std::size_t> res,
                          std::vector<ParticleSpecies>& parts)
      : Fields<D>(res) {
    for (auto& part : parts) {
      particles.emplace_back(part);
    }
  }

  template <Dimension D>
  void Meshblock<D>::verify(const SimulationParams&) {
    if ((this->Ni == 1) || 
       ((this->Nj > 1) && (static_cast<short>(D) < 2)) || 
       ((this->Nk > 1) && (static_cast<short>(D) < 3))) {
      throw std::logic_error("# Error: wrong dimension inferred in Meshblock.");
    }
    for (auto& p : particles) {
      if (p.get_pusher() == UNDEFINED_PUSHER) {
        throw std::logic_error("# Error: undefined particle pusher.");
      }
    }
  }

} // namespace ntt

template struct ntt::Meshblock<ntt::ONE_D>;
template struct ntt::Meshblock<ntt::TWO_D>;
template struct ntt::Meshblock<ntt::THREE_D>;
