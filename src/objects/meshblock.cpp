#include "global.h"
#include "meshblock.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

  template <Dimension D>
  Meshblock<D>::Meshblock(std::vector<real_t> ext,
                          std::vector<std::size_t> res,
                          std::vector<ParticleSpecies>& parts)
      : Fields<D>(res),
        Grid<D>(ext, res) {
    for (auto& part : parts) {
      particles.emplace_back(part);
    }
  }

  template <Dimension D>
  void Meshblock<D>::verify(const SimulationParams&) {
    if (this->m_extent.size() != 2 * static_cast<short>(D)) {
      throw std::logic_error("# Error: not enough values in extent.");
    }
    if (this->m_resolution.size() != static_cast<short>(D)) {
      throw std::logic_error("# Error: not enough values in resolution.");
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
