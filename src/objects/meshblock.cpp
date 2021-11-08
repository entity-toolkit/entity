#include "global.h"
#include "meshblock.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

template <>
Meshblock<ONE_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : em_fields {"EM", res[0] + 2 * N_GHOSTS},
      j_fields {"J", res[0] + 2 * N_GHOSTS},
      m_resolution {std::move(res)} {
  for (auto& part : parts) {
    particles.emplace_back(part);
  }
}

template <>
Meshblock<TWO_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : em_fields {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      j_fields {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      m_resolution {std::move(res)} {
  for (auto& part : parts) {
    particles.emplace_back(part);
  }
}

template <>
Meshblock<THREE_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : em_fields {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      j_fields {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      m_resolution {std::move(res)} {
  for (auto& part : parts) {
    particles.emplace_back(part);
  }
}

template <>
auto Meshblock<ONE_D>::loopActiveCells() -> ntt_1drange_t {
  return NTT1DRange(static_cast<range_t>(get_imin()), static_cast<range_t>(get_imax()));
}
template <>
auto Meshblock<TWO_D>::loopActiveCells() -> ntt_2drange_t {
  return NTT2DRange({get_imin(), get_jmin()}, {get_imax(), get_jmax()});
}
template <>
auto Meshblock<THREE_D>::loopActiveCells() -> ntt_3drange_t {
  return NTT3DRange({get_imin(), get_jmin(), get_kmin()}, {get_imax(), get_jmax(), get_kmax()});
}

template <>
void Meshblock<ONE_D>::verify(const SimulationParams&) {
  for (auto& p : particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("ERROR: undefined particle pusher.");
    }
  }
}

template <>
void Meshblock<TWO_D>::verify(const SimulationParams&) {
  for (auto& p : particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("ERROR: undefined particle pusher.");
    }
  }
}

template <>
void Meshblock<THREE_D>::verify(const SimulationParams&) {
  for (auto& p : particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("ERROR: undefined particle pusher.");
    }
  }
}

} // namespace ntt

template struct ntt::Meshblock<ntt::ONE_D>;
template struct ntt::Meshblock<ntt::TWO_D>;
template struct ntt::Meshblock<ntt::THREE_D>;
