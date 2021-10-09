#include "global.h"
#include "meshblock.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

template <Dimension D>
void Meshblock<D>::printDetails() {
  if (particles.size() > 0) {
    PLOGI << "[particles]";
    for (std::size_t i {0}; i < particles.size(); ++i) {
      PLOGI << "   [species #" << i + 1 << "]";
      PLOGI << "      label: " << particles[i].get_label();
      PLOGI << "      mass: " << particles[i].get_mass();
      PLOGI << "      charge: " << particles[i].get_charge();
      PLOGI << "      pusher: " << stringifyParticlePusher(particles[i].get_pusher());
      PLOGI << "      maxnpart: " << particles[i].get_maxnpart() << " (" << particles[i].get_npart()
            << ")";
    }
  } else {
    PLOGI << "[no particles]";
  }
}

template <>
Meshblock<ONE_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : ex1 {"Ex1", res[0] + 2 * N_GHOSTS},
      ex2 {"Ex2", res[0] + 2 * N_GHOSTS},
      ex3 {"Ex3", res[0] + 2 * N_GHOSTS},
      bx1 {"Bx1", res[0] + 2 * N_GHOSTS},
      bx2 {"Bx2", res[0] + 2 * N_GHOSTS},
      bx3 {"Bx3", res[0] + 2 * N_GHOSTS},
      jx1 {"Jx1", res[0] + 2 * N_GHOSTS},
      jx2 {"Jx2", res[0] + 2 * N_GHOSTS},
      jx3 {"Jx3", res[0] + 2 * N_GHOSTS},
      m_resolution {std::move(res)} {
  for (auto& part : parts) {
    particles.emplace_back(part);
  }
}

template <>
Meshblock<TWO_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : ex1 {"Ex1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      ex2 {"Ex2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      ex3 {"Ex3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx1 {"Bx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx2 {"Bx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx3 {"Bx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx1 {"Jx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx2 {"Jx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx3 {"Jx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      m_resolution {std::move(res)} {
  for (auto& part : parts) {
    particles.emplace_back(part);
  }
}

template <>
Meshblock<THREE_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : ex1 {"Ex1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      ex2 {"Ex2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      ex3 {"Ex3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx1 {"Bx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx2 {"Bx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx3 {"Bx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx1 {"Jx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx2 {"Jx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx3 {"Jx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
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
void Meshblock<ONE_D>::verify(const SimulationParams& sim_params) {
  if (m_coord_system == CARTESIAN_COORD) {
    if (get_dx1() * 0.5 <= sim_params.get_timestep()) {
      throw std::logic_error("ERROR: timestep is too large (CFL not satisfied).");
    }
  } else {
    throw std::logic_error("ERROR: only cartesian coordinate system is available.");
  }
  for (auto& p : particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("ERROR: undefined particle pusher.");
    }
  }
}

template <>
void Meshblock<TWO_D>::verify(const SimulationParams& sim_params) {
  if (m_coord_system == CARTESIAN_COORD) {
    // uniform cartesian grid
    if (get_dx1() != get_dx2()) {
      throw std::logic_error("ERROR: unequal cell size on a cartesian grid.");
    }
    if (get_dx1() * 0.5 <= sim_params.get_timestep()) {
      throw std::logic_error("ERROR: timestep is too large (CFL not satisfied).");
    }
  } else {
    throw std::logic_error("ERROR: only cartesian coordinate system is available.");
  }
  for (auto& p : particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("ERROR: undefined particle pusher.");
    }
  }
}

template <>
void Meshblock<THREE_D>::verify(const SimulationParams& sim_params) {
  if (m_coord_system == CARTESIAN_COORD) {
    // uniform cartesian grid
    if ((get_dx1() != get_dx2()) || (get_dx2() != get_dx3())) {
      throw std::logic_error("ERROR: unequal cell size on a cartesian grid.");
    }
    if (get_dx1() * 0.5 <= sim_params.get_timestep()) {
      throw std::logic_error("ERROR: timestep is too large (CFL not satisfied).");
    }
  } else {
    throw std::logic_error("ERROR: only cartesian coordinate system is available.");
  }
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
