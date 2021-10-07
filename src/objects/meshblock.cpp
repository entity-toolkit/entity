#include "global.h"
#include "meshblock.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

template <Dimension D>
Meshblock<D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : m_resolution{res} {
  for (auto& part : parts) {
    particles.emplace_back(part);
  }
}

template <Dimension D>
void Meshblock<D>::printDetails() {
  if (particles.size() > 0) {
    PLOGI << "[particles]";
    for (std::size_t i{0}; i < particles.size(); ++i) {
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

Meshblock1D::Meshblock1D(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : Meshblock<ONE_D>{res, parts},
      ex1{"Ex1", res[0] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS} {}

Meshblock2D::Meshblock2D(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : Meshblock<TWO_D>{res, parts},
      ex1{"Ex1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {}

Meshblock3D::Meshblock3D(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : Meshblock<THREE_D>{res, parts},
      ex1{"Ex1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {}

auto loopActiveCells(const Meshblock1D& mblock) -> ntt_1drange_t {
  return NTT1DRange(static_cast<range_t>(mblock.get_imin()),
                    static_cast<range_t>(mblock.get_imax()));
}
auto loopActiveCells(const Meshblock2D& mblock) -> ntt_2drange_t {
  return NTT2DRange({mblock.get_imin(), mblock.get_jmin()}, {mblock.get_imax(), mblock.get_jmax()});
}
auto loopActiveCells(const Meshblock3D& mblock) -> ntt_3drange_t {
  return NTT3DRange({mblock.get_imin(), mblock.get_jmin(), mblock.get_kmin()},
                    {mblock.get_imax(), mblock.get_jmax(), mblock.get_kmax()});
}

void Meshblock1D::verify(const SimulationParams& sim_params) {
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

void Meshblock2D::verify(const SimulationParams& sim_params) {
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

void Meshblock3D::verify(const SimulationParams& sim_params) {
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
