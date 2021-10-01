#include "global.h"
#include "meshblock.h"

namespace ntt {

template <>
Meshblock<One_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : ex1{"Ex1", res[0] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS},
      m_resolution{res} {
  for (auto &part : parts) {
    particles.emplace_back(part);
  }
}

template <>
Meshblock<Two_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : ex1{"Ex1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      m_resolution{res} {
  for (auto &part : parts) {
    particles.emplace_back(part);
  }
}

template <>
Meshblock<Three_D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : ex1{"Ex1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      m_resolution{res} {
  for (auto &part : parts) {
    particles.emplace_back(part);
  }
}

auto loopActiveCells(const Meshblock<One_D> &mblock) -> NTT1DRange {
  return NTT1DRange({mblock.get_imin()}, {mblock.get_imax()});
}
auto loopActiveCells(const Meshblock<Two_D> &mblock) -> NTT2DRange {
  return NTT2DRange({mblock.get_imin(), mblock.get_jmin()}, {mblock.get_imax(), mblock.get_jmax()});
}
auto loopActiveCells(const Meshblock<Three_D> &mblock) -> NTT3DRange {
  return NTT3DRange({mblock.get_imin(), mblock.get_jmin(), mblock.get_kmin()},
                    {mblock.get_imax(), mblock.get_jmax(), mblock.get_kmax()});
}

} // namespace ntt

template class ntt::Meshblock<ntt::One_D>;
template class ntt::Meshblock<ntt::Two_D>;
template class ntt::Meshblock<ntt::Three_D>;
