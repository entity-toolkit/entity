#include "global.h"
#include "meshblock.h"

#include <cassert>

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
  for (auto& part : parts) {
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
  for (auto& part : parts) {
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
  for (auto& part : parts) {
    particles.emplace_back(part);
  }
}

auto loopActiveCells(const Meshblock<One_D>& mblock) -> ntt_1drange_t {
  return NTT1DRange(static_cast<range_t>(mblock.get_imin()),
                    static_cast<range_t>(mblock.get_imax()));
}
auto loopActiveCells(const Meshblock<Two_D>& mblock) -> ntt_2drange_t {
  return NTT2DRange({mblock.get_imin(), mblock.get_jmin()}, {mblock.get_imax(), mblock.get_jmax()});
}
auto loopActiveCells(const Meshblock<Three_D>& mblock) -> ntt_3drange_t {
  return NTT3DRange({mblock.get_imin(), mblock.get_jmin(), mblock.get_kmin()},
                    {mblock.get_imax(), mblock.get_jmax(), mblock.get_kmax()});
}

} // namespace ntt

template struct ntt::Meshblock<ntt::One_D>;
template struct ntt::Meshblock<ntt::Two_D>;
template struct ntt::Meshblock<ntt::Three_D>;
