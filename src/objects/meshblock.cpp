#include "global.h"
#include "meshblock.h"

#include <cassert>

namespace ntt {

template<Dimension D>
Meshblock<D>::Meshblock(std::vector<std::size_t> res) : m_resolution{res} {}

Meshblock1D::Meshblock1D(std::vector<std::size_t> res)
    : Meshblock<ONE_D>{res},
      ex1{"Ex1", res[0] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS} {
  // for (auto& part : parts) {
  //   particles.emplace_back(part);
  // }
}

Meshblock2D::Meshblock2D(std::vector<std::size_t> res)
    : Meshblock<TWO_D>{res},
      ex1{"Ex1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {
  // for (auto& part : parts) {
  //   particles.emplace_back(part);
  // }
}

// Meshblock3D::Meshblock3D(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
Meshblock3D::Meshblock3D(std::vector<std::size_t> res)
    : Meshblock<THREE_D>{res},
      ex1{"Ex1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      ex2{"Ex2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      ex3{"Ex3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx1{"Bx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx2{"Bx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      bx3{"Bx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx1{"Jx1", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx2{"Jx2", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      jx3{"Jx3", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {
  // for (auto& part : parts) {
  //   particles.emplace_back(part);
  // }
}

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

template struct Meshblock<ONE_D>;
template struct Meshblock<TWO_D>;
template struct Meshblock<THREE_D>;

} // namespace ntt
