#include "global.h"
#include "meshblock.h"
#include "particles.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

  template <Dimension D>
  Mesh<D>::Mesh(std::vector<std::size_t> res)
    : i_min {res.size() > 0 ? N_GHOSTS : 0},
      i_max {res.size() > 0 ? N_GHOSTS + (int)(res[0]) : 1},
      j_min {res.size() > 1 ? N_GHOSTS : 0},
      j_max {res.size() > 1 ? N_GHOSTS + (int)(res[1]) : 1},
      k_min {res.size() > 2 ? N_GHOSTS : 0},
      k_max {res.size() > 2 ? N_GHOSTS + (int)(res[2]) : 1},
      Ni {res.size() > 0 ? res[0] : 1},
      Nj {res.size() > 1 ? res[1] : 1},
      Nk {res.size() > 2 ? res[2] : 1} {}

  template <>
  auto Mesh<Dimension::ONE_D>::loopAllCells() -> RangeND<Dimension::ONE_D> {
    return NTTRange<Dimension::ONE_D>({i_min - N_GHOSTS}, {i_max + N_GHOSTS});
  }
  template <>
  auto Mesh<Dimension::TWO_D>::loopAllCells() -> RangeND<Dimension::TWO_D> {
    return NTTRange<Dimension::TWO_D>({i_min - N_GHOSTS, j_min - N_GHOSTS}, {i_max + N_GHOSTS, j_max + N_GHOSTS});
  }
  template <>
  auto Mesh<Dimension::THREE_D>::loopAllCells() -> RangeND<Dimension::THREE_D> {
    return NTTRange<Dimension::THREE_D>({i_min - N_GHOSTS, j_min - N_GHOSTS, k_min - N_GHOSTS},
                                        {i_max + N_GHOSTS, j_max + N_GHOSTS, k_max + N_GHOSTS});
  }
  template <>
  auto Mesh<Dimension::ONE_D>::loopActiveCells() -> RangeND<Dimension::ONE_D> {
    return NTTRange<Dimension::ONE_D>({i_min}, {i_max});
  }
  template <>
  auto Mesh<Dimension::TWO_D>::loopActiveCells() -> RangeND<Dimension::TWO_D> {
    return NTTRange<Dimension::TWO_D>({i_min, j_min}, {i_max, j_max});
  }
  template <>
  auto Mesh<Dimension::THREE_D>::loopActiveCells() -> RangeND<Dimension::THREE_D> {
    return NTTRange<Dimension::THREE_D>({i_min, j_min, k_min}, {i_max, j_max, k_max});
  }

  template <Dimension D, SimulationType S>
  Meshblock<D, S>::Meshblock(const std::vector<std::size_t>& res, const std::vector<ParticleSpecies>& parts)
    : Mesh<D>(res), Fields<D, S>(res) {
    for (auto& part : parts) {
      particles.emplace_back(part);
    }
  }

  //   template <Dimension D>
  //   void Meshblock<D>::verify(const SimulationParams&) {
  //     if ((this->Ni == 1) ||
  //        ((this->Nj > 1) && (static_cast<short>(D) < 2)) ||
  //        ((this->Nk > 1) && (static_cast<short>(D) < 3))) {
  //       throw std::logic_error("# Error: wrong dimension inferred in Meshblock.");
  //     }
  //     for (auto& p : particles) {
  //       if (p.get_pusher() == UNDEFINED_PUSHER) {
  //         throw std::logic_error("# Error: undefined particle pusher.");
  //       }
  //     }
  //   }

} // namespace ntt

template class ntt::Mesh<ntt::Dimension::ONE_D>;
template class ntt::Mesh<ntt::Dimension::TWO_D>;
template class ntt::Mesh<ntt::Dimension::THREE_D>;

template class ntt::Meshblock<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template class ntt::Meshblock<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template class ntt::Meshblock<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
