#include "global.h"
#include "meshblock.h"
#include "particles.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

  template <Dimension D>
  Mesh<D>::Mesh(const std::vector<unsigned int>& res, const std::vector<real_t>& ext, const real_t* params)
    : m_imin {res.size() > 0 ? N_GHOSTS : 0},
      m_imax {res.size() > 0 ? N_GHOSTS + (int)(res[0]) : 1},
      m_jmin {res.size() > 1 ? N_GHOSTS : 0},
      m_jmax {res.size() > 1 ? N_GHOSTS + (int)(res[1]) : 1},
      m_kmin {res.size() > 2 ? N_GHOSTS : 0},
      m_kmax {res.size() > 2 ? N_GHOSTS + (int)(res[2]) : 1},
      m_Ni {res.size() > 0 ? (int)(res[0]) : 1},
      m_Nj {res.size() > 1 ? (int)(res[1]) : 1},
      m_Nk {res.size() > 2 ? (int)(res[2]) : 1},
      metric{res, ext, params} {}

  template <>
  auto Mesh<Dimension::ONE_D>::loopAllCells() -> RangeND<Dimension::ONE_D> {
    return NTTRange<Dimension::ONE_D>({m_imin - N_GHOSTS}, {m_imax + N_GHOSTS});
  }
  template <>
  auto Mesh<Dimension::TWO_D>::loopAllCells() -> RangeND<Dimension::TWO_D> {
    return NTTRange<Dimension::TWO_D>({m_imin - N_GHOSTS, m_jmin - N_GHOSTS}, {m_imax + N_GHOSTS, m_jmax + N_GHOSTS});
  }
  template <>
  auto Mesh<Dimension::THREE_D>::loopAllCells() -> RangeND<Dimension::THREE_D> {
    return NTTRange<Dimension::THREE_D>({m_imin - N_GHOSTS, m_jmin - N_GHOSTS, m_kmin - N_GHOSTS},
                                        {m_imax + N_GHOSTS, m_jmax + N_GHOSTS, m_kmax + N_GHOSTS});
  }
  template <>
  auto Mesh<Dimension::ONE_D>::loopActiveCells() -> RangeND<Dimension::ONE_D> {
    return NTTRange<Dimension::ONE_D>({m_imin}, {m_imax});
  }
  template <>
  auto Mesh<Dimension::TWO_D>::loopActiveCells() -> RangeND<Dimension::TWO_D> {
    return NTTRange<Dimension::TWO_D>({m_imin, m_jmin}, {m_imax, m_jmax});
  }
  template <>
  auto Mesh<Dimension::THREE_D>::loopActiveCells() -> RangeND<Dimension::THREE_D> {
    return NTTRange<Dimension::THREE_D>({m_imin, m_jmin, m_kmin}, {m_imax, m_jmax, m_kmax});
  }

  template <Dimension D, SimulationType S>
  Meshblock<D, S>::Meshblock(const std::vector<unsigned int>& res,
                             const std::vector<real_t>& ext,
                             const real_t* params,
                             const std::vector<ParticleSpecies>& species)
    : Mesh<D>(res, ext, params), Fields<D, S>(res) {
    for (auto& part : species) {
      particles.emplace_back(part);
    }
  }
} // namespace ntt

template class ntt::Mesh<ntt::Dimension::ONE_D>;
template class ntt::Mesh<ntt::Dimension::TWO_D>;
template class ntt::Mesh<ntt::Dimension::THREE_D>;

#if SIMTYPE == PIC_SIMTYPE
template class ntt::Meshblock<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template class ntt::Meshblock<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template class ntt::Meshblock<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
#elif SIMTYPE == GRPIC_SIMTYPE
template class ntt::Meshblock<ntt::Dimension::TWO_D, ntt::SimulationType::GRPIC>;
template class ntt::Meshblock<ntt::Dimension::THREE_D, ntt::SimulationType::GRPIC>;
#endif