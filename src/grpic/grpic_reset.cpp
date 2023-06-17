#include "wrapper.h"

#include "grpic.h"
#include "simulation.h"

#include "io/output.h"

namespace ntt {
  template <Dimension D>
  void GRPIC<D>::ResetSimulation() {
    this->m_tstep = 0;
    this->m_time  = ZERO;
    ResetFields();
    ResetParticles();
    ResetCurrents();
    InitializeSetup();
    Exchange(GhostCells::fields);
    NTTLog();
  }

  template <Dimension D>
  void GRPIC<D>::ResetParticles() {
    auto& mblock = this->meshblock;
    for (auto& species : mblock.particles) {
      if constexpr ((D == Dim1) || (D == Dim2) || (D == Dim3)) {
        Kokkos::deep_copy(species.i1, 0);
        Kokkos::deep_copy(species.dx1, ZERO);
      }
      if constexpr ((D == Dim2) || (D == Dim3)) {
        Kokkos::deep_copy(species.i2, 0);
        Kokkos::deep_copy(species.dx2, ZERO);
#ifndef MINKOWSKI_METRIC
        Kokkos::deep_copy(species.phi, ZERO);
#endif
      }
      if constexpr (D == Dim3) {
        Kokkos::deep_copy(species.i3, 0);
        Kokkos::deep_copy(species.dx3, ZERO);
      }
      Kokkos::deep_copy(species.ux1, ZERO);
      Kokkos::deep_copy(species.ux2, ZERO);
      Kokkos::deep_copy(species.ux3, ZERO);
      species.setNpart(0);
    }
  }

  template <Dimension D>
  void GRPIC<D>::ResetFields() {
    auto& mblock = this->meshblock;
    Kokkos::deep_copy(mblock.em, ZERO);
    Kokkos::deep_copy(mblock.em0, ZERO);
    Kokkos::deep_copy(mblock.aux, ZERO);
  }

  template <Dimension D>
  void GRPIC<D>::ResetCurrents() {
    auto& mblock = this->meshblock;
    Kokkos::deep_copy(mblock.buff, ZERO);
    Kokkos::deep_copy(mblock.cur0, ZERO);
  }
}    // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::ResetSimulation();
template void ntt::GRPIC<ntt::Dim3>::ResetSimulation();

template void ntt::GRPIC<ntt::Dim2>::ResetParticles();
template void ntt::GRPIC<ntt::Dim3>::ResetParticles();

template void ntt::GRPIC<ntt::Dim2>::ResetFields();
template void ntt::GRPIC<ntt::Dim3>::ResetFields();

template void ntt::GRPIC<ntt::Dim2>::ResetCurrents();
template void ntt::GRPIC<ntt::Dim3>::ResetCurrents();