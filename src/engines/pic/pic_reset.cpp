#include "wrapper.h"

#include "pic.h"
#include "simulation.h"

#include "io/output.h"

namespace ntt {
  /**
   * @brief reset simulation.
   *
   */
  template <Dimension D>
  void PIC<D>::ResetSimulation() {
    this->m_tstep = 0;
    this->m_time  = ZERO;
    ResetFields();
    ResetParticles();
    ResetCurrents();
    InitializeSetup();
    this->Communicate(Comm_E | Comm_B);
    FieldsBoundaryConditions();
    NTTLog();
  }

  /**
   * @brief reset fields.
   *
   */
  template <Dimension D>
  void PIC<D>::ResetParticles() {
    auto& mblock = this->meshblock;
    for (auto& species : mblock.particles) {
      Kokkos::deep_copy(species.i1, 0);
      Kokkos::deep_copy(species.dx1, ZERO);
      if constexpr (D != Dim1) {
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

      Kokkos::deep_copy(species.tag, 0);
      Kokkos::deep_copy(species.weight, ZERO);

      species.setNpart(0);
    }
  }

  /**
   * @brief reset fields.
   *
   */
  template <Dimension D>
  void PIC<D>::ResetFields() {
    auto& mblock = this->meshblock;
    Kokkos::deep_copy(mblock.em, ZERO);
    Kokkos::deep_copy(mblock.bckp, ZERO);
  }

  /**
   * @brief reset currents.
   *
   */
  template <Dimension D>
  void PIC<D>::ResetCurrents() {
    auto& mblock = this->meshblock;
    Kokkos::deep_copy(mblock.cur, ZERO);
    Kokkos::deep_copy(mblock.buff, ZERO);
  }
}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::ResetParticles();
template void ntt::PIC<ntt::Dim2>::ResetParticles();
template void ntt::PIC<ntt::Dim3>::ResetParticles();

template void ntt::PIC<ntt::Dim1>::ResetFields();
template void ntt::PIC<ntt::Dim2>::ResetFields();
template void ntt::PIC<ntt::Dim3>::ResetFields();

template void ntt::PIC<ntt::Dim1>::ResetCurrents();
template void ntt::PIC<ntt::Dim2>::ResetCurrents();
template void ntt::PIC<ntt::Dim3>::ResetCurrents();

template void ntt::PIC<ntt::Dim1>::ResetSimulation();
template void ntt::PIC<ntt::Dim2>::ResetSimulation();
template void ntt::PIC<ntt::Dim3>::ResetSimulation();