#include "wrapper.h"
#include "pic.h"
#include "pic_reset.hpp"

namespace ntt {
  /**
   * @brief reset fields.
   *
   */
  template <Dimension D>
  void PIC<D>::ResetParticles() {
    auto& mblock = this->meshblock;
    for (auto& species : mblock.particles) {
      Kokkos::parallel_for("reset_particles",
                           species.rangeAllParticles(),
                           ResetParticles_kernel<D>(mblock, species));
    }
  }

  /**
   * @brief reset fields.
   *
   */
  template <Dimension D>
  void PIC<D>::ResetFields() {
    auto& mblock = this->meshblock;
    Kokkos::parallel_for(
      "reset_fields", mblock.rangeAllCells(), ResetFields_kernel<D>(mblock));
  }

  /**
   * @brief reset currents.
   *
   */
  template <Dimension D>
  void PIC<D>::ResetCurrents() {
    auto& mblock = this->meshblock;
    Kokkos::parallel_for(
      "reset_currents", mblock.rangeAllCells(), ResetCurrents_kernel<D>(mblock));
  }
} // namespace ntt

template void ntt::PIC<ntt::Dim1>::ResetParticles();
template void ntt::PIC<ntt::Dim2>::ResetParticles();
template void ntt::PIC<ntt::Dim3>::ResetParticles();

template void ntt::PIC<ntt::Dim1>::ResetFields();
template void ntt::PIC<ntt::Dim2>::ResetFields();
template void ntt::PIC<ntt::Dim3>::ResetFields();

template void ntt::PIC<ntt::Dim1>::ResetCurrents();
template void ntt::PIC<ntt::Dim2>::ResetCurrents();
template void ntt::PIC<ntt::Dim3>::ResetCurrents();