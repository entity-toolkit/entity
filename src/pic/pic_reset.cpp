#include "global.h"
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

template struct ntt::PIC<ntt::Dim1>;
template struct ntt::PIC<ntt::Dim2>;
template struct ntt::PIC<ntt::Dim3>;
