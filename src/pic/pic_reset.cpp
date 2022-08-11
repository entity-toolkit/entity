#include "global.h"
#include "pic.h"
#include "pic_reset.hpp"

namespace ntt {
  /**
   * @brief reset fields.
   *
   */
  template <Dimension D>
  void PIC<D>::resetParticles(const real_t&) {
    for (auto& species : this->m_mblock.particles) {
      ResetParticles<D> reset_particles(this->m_mblock, species);
      reset_particles.resetParticles();
    }
  }

  /**
   * @brief reset fields.
   *
   */
  template <Dimension D>
  void PIC<D>::resetFields(const real_t&) {
    Kokkos::parallel_for(
      "reset_fields", this->m_mblock.loopActiveCells(), ResetFields<D>(this->m_mblock));
  }

  /**
   * @brief reset currents.
   *
   */
  template <Dimension D>
  void PIC<D>::resetCurrents(const real_t&) {
    Kokkos::parallel_for(
      "reset_currents", this->m_mblock.loopAllCells(), ResetCurrents<D>(this->m_mblock));
  }
} // namespace ntt

template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;
