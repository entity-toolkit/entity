#include "global.h"
#include "pic.h"
#include "pic_currents_deposit.hpp"

namespace ntt {
  template <Dimension D>
  void PIC<D>::depositCurrentsSubstep(const real_t&) {
    for (auto& species : this->m_mblock.particles) {
      if (species.charge() != 0.0) {
        const real_t dt {this->m_mblock.timestep()};
        Deposit<D>   deposit(this->m_mblock, species, 1.0, dt);
        deposit.depositCurrents();
      }
    }
  }

} // namespace ntt

template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;