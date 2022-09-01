#include "global.h"
#include "pic.h"
#include "pic_currents_deposit.hpp"

namespace ntt {
  template <Dimension D>
  void PIC<D>::depositCurrentsSubstep(const real_t&) {
    auto scatter_cur = Kokkos::Experimental::create_scatter_view(this->m_mblock.cur);
    for (auto& species : this->m_mblock.particles) {
      if (species.charge() != 0.0) {
        const real_t dt {this->m_mblock.timestep()};
        const real_t charge {species.charge()};
        Deposit<D>   deposit(this->m_mblock, species, scatter_cur, charge, dt);
        deposit.depositCurrents();
      }
    }
    Kokkos::Experimental::contribute(this->m_mblock.cur, scatter_cur);
  }

} // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;
