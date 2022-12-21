#include "deposit_currents.hpp"

#include "pic.h"
#include "wrapper.h"

namespace ntt {
  template <Dimension D>
  void PIC<D>::CurrentsDeposit() {
    auto& mblock      = this->meshblock;
    auto  scatter_cur = Kokkos::Experimental::create_scatter_view(mblock.cur);
    for (auto& species : mblock.particles) {
      if (species.charge() != 0.0) {
        const real_t              dt { mblock.timestep() };
        const real_t              charge { species.charge() };
        DepositCurrents_kernel<D> deposit(mblock, species, scatter_cur, charge, dt);
        Kokkos::parallel_for("deposit", species.rangeActiveParticles(), deposit);
      }
    }
    Kokkos::Experimental::contribute(mblock.cur, scatter_cur);
    PLOGD << "... ... currents filter substep finished";
  }

}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::CurrentsDeposit();
template void ntt::PIC<ntt::Dim2>::CurrentsDeposit();
template void ntt::PIC<ntt::Dim3>::CurrentsDeposit();