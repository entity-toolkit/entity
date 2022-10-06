#include "global.h"
#include "pic.h"
#include "deposit_currents.hpp"

namespace ntt {
  template <Dimension D>
  void PIC<D>::CurrentsDeposit() {
    auto& mblock      = this->meshblock;
    auto  scatter_cur = Kokkos::Experimental::create_scatter_view(mblock.cur);
    for (auto& species : mblock.particles) {
      if (species.charge() != 0.0) {
        const real_t              dt {mblock.timestep()};
        const real_t              charge {species.charge()};
        DepositCurrents_kernel<D> deposit(mblock, species, scatter_cur, charge, dt);
        deposit.apply();
      }
    }
    Kokkos::Experimental::contribute(mblock.cur, scatter_cur);
  }

} // namespace ntt

template struct ntt::PIC<ntt::Dim1>;
template struct ntt::PIC<ntt::Dim2>;
template struct ntt::PIC<ntt::Dim3>;
