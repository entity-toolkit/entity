/**
 * @file currents_deposit.cpp
 * @brief Atomic current deposition for all charged particles.
 * @implements: `CurrentsDeposit` method of the `PIC` class
 * @includes: `utils/currents_deposit.hpp
 * @depends: `pic.h`
 *
 * @notes: - The deposited currents are not the "physical" currents used ...
 *           ... in the Ampere's law, they need to be converted further.
 *         - Previous coordinate of the particle is stored in _prev arrays.
 */

#include "utils/currents_deposit.hpp"

#include "wrapper.h"

#include "pic.h"

#include "io/output.h"

namespace ntt {
  template <Dimension D>
  void PIC<D>::CurrentsDeposit() {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());

    Kokkos::deep_copy(mblock.cur, ZERO);
    auto scatter_cur = Kokkos::Experimental::create_scatter_view(mblock.cur);
    for (auto& species : mblock.particles) {
      if (species.npart() == 0 || species.charge() == 0.0) {
        continue;
      }
      const real_t dt { mblock.timestep() };
      const real_t charge { species.charge() };
      Kokkos::parallel_for(
        "CurrentsDeposit",
        species.rangeActiveParticles(),
        DepositCurrents_kernel<D, PICEngine>(mblock, species, scatter_cur, charge, dt));
    }
    Kokkos::Experimental::contribute(mblock.cur, scatter_cur);

    NTTLog();
  }
} // namespace ntt

template void ntt::PIC<ntt::Dim1>::CurrentsDeposit();
template void ntt::PIC<ntt::Dim2>::CurrentsDeposit();
template void ntt::PIC<ntt::Dim3>::CurrentsDeposit();