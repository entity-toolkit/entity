/**
 * @file currents_deposit.cpp
 * @brief Atomic current deposition for all charged particles.
 * @implements: `CurrentsDeposit` method of the `GRPIC` class
 * @includes: `currents_deposit.hpp
 * @depends: `grpic.h`
 *
 * @notes: - The deposited currents are not the "physical" currents used ...
 *           ... in the Ampere's law, they need to be converted further.
 *
 */

#include "currents_deposit.hpp"

#include "wrapper.h"

#include "grpic.h"

#include "io/output.h"

namespace ntt {
  template <Dimension D>
  void GRPIC<D>::CurrentsDeposit() {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());

    Kokkos::deep_copy(mblock.cur0, ZERO);
    auto scatter_cur0 = Kokkos::Experimental::create_scatter_view(mblock.cur0);
    for (auto& species : mblock.particles) {
      if (species.charge() != 0.0) {
        const real_t dt { mblock.timestep() };
        const real_t charge { species.charge() };
        Kokkos::parallel_for("CurrentsDeposit",
                             species.rangeActiveParticles(),
                             DepositCurrents_kernel<D>(
                               mblock, species, scatter_cur0, charge, params.useWeights(), dt));
      }
    }
    Kokkos::Experimental::contribute(mblock.cur0, scatter_cur0);

    NTTLog();
  }
}    // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::CurrentsDeposit();
template void ntt::GRPIC<ntt::Dim3>::CurrentsDeposit();
