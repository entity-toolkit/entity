/**
 * @file currents_deposit.cpp
 * @brief Atomic current deposition for all charged particles.
 * @implements: `CurrentsDeposit` method of the `PIC` class
 * @includes: `currents_deposit.hpp
 * @depends: `pic.h`
 *
 * @notes: - The deposited currents are not the "physical" currents used ...
 *           ... in the Ampere's law, they need to be converted further.
 *         - Previous coordinate of the particle is recovered from its velocity.
 *
 */

#include "currents_deposit.hpp"

#include "wrapper.h"

#include "io/output.h"
#include "pic.h"

namespace ntt {
  template <Dimension D>
  void PIC<D>::CurrentsDeposit() {
    auto& mblock = this->meshblock;
    auto  params = *(this->params());

    auto scatter_cur = Kokkos::Experimental::create_scatter_view(mblock.cur);
    for (auto& species : mblock.particles) {
      if (species.charge() != 0.0) {
        const real_t              dt { mblock.timestep() };
        const real_t              charge { species.charge() };
        DepositCurrents_kernel<D> deposit(
          mblock, species, scatter_cur, charge, params.useWeights(), dt);
        Kokkos::parallel_for("CurrentsDeposit", species.rangeActiveParticles(), deposit);
      }
    }
    Kokkos::Experimental::contribute(mblock.cur, scatter_cur);

    NTTLog();
  }
}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::CurrentsDeposit();
template void ntt::PIC<ntt::Dim2>::CurrentsDeposit();
template void ntt::PIC<ntt::Dim3>::CurrentsDeposit();