/**
 * @file deposit_currents.cpp
 * @brief Atomic current deposition for all charged particles.
 * @implements: `CurrentsDeposit` method of the `PIC` class
 * @includes: `deposit_currents.hpp
 * @depends: `pic.h`
 *
 * @notes: - The deposited currents are not the "physical" currents used ...
 *           ... in the Ampere's law, they need to be converted further.
 *         - Previous coordinate of the particle is recovered from its velocity.
 *
 */

#include "deposit_currents.hpp"

#include "wrapper.h"

#include "fields.h"
#include "pic.h"

namespace ntt {
  template <Dimension D>
  void PIC<D>::CurrentsDeposit() {
    auto& mblock = this->meshblock;

    AssertEmptyContent(mblock.cur_content);

    auto scatter_cur = Kokkos::Experimental::create_scatter_view(mblock.cur);
    for (auto& species : mblock.particles) {
      if (species.charge() != 0.0) {
        const real_t              dt { mblock.timestep() };
        const real_t              charge { species.charge() };
        DepositCurrents_kernel<D> deposit(mblock, species, scatter_cur, charge, dt);
        Kokkos::parallel_for("deposit", species.rangeActiveParticles(), deposit);
      }
    }
    Kokkos::Experimental::contribute(mblock.cur, scatter_cur);

    ImposeContent(mblock.cur_content,
                  { Content::jx1_curly, Content::jx2_curly, Content::jx3_curly });

    PLOGD << "... ... currents filter substep finished";
  }
}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::CurrentsDeposit();
template void ntt::PIC<ntt::Dim2>::CurrentsDeposit();
template void ntt::PIC<ntt::Dim3>::CurrentsDeposit();