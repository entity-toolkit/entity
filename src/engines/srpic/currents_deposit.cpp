#include "kernels/currents_deposit.hpp"

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/srpic/srpic.h"
#include "framework/domain/domain.h"
#include "framework/parameters.h"

#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <map>
#include <string>

namespace ntt {

  template <class M>
  void SRPICEngine<M>::CurrentsDeposit(domain_t& domain) {
    logger::Checkpoint("Launching currents deposit kernel", HERE);
    auto scatter_cur = Kokkos::Experimental::create_scatter_view(domain.fields.cur);
    for (auto& species : domain.species) {
      const auto npart = species.npart();
      if (npart == 0 || cmp::AlmostZero(species.charge())) {
        continue;
      }
      Kokkos::parallel_for("CurrentsDeposit",
                           species.rangeActiveParticles(),
                           kernel::DepositCurrents_kernel<SimEngine::SRPIC, M>(
                             scatter_cur,
                             species.i1,
                             species.i2,
                             species.i3,
                             species.i1_prev,
                             species.i2_prev,
                             species.i3_prev,
                             species.dx1,
                             species.dx2,
                             species.dx3,
                             species.dx1_prev,
                             species.dx2_prev,
                             species.dx3_prev,
                             species.ux1,
                             species.ux2,
                             species.ux3,
                             species.phi,
                             species.weight,
                             species.tag,
                             domain.mesh.metric,
                             (real_t)(species.charge()),
                             dt));
    }

    Kokkos::Experimental::contribute(domain.fields.cur, scatter_cur);
  }
  template class SRPICEngine<metric::Minkowski<Dim::_1D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_2D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_3D>>;
  template class SRPICEngine<metric::Spherical<Dim::_2D>>;
  template class SRPICEngine<metric::QSpherical<Dim::_2D>>;

} // namespace ntt
