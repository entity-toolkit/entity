#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/srpic/srpic.h"
#include "framework/domain/domain.h"

#include "kernels/ampere_mink.hpp"
#include "kernels/ampere_sr.hpp"

namespace ntt {

  template <class M>
  void SRPICEngine<M>::Ampere(Domain<SimEngine::SRPIC, M>& domain, real_t fraction) {
    logger::Checkpoint("Launching Ampere kernel", HERE);
    const auto dT = fraction *
                    m_params.template get<real_t>(
                      "algorithms.timestep.correction") *
                    dt;
    if constexpr (M::CoordType == Coord::Cart) {
      // minkowski case
      const auto dx = math::sqrt(domain.mesh.metric.template h_<1, 1>({}));
      real_t     coeff1, coeff2;
      if constexpr (M::Dim == Dim::_2D) {
        coeff1 = dT / SQR(dx);
        coeff2 = dT;
      } else {
        coeff1 = dT / dx;
        coeff2 = ZERO;
      }

      Kokkos::parallel_for(
        "Ampere",
        domain.mesh.rangeActiveCells(),
        kernel::mink::Ampere_kernel<M::Dim>(domain.fields.em, coeff1, coeff2));
    } else {
      range_t<M::Dim> range {};
      if constexpr (M::Dim == Dim::_2D) {
        range = CreateRangePolicy<Dim::_2D>(
          { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
          { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
      }
      const auto ni2 = domain.mesh.n_active(in::x2);
      Kokkos::parallel_for("Ampere",
                           range,
                           kernel::sr::Ampere_kernel<M>(domain.fields.em,
                                                        domain.mesh.metric,
                                                        dT,
                                                        ni2,
                                                        domain.mesh.flds_bc()));
    }
  }

  template class SRPICEngine<metric::Minkowski<Dim::_1D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_2D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_3D>>;
  template class SRPICEngine<metric::Spherical<Dim::_2D>>;
  template class SRPICEngine<metric::QSpherical<Dim::_2D>>;

} // namespace ntt