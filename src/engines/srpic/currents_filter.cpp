#include "enums.h"
#include "global.h"

#include "utils/log.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/srpic/srpic.h"
#include "framework/domain/domain.h"

#include "kernels/faraday_mink.hpp"
#include "kernels/faraday_sr.hpp"

#include <Kokkos_Core.hpp>

namespace ntt {

  template <class M>
  void SRPICEngine<M>::Faraday(domain_t& domain, real_t fraction) {
    logger::Checkpoint("Launching Faraday kernel", HERE);
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
        "Faraday",
        domain.mesh.rangeActiveCells(),
        kernel::mink::Faraday_kernel<M::Dim>(domain.fields.em, coeff1, coeff2));
    } else {
      Kokkos::parallel_for("Faraday",
                           domain.mesh.rangeActiveCells(),
                           kernel::sr::Faraday_kernel<M>(domain.fields.em,
                                                         domain.mesh.metric,
                                                         dT,
                                                         domain.mesh.flds_bc()));
    }
  }

  template class SRPICEngine<metric::Minkowski<Dim::_1D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_2D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_3D>>;
  template class SRPICEngine<metric::Spherical<Dim::_2D>>;
  template class SRPICEngine<metric::QSpherical<Dim::_2D>>;

} // namespace ntt