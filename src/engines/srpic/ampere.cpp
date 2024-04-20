#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/srpic/srpic.h"

#include "kernels/ampere_mink.hpp"
#include "kernels/ampere_sr.hpp"

#include <Kokkos_Core.hpp>

namespace ntt {

  template <class M>
  void SRPICEngine<M>::Ampere(domain_t& domain, real_t fraction) {
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

  template <class M>
  void SRPICEngine<M>::CurrentsAmpere(domain_t& domain) {
    logger::Checkpoint("Launching Ampere kernel for adding currents", HERE);
    const auto q0    = m_params.template get<real_t>("scales.q0");
    const auto n0    = m_params.template get<real_t>("scales.n0");
    const auto B0    = m_params.template get<real_t>("scales.B0");
    const auto coeff = -dt * q0 * n0 / B0;
    if constexpr (M::CoordType == Coord::Cart) {
      // minkowski case
      const auto V0 = m_params.template get<real_t>("scales.V0");

      Kokkos::parallel_for(
        "Ampere",
        domain.mesh.rangeActiveCells(),
        kernel::mink::CurrentsAmpere_kernel<M::Dim>(domain.fields.em,
                                                    domain.fields.cur,
                                                    coeff / V0,
                                                    ONE / n0));
    } else {
      range_t<M::Dim> range {};
      if constexpr (M::Dim == Dim::_2D) {
        range = CreateRangePolicy<Dim::_2D>(
          { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
          { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
      }
      const auto ni2 = domain.mesh.n_active(in::x2);
      Kokkos::parallel_for(
        "Ampere",
        range,
        kernel::sr::CurrentsAmpere_kernel<M>(domain.fields.em,
                                             domain.fields.cur,
                                             domain.mesh.metric,
                                             coeff,
                                             ONE / n0,
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