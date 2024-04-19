#include "enums.h"
#include "global.h"

#include "utils/log.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/srpic/srpic.h"
#include "framework/domain/domain.h"

#include "kernels/particle_pusher_sr.hpp"
#include "pgen.hpp"

namespace ntt {

  template <class M>
  void SRPICEngine<M>::ParticlePush(Domain<SimEngine::SRPIC, M>& domain) {
    logger::Checkpoint("Launching particle pusher kernel", HERE);
    for (auto& species : domain.species) {
      const auto charge_ovr_mass = species.mass() > ZERO
                                     ? species.charge() / species.mass()
                                     : ZERO;
      //  coeff = q / m (dt / 2) omegaB0
      const auto coeff           = charge_ovr_mass * HALF * dt *
                         m_params.template get<real_t>("scales.omegaB0");
      auto pusher_base =
        kernel::sr::PusherBase_kernel<M, user::PGen<SimEngine::SRPIC, M>>(
          domain.fields.em,
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
          species.tag,
          domain.mesh.metric,
          m_pgen,
          time,
          coeff,
          dt,
          domain.mesh.n_active(in::x1),
          domain.mesh.n_active(in::x2),
          domain.mesh.n_active(in::x3),
          domain.mesh.prtl_bc());
      // template <class M, class PG, typename P, bool ExtForce, typename... Cs>
      // struct Pusher_kernel : public PusherBase_kernel<M, PG> {
    }
  }

  template class SRPICEngine<metric::Minkowski<Dim::_1D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_2D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_3D>>;
  template class SRPICEngine<metric::Spherical<Dim::_2D>>;
  template class SRPICEngine<metric::QSpherical<Dim::_2D>>;

} // namespace ntt

// const ndfield_t<D, 6>&      EB,
// array_t<int*>&              i1,
// array_t<int*>&              i2,
// array_t<int*>&              i3,
// array_t<int*>&              i1_prev,
// array_t<int*>&              i2_prev,
// array_t<int*>&              i3_prev,
// array_t<prtldx_t*>&         dx1,
// array_t<prtldx_t*>&         dx2,
// array_t<prtldx_t*>&         dx3,
// array_t<prtldx_t*>&         dx1_prev,
// array_t<prtldx_t*>&         dx2_prev,
// array_t<prtldx_t*>&         dx3_prev,
// array_t<real_t*>&           ux1,
// array_t<real_t*>&           ux2,
// array_t<real_t*>&           ux3,
// array_t<real_t*>&           phi,
// array_t<short*>&            tag,
// const M&                    metric,
// const PG&                   pgen,
// real_t                      time,
// real_t                      coeff,
// real_t                      dt,
// int                         ni1,
// int                         ni2,
// int                         ni3,
// const boundaries_t<PrtlBC>& boundaries) :