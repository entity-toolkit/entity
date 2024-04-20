#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "metrics/minkowski.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/srpic/srpic.h"
#include "framework/domain/domain.h"
#include "framework/parameters.h"

#include "kernels/particle_pusher_sr.hpp"
#include "pgen.hpp"

namespace ntt {
  template <class M>
  using pgen_t = user::PGen<SimEngine::SRPIC, M>;

  template <class M>
  using basekernel_t = kernel::sr::PusherBase_kernel<M, pgen_t<M>>;

  namespace {
    // clang-format off
    template <class M, bool ef, typename p, typename... cs>
    void dispatch_cooling(const SimulationParams&, real_t, real_t,
                          basekernel_t<M>&, std::size_t);

    template <class M, bool ef, typename p>
    void dispatch_pusher(const SimulationParams&, real_t, real_t,
                         basekernel_t<M>&, std::size_t, Cooling);

    template <class M, bool ef>
    void dispatch_extforce(const SimulationParams&, real_t, real_t,
                           basekernel_t<M>&, std::size_t, Cooling, PrtlPusher);

    template <class M>
    void dispatch(const SimulationParams&, real_t, real_t,
                  basekernel_t<M>&, std::size_t, bool, Cooling, PrtlPusher);
    // clang-format on
  } // namespace

  template <class M>
  void SRPICEngine<M>::ParticlePush(Domain<SimEngine::SRPIC, M>& domain) {
    logger::Checkpoint("Launching particle pusher kernel", HERE);
    for (auto& species : domain.species) {
      const auto npart = species.npart();
      if (npart == 0) {
        continue;
      }
      const auto q_ovr_m = species.mass() > ZERO
                             ? species.charge() / species.mass()
                             : ZERO;
      //  coeff = q / m (dt / 2) omegaB0
      const auto coeff   = q_ovr_m * HALF * dt *
                         m_params.template get<real_t>("scales.omegaB0");
      const auto pusher    = species.pusher();
      const auto cooling   = species.cooling();
      const auto ext_force = m_params.template get<bool>(
        "algorithms.toggles.extforce");
      // clang-format off
      auto pusher_base = basekernel_t<M>(
          domain.fields.em,
          species.i1, species.i2, species.i3,
          species.i1_prev, species.i2_prev, species.i3_prev,
          species.dx1, species.dx2, species.dx3,
          species.dx1_prev, species.dx2_prev, species.dx3_prev,
          species.ux1, species.ux2, species.ux3,
          species.phi, species.tag,
          domain.mesh.metric, m_pgen,
          time, coeff, dt,
          domain.mesh.n_active(in::x1), domain.mesh.n_active(in::x2), domain.mesh.n_active(in::x3),
          domain.mesh.prtl_bc()
        );
        dispatch(m_params, q_ovr_m, coeff, pusher_base, npart, ext_force, cooling, pusher);
      // clang-format on
    }
  }

  namespace {

    template <class M, bool ef, typename p, typename... cs>
    void dispatch_cooling(const SimulationParams& params,
                          real_t                  q_ovr_m,
                          real_t                  coeff,
                          basekernel_t<M>&        base,
                          std::size_t             npart) {
      // gca
      const auto gca_larmor_max = params.template get<real_t>(
        "algorithms.gca.larmor_max");
      const auto gca_eovrb_max = params.template get<real_t>(
        "algorithms.gca.e_ovr_b_max");
      // cooling
      // const auto grad_sync = params.template get<real_t>(
      //   "algorithms.synchrotron.gamma_rad");
      Kokkos::parallel_for(
        "ParticlePusher",
        Kokkos::RangePolicy<AccelExeSpace, p>(0, npart),
        kernel::sr::Pusher_kernel<M, pgen_t<M>, p, ef, cs...>(base,
                                                              gca_larmor_max,
                                                              gca_eovrb_max,
                                                              ZERO));
    }

    template <class M, bool ef, typename p>
    void dispatch_pusher(const SimulationParams& params,
                         real_t                  q_ovr_m,
                         real_t                  coeff,
                         basekernel_t<M>&        base,
                         std::size_t             npart,
                         Cooling                 cooling) {
      if (cooling == Cooling::SYNCHROTRON) {
        dispatch_cooling<M, ef, p, kernel::sr::Synchrotron_t>(params,
                                                              q_ovr_m,
                                                              coeff,
                                                              base,
                                                              npart);
      } else if (cooling == Cooling::NONE) {
        dispatch_cooling<M, ef, p, kernel::sr::NoCooling_t>(params,
                                                            q_ovr_m,
                                                            coeff,
                                                            base,
                                                            npart);
      } else {
        raise::Error("Unknown cooling model", HERE);
      }
    }

    template <class M, bool ef>
    void dispatch_extforce(const SimulationParams& params,
                           real_t                  q_ovr_m,
                           real_t                  coeff,
                           basekernel_t<M>&        base,
                           std::size_t             npart,
                           Cooling                 cooling,
                           PrtlPusher              pusher) {
      if (pusher == PrtlPusher::BORIS) {
        dispatch_pusher<M, ef, kernel::sr::Boris_t>(params,
                                                    q_ovr_m,
                                                    coeff,
                                                    base,
                                                    npart,
                                                    cooling);
      } else if (pusher == PrtlPusher::VAY) {
        dispatch_pusher<M, ef, kernel::sr::Vay_t>(params,
                                                  q_ovr_m,
                                                  coeff,
                                                  base,
                                                  npart,
                                                  cooling);
      } else if (pusher == PrtlPusher::BORIS_GCA) {
        dispatch_pusher<M, ef, kernel::sr::Boris_GCA_t>(params,
                                                        q_ovr_m,
                                                        coeff,
                                                        base,
                                                        npart,
                                                        cooling);
      } else if (pusher == PrtlPusher::VAY_GCA) {
        dispatch_pusher<M, ef, kernel::sr::Vay_GCA_t>(params,
                                                      q_ovr_m,
                                                      coeff,
                                                      base,
                                                      npart,
                                                      cooling);
      } else if (pusher == PrtlPusher::PHOTON) {
        dispatch_pusher<M, ef, kernel::sr::Photon_t>(params,
                                                     q_ovr_m,
                                                     coeff,
                                                     base,
                                                     npart,
                                                     cooling);
      } else {
        raise::Error("Unknown particle pusher", HERE);
      }
    }

    template <class M>
    void dispatch(const SimulationParams& params,
                  real_t                  q_ovr_m,
                  real_t                  coeff,
                  basekernel_t<M>&        base,
                  std::size_t             npart,
                  bool                    enable_extforce,
                  Cooling                 cooling,
                  PrtlPusher              pusher) {
      if (enable_extforce) {
        dispatch_extforce<M, true>(params, q_ovr_m, coeff, base, npart, cooling, pusher);
      } else {
        dispatch_extforce<M, false>(params, q_ovr_m, coeff, base, npart, cooling, pusher);
      }
    }
  } // namespace

  template class SRPICEngine<metric::Minkowski<Dim::_1D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_2D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_3D>>;
  template class SRPICEngine<metric::Spherical<Dim::_2D>>;
  template class SRPICEngine<metric::QSpherical<Dim::_2D>>;

} // namespace ntt