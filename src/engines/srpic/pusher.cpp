#include "enums.h"
#include "global.h"

#include "arch/traits.h"
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

#include <map>
#include <string>

namespace ntt {
  template <class M>
  using pgen_t = user::PGen<SimEngine::SRPIC, M>;

  template <class M, bool ExtFSp>
  using basekernel_t = kernel::sr::PusherBase_kernel<M, pgen_t<M>, ExtFSp>;

  namespace {

    template <class M, bool ExtFSp, typename p, typename... cs>
    void dispatch_cooling(const std::map<std::string, real_t>& coeffs_map,
                          basekernel_t<M, ExtFSp>&             base,
                          std::size_t                          npart) {
      Kokkos::parallel_for("ParticlePusher",
                           Kokkos::RangePolicy<AccelExeSpace, p>(0, npart),
                           kernel::sr::Pusher_kernel<M, pgen_t<M>, ExtFSp, p, cs...>(
                             base,
                             coeffs_map.at("gca_larmor_max"),
                             coeffs_map.at("gca_eovrb_max"),
                             coeffs_map.at("sync_coeff")));
    }

    template <class M, bool ExtFSp, typename p>
    void dispatch_pusher(const std::map<std::string, real_t>& coeffs_map,
                         basekernel_t<M, ExtFSp>&             base,
                         std::size_t                          npart,
                         Cooling                              cooling) {
      if (cooling == Cooling::SYNCHROTRON) {
        dispatch_cooling<M, ExtFSp, p, kernel::sr::Synchrotron_t>(coeffs_map,
                                                                  base,
                                                                  npart);
      } else if (cooling == Cooling::NONE) {
        dispatch_cooling<M, ExtFSp, p, kernel::sr::NoCooling_t>(coeffs_map,
                                                                base,
                                                                npart);
      } else {
        raise::Error("Unknown cooling model", HERE);
      }
    }

    template <class M, bool ExtFSp>
    void dispatch(const std::map<std::string, real_t>& coeffs_map,
                  basekernel_t<M, ExtFSp>&             base,
                  std::size_t                          npart,
                  Cooling                              cooling,
                  PrtlPusher                           pusher) {
      if (pusher == PrtlPusher::BORIS) {
        dispatch_pusher<M, ExtFSp, kernel::sr::Boris_t>(coeffs_map, base, npart, cooling);
      } else if (pusher == PrtlPusher::VAY) {
        dispatch_pusher<M, ExtFSp, kernel::sr::Vay_t>(coeffs_map, base, npart, cooling);
      } else if (pusher == PrtlPusher::BORIS_GCA) {
        dispatch_pusher<M, ExtFSp, kernel::sr::Boris_GCA_t>(coeffs_map,
                                                            base,
                                                            npart,
                                                            cooling);
      } else if (pusher == PrtlPusher::VAY_GCA) {
        dispatch_pusher<M, ExtFSp, kernel::sr::Vay_GCA_t>(coeffs_map,
                                                          base,
                                                          npart,
                                                          cooling);
      } else if (pusher == PrtlPusher::PHOTON) {
        dispatch_pusher<M, ExtFSp, kernel::sr::Photon_t>(coeffs_map,
                                                         base,
                                                         npart,
                                                         cooling);
      } else {
        raise::Error("Unknown particle pusher", HERE);
      }
    }
  } // namespace

  template <class M>
  void SRPICEngine<M>::ParticlePush(Domain<SimEngine::SRPIC, M>& domain) {
    logger::Checkpoint("Launching particle pusher kernel", HERE);
    for (auto& species : domain.species) {
      const auto npart   = species.npart();
      // if (npart == 0) {
      //   continue;
      // }
      const auto q_ovr_m = species.mass() > ZERO
                             ? species.charge() / species.mass()
                             : ZERO;
      //  coeff = q / m (dt / 2) omegaB0
      const auto coeff   = q_ovr_m * HALF * dt *
                         m_params.template get<real_t>("scales.omegaB0");
      const auto pusher  = species.pusher();
      const auto cooling = species.cooling();

      // coefficients to be forwarded to the dispatcher
      // gca
      const auto has_gca = (pusher == PrtlPusher::VAY_GCA) ||
                           (pusher == PrtlPusher::BORIS_GCA);
      const auto gca_larmor_max = has_gca ? m_params.template get<real_t>(
                                              "algorithms.gca.larmor_max")
                                          : ZERO;
      const auto gca_eovrb_max  = has_gca ? m_params.template get<real_t>(
                                             "algorithms.gca.e_ovr_b_max")
                                          : ZERO;
      // cooling
      const auto has_cooling    = (cooling == Cooling::SYNCHROTRON);
      const auto sync_grad      = has_cooling ? m_params.template get<real_t>(
                                             "algorithms.synchrotron.gamma_rad")
                                              : ZERO;
      const auto sync_coeff     = has_cooling ? (real_t)(0.1) * dt *
                                              m_params.template get<real_t>(
                                                "scales.omegaB0") /
                                              (SQR(sync_grad) * species.mass())
                                              : ZERO;

      const auto coeff_map = std::map<std::string, real_t> {
        {"gca_larmor_max", gca_larmor_max},
        { "gca_eovrb_max",  gca_eovrb_max},
        {    "sync_coeff",     sync_coeff}
      };

      // bool apply_extforce = traits::has_member<traits::ext_force_t, pgen_t<M>>::value;
      // toggle to indicate whether the ext force applies to current species
      // toggle to indicate whether pgen defines the external force
      bool apply_extforce = false;
      if constexpr (traits::has_member<traits::pgen::ext_force_t, pgen_t<M>>::value) {
        apply_extforce = true;
        if (traits::has_member<traits::species_t, decltype(pgen_t<M>::ext_force)>::value) {
          // species is a vector of unsigned int, check that species.index() is in that vector
          apply_extforce &= std::find(m_pgen.ext_force.species.begin(),
                                      m_pgen.ext_force.species.end(),
                                      species.index()) !=
                            m_pgen.ext_force.species.end();
        }
      }

      if (apply_extforce) {
        std::cout << "Applying external force for " << species.index() << std::endl;
        auto pusher_base = basekernel_t<M, true>(domain.fields.em,
                                                 species.index(),
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
        dispatch(coeff_map, pusher_base, npart, cooling, pusher);
      } else {
        std::cout << "Not applying external force for " << species.index()
                  << std::endl;
        auto pusher_base = basekernel_t<M, false>(domain.fields.em,
                                                  species.index(),
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
        dispatch(coeff_map, pusher_base, npart, cooling, pusher);
      }
    }
  }

  template class SRPICEngine<metric::Minkowski<Dim::_1D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_2D>>;
  template class SRPICEngine<metric::Minkowski<Dim::_3D>>;
  template class SRPICEngine<metric::Spherical<Dim::_2D>>;
  template class SRPICEngine<metric::QSpherical<Dim::_2D>>;

} // namespace ntt