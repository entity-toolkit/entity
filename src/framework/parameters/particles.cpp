#include "framework/parameters/particles.h"

#include "defaults.h"
#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/toml.h"

#include "framework/containers/species.h"
#include "framework/parameters/parameters.h"

#include <string>

namespace ntt {

  namespace params {

    /*
     * Auxiliary functions
     */
    auto getRadiativeDragFlags(
      const std::string& radiative_drag_str) -> RadiativeDragFlags {
      if (fmt::toLower(radiative_drag_str) == "none") {
        return RadiativeDrag::NONE;
      } else {
        // separate comas
        RadiativeDragFlags flags = RadiativeDrag::NONE;
        std::string        token;
        std::istringstream tokenStream(radiative_drag_str);
        while (std::getline(tokenStream, token, ',')) {
          const auto token_lower = fmt::toLower(token);
          if (token_lower == "synchrotron") {
            flags |= RadiativeDrag::SYNCHROTRON;
          } else if (token_lower == "compton") {
            flags |= RadiativeDrag::COMPTON;
          } else {
            raise::Error(fmt::format("Invalid radiative_drag value: %s",
                                     radiative_drag_str.c_str()),
                         HERE);
          }
        }
        return flags;
      }
    }

    auto getPusherFlags(
      const std::string& particle_pusher_str) -> ParticlePusherFlags {
      if (fmt::toLower(particle_pusher_str) == "none") {
        return ParticlePusher::NONE;
      } else {
        // separate comas
        ParticlePusherFlags flags = ParticlePusher::NONE;
        std::string         token;
        std::istringstream  tokenStream(particle_pusher_str);
        while (std::getline(tokenStream, token, ',')) {
          const auto token_lower = fmt::toLower(token);
          if (token_lower == "photon") {
            flags |= ParticlePusher::PHOTON;
          } else if (token_lower == "boris") {
            flags |= ParticlePusher::BORIS;
          } else if (token_lower == "vay") {
            flags |= ParticlePusher::VAY;
          } else if (token_lower == "gca") {
            flags |= ParticlePusher::GCA;
          } else {
            raise::Error(
              fmt::format("Invalid pusher value: %s", particle_pusher_str.c_str()),
              HERE);
          }
        }
        if (flags & ParticlePusher::PHOTON and flags & ParticlePusher::GCA) {
          raise::Error("Photon pusher cannot be used with GCA", HERE);
        }
        return flags;
      }
    }

    auto GetParticleSpecies(SimulationParams*  params,
                            const SimEngine&   engine_enum,
                            spidx_t            idx,
                            const toml::value& sp) -> ParticleSpecies {
      const auto label  = toml::find_or<std::string>(sp,
                                                    "label",
                                                    "s" + std::to_string(idx));
      const auto mass   = toml::find<float>(sp, "mass");
      const auto charge = toml::find<float>(sp, "charge");
      raise::ErrorIf((charge != 0.0f) && (mass == 0.0f),
                     "mass of the charged species must be non-zero",
                     HERE);
      const auto is_massless   = (mass == 0.0f) && (charge == 0.0f);
      const auto def_pusher    = (is_massless ? defaults::ph_pusher
                                              : defaults::em_pusher);
      const auto maxnpart_real = toml::find<double>(sp, "maxnpart");
      const auto maxnpart      = static_cast<npart_t>(maxnpart_real);
      auto pusher_str = toml::find_or(sp, "pusher", std::string(def_pusher));
      const auto npayloads_real = toml::find_or(sp,
                                                "n_payloads_real",
                                                static_cast<unsigned short>(0));
      const auto use_tracking   = toml::find_or(sp, "tracking", false);
      auto       npayloads_int  = toml::find_or(sp,
                                         "n_payloads_int",
                                         static_cast<unsigned short>(0));
      if (use_tracking) {
#if !defined(MPI_ENABLED)
        npayloads_int += 1;
#else
        npayloads_int += 2;
#endif
      }
      const auto radiative_drag_str = toml::find_or(sp,
                                                    "radiative_drag",
                                                    std::string("none"));
      raise::ErrorIf((fmt::toLower(radiative_drag_str) != "none") && is_massless,
                     "radiative drag is only applicable to massive particles",
                     HERE);
      raise::ErrorIf((fmt::toLower(pusher_str) == "photon") && !is_massless,
                     "photon pusher is only applicable to massless particles",
                     HERE);
      const auto particle_pusher_flags = getPusherFlags(pusher_str);
      const auto radiative_drag_flags = getRadiativeDragFlags(radiative_drag_str);
      if (radiative_drag_flags & RadiativeDrag::SYNCHROTRON) {
        raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                       "Synchrotron radiative drag is only supported for SRPIC",
                       HERE);
        params->promiseToDefine("algorithms.synchrotron.gamma_rad");
      }
      if (radiative_drag_flags & RadiativeDrag::COMPTON) {
        raise::ErrorIf(
          engine_enum != SimEngine::SRPIC,
          "Inverse Compton radiative drag is only supported for SRPIC",
          HERE);
        params->promiseToDefine("algorithms.compton.gamma_rad");
      }
      if (particle_pusher_flags & ParticlePusher::GCA) {
        raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                       "GCA pushers are only supported for SRPIC",
                       HERE);
        params->promiseToDefine("algorithms.gca.e_ovr_b_max");
        params->promiseToDefine("algorithms.gca.larmor_max");
      }
      return ParticleSpecies(idx,
                             label,
                             mass,
                             charge,
                             maxnpart,
                             particle_pusher_flags,
                             use_tracking,
                             radiative_drag_flags,
                             npayloads_real,
                             npayloads_int);
    }

  } // namespace params

} // namespace ntt
