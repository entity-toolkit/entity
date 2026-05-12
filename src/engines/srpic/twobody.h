#ifndef ENGINES_SRPIC_TWOBODY_H
#define ENGINES_SRPIC_TWOBODY_H

#include "enums.h"

#include "traits/metric.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/param_container.h"

#include "archetypes/qed/compton.h"
#include "framework/domain/domain.h"
#include "framework/parameters/extra.h"
#include "framework/parameters/parameters.h"
#include "kernels/twobody_interactions.hpp"

namespace ntt {
  namespace srpic {

    template <CartesianMetricClass M>
    void TwoBodyInteractions(Domain<SimEngine::SRPIC, M>& domain,
                             const prm::Parameters&       engine_params,
                             const SimulationParams&      params) {
      logger::Checkpoint("Launching TwoBodyInteractions routines", HERE);
      const auto dt   = engine_params.get<real_t>("dt");
      const auto step = engine_params.get<timestep_t>("step");
      for (const auto& interaction :
           params.template get<std::vector<::ntt::params::TwoBodyInteractionParams>>(
             "two_body.interactions")) {
        if (step % interaction.interval == 0u) {
          if (interaction.type == TwoBodyInteraction::COMPTON) {

            printf("CALLING COMPTON\n");
          } else if (interaction.type == TwoBodyInteraction::CUSTOM) {
            raise::Error("Custom two-body interactions not implemented yet", HERE);
          }
        }
      }
    }
  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_TWOBODY_H