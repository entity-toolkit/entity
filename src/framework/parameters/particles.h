#ifndef FRAMEWORK_PARAMETERS_PARTICLES_H
#define FRAMEWORK_PARAMETERS_PARTICLES_H

#include "enums.h"

#include "utils/toml.h"

#include "framework/containers/species.h"
#include "framework/parameters/parameters.h"

namespace ntt {

  namespace params {

    auto GetParticleSpecies(SimulationParams*,
                            const SimEngine&,
                            spidx_t,
                            const toml::value&) -> ParticleSpecies;

  } // namespace params

} // namespace ntt

#endif
