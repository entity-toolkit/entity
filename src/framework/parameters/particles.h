/**
 * @file framework/parameters/particles.h
 * @brief Auxiliary functions for reading in particle species parameters
 * @implements
 *   - ntt::params::GetParticleSpecies -> ParticleSpecies
 * @cpp:
 *   - particles.cpp
 * @namespaces:
 *   - ntt::params::
 */
#ifndef FRAMEWORK_PARAMETERS_PARTICLES_H
#define FRAMEWORK_PARAMETERS_PARTICLES_H

#include "enums.h"

#include <toml11/toml.hpp>

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
