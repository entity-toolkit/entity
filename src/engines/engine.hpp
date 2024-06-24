/**
 * @file engines/engine.hpp
 * @brief Base simulation class which just initializes the metadomain
 * @implements
 *   - ntt::Engine<>
 * @cpp:
 *   - engine_init.cpp
 *   - engine_printer.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI
 *   - DEBUG
 *   - OUTPUT_ENABLED
 *   - CUDA_ENABLED
 */

#ifndef ENGINES_ENGINE_H
#define ENGINES_ENGINE_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/progressbar.h"
#include "utils/timer.h"

#include "framework/containers/species.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"

#include "pgen.hpp"

#include <Kokkos_Core.hpp>

#include <map>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  class Engine {
    static_assert(M::is_metric, "template arg for Engine class has to be a metric");
    static_assert(user::PGen<S, M>::is_pgen, "unrecognized problem generator");

  protected:
    SimulationParams& m_params;
    Metadomain<S, M>  m_metadomain;
    user::PGen<S, M>  m_pgen;

    const long double runtime;
    const real_t      dt;
    const std::size_t max_steps;
    long double       time { 0.0 };
    std::size_t       step { 0 };

  public:
    static constexpr bool pgen_is_ok {
      traits::check_compatibility<S>::value(user::PGen<S, M>::engines) and
      traits::check_compatibility<M::MetricType>::value(user::PGen<S, M>::metrics) and
      traits::check_compatibility<M::Dim>::value(user::PGen<S, M>::dimensions)
    };

    static constexpr Dimension D { M::Dim };
    static constexpr bool      is_engine { true };

#if defined(OUTPUT_ENABLED)
    Engine(SimulationParams& params)
      : m_params { params }
      , m_metadomain { params.get<unsigned int>("simulation.domain.number"),
                       params.get<std::vector<int>>(
                         "simulation.domain.decomposition"),
                       params.get<std::vector<std::size_t>>("grid.resolution"),
                       params.get<boundaries_t<real_t>>("grid.extent"),
                       params.get<boundaries_t<FldsBC>>(
                         "grid.boundaries.fields"),
                       params.get<boundaries_t<PrtlBC>>(
                         "grid.boundaries.particles"),
                       params.get<std::map<std::string, real_t>>(
                         "grid.metric.params"),
                       params.get<std::vector<ParticleSpecies>>(
                         "particles.species"),
                       params.template get<std::string>("output.format") }
      , m_pgen { m_params, m_metadomain }
      , runtime { params.get<long double>("simulation.runtime") }
      , dt { params.get<real_t>("algorithms.timestep.dt") }
      , max_steps { static_cast<std::size_t>(runtime / dt) }

#else // not OUTPUT_ENABLED
    Engine(SimulationParams& params)
      : m_params { params }
      , m_metadomain { params.get<unsigned int>("simulation.domain.number"),
                       params.get<std::vector<int>>(
                         "simulation.domain.decomposition"),
                       params.get<std::vector<std::size_t>>("grid.resolution"),
                       params.get<boundaries_t<real_t>>("grid.extent"),
                       params.get<boundaries_t<FldsBC>>(
                         "grid.boundaries.fields"),
                       params.get<boundaries_t<PrtlBC>>(
                         "grid.boundaries.particles"),
                       params.get<std::map<std::string, real_t>>(
                         "grid.metric.params"),
                       params.get<std::vector<ParticleSpecies>>(
                         "particles.species") }
      , m_pgen { m_params, m_metadomain }
      , runtime { params.get<long double>("simulation.runtime") }
      , dt { params.get<real_t>("algorithms.timestep.dt") }
      , max_steps { static_cast<std::size_t>(runtime / dt) }
#endif
    {

      raise::ErrorIf(not pgen_is_ok, "Problem generator is not compatible with the picked engine/metric/dimension", HERE);
      print_report();
    }

    ~Engine() = default;

    void init();
    void print_report() const;
    void print_step_report(timer::Timers&, pbar::DurationHistory&, bool, bool) const;

    virtual void step_forward(timer::Timers&, Domain<S, M>&) = 0;

    void run();
  };

} // namespace ntt

#endif // ENGINES_ENGINE_H
