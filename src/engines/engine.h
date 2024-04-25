/**
 * @file engines/engine.h
 * @brief Base simulation class which just initializes the metadomain
 * @implements
 *   - ntt::Engine<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - pgen.hpp
 *   - arch/traits.h
 *   - arch/directions.h
 *   - arch/mpi_aliases.h
 *   - utils/error.h
 *   - utils/log.h
 *   - utils/formatting.h
 *   - utils/progressbar.h
 *   - utils/colors.h
 *   - utils/timer.h
 *   - archetypes/field_setter.h
 *   - framework/containers/fields.h
 *   - framework/containers/particles.h
 *   - framework/containers/species.h
 *   - framework/domain/metadomain.h
 *   - framework/parameters.h
 * @cpp:
 *   - engine_init.cpp
 *   - engine_printer.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI
 *   - DEBUG
 *   - OUTPUT_ENABLED
 *   - GPU_ENABLED
 */

#ifndef ENGINES_ENGINE_H
#define ENGINES_ENGINE_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/progressbar.h"
#include "utils/timer.h"

#include "framework/containers/fields.h"
#include "framework/containers/particles.h"
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
      traits::check_compatibility<S>::value(user::PGen<S, M>::engines) &&
      traits::check_compatibility<M::MetricType>::value(user::PGen<S, M>::metrics) &&
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
    void print_step_report(timer::Timers&, pbar::DurationHistory&, bool) const;

    virtual void step_forward(timer::Timers&, Domain<S, M>&) = 0;

    void run() {
      init();

      auto timers = timer::Timers {
        { "FieldSolver",
         "CurrentFiltering", "CurrentDeposit",
         "ParticlePusher", "FieldBoundaries",
         "ParticleBoundaries", "Communications",
         "Custom", "Output" },
        []() {
          Kokkos::fence();
         },
        m_params.get<bool>("diagnostics.blocking_timers")
      };
      const auto diag_interval = m_params.get<std::size_t>(
        "diagnostics.interval");

#if defined(OUTPUT_ENABLED)
      const auto interval = m_params.template get<std::size_t>(
        "output.interval");
      const auto interval_time = m_params.template get<long double>(
        "output.interval_time");
      long double last_output_time = -interval_time - 1.0;
      const auto  should_output =
        [&interval, &interval_time, &last_output_time](auto step, auto time) {
          return ((interval_time <= 0.0) and (step % interval == 0)) or
                 ((time - last_output_time >= interval_time) and
                  (interval_time > 0.0));
        };
#endif

      auto time_history = pbar::DurationHistory { 1000 };

      // main algorithm loop
      while (step < max_steps) {
        // run the engine-dependent algorithm step
        m_metadomain.runOnLocalDomains([&timers, this](auto& dom) {
          step_forward(timers, dom);
        });

        m_metadomain.runOnLocalDomains([&timers, this](auto& dom) {
          timers.start("Custom");
          if (traits::has_member<traits::pgen::custom_poststep_t, user::PGen<S, M>>::value) {
            m_pgen.CustomPostStep(step, time, dom);
          }
          timers.stop("Custom");
        });

        // advance time & timestep
        ++step;
        time += dt;

        auto print_output = false;
#if defined(OUTPUT_ENABLED)
        // write timestep if needed
        if (should_output(step, time)) {
          timers.start("Output");
          m_metadomain.Write(m_params, step, time);
          timers.stop("Output");
          last_output_time = time;
          print_output     = true;
        }
#endif

        // advance time_history
        time_history.tick();
        // print final timestep report
        if (diag_interval > 0 and step % diag_interval == 0) {
          print_step_report(timers, time_history, print_output);
        }
        timers.resetAll();
      }
    }
  };

} // namespace ntt

#endif // ENGINES_ENGINE_H