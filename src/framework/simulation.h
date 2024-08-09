/**
 * @file framework/simulation.h
 * @brief Simulation class which creates and calles the engines
 * @implements
 *   - ntt::Simulation
 * @cpp:
 *   - simulation.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - DEBUG
 */

#ifndef FRAMEWORK_SIMULATION_H
#define FRAMEWORK_SIMULATION_H

#include "enums.h"

#include "arch/traits.h"
#include "utils/error.h"

#include "framework/parameters.h"

#include <stdexcept>
#include <type_traits>

namespace ntt {

  class Simulation {
    SimulationParams params;

  public:
    Simulation(int argc, char* argv[]);
    ~Simulation();

    template <template <class> class E, template <Dimension> class M, Dimension D>
    inline void run() {
      using engine_t = E<M<D>>;
      static_assert(engine_t::is_engine,
                    "template arg for Simulation::run has to be an engine");
      static_assert(traits::has_method<traits::run_t, engine_t>::value,
                    "Engine must contain a ::run() method");
      try {
        engine_t engine { params };
        engine.run();
      } catch (const std::exception& e) {
        raise::Fatal(e.what(), HERE);
      }
    }

    [[nodiscard]]
    inline auto requested_dimension() const -> Dimension {
      return params.get<Dimension>("grid.dim");
    }

    [[nodiscard]]
    inline auto requested_engine() const -> SimEngine {
      return params.get<SimEngine>("simulation.engine");
    }

    [[nodiscard]]
    inline auto requested_metric() const -> Metric {
      return params.get<Metric>("grid.metric.metric");
    }
  };

} // namespace ntt

#endif // FRAMEWORK_SIMULATION_H
