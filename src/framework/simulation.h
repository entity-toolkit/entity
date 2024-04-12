/**
 * @file framework/simulation.h
 * @brief Simulation class which creates and calles the engines
 * @implements
 *   - ntt::Simulation
 * @depends:
 *   - defaults.h
 *   - global.h
 *   - utils/error.h
 *   - utils/formatting.h
 *   - utils/plog.h
 *   - framework/io/cargs.h
 *   - framework/parameters.h
 * @cpp:
 *   - simulation.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - DEBUG
 */

#ifndef FRAMEWORK_SIMULATION_H
#define FRAMEWORK_SIMULATION_H

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

    template <typename E>
    void run() {
      static_assert(std::is_member_function_pointer<decltype(&E::run)>::value,
                    "Engine must contain a ::run() method");
      try {
        const E engine { params };
        engine.run();
      } catch (const std::exception& e) {
        raise::Fatal(e.what(), HERE);
      }
    }
  };

} // namespace ntt

#endif // FRAMEWORK_SIMULATION_H