#ifndef SANDBOX_SANDBOX_H
#define SANDBOX_SANDBOX_H

#include "wrapper.h"

#include "simulation.h"

#include <toml.hpp>

namespace ntt {
  /**
   * @brief Class for SANDBOX simulations, inherits from `Simulation<D, SANDBOXEngine>`.
   * @tparam D dimension.
   */
  template <Dimension D>
  struct SANDBOX : public Simulation<D, SANDBOXEngine> {
    /**
     * @brief Constructor for SANDBOX class.
     * @param inputdata toml-object with parsed toml parameters.
     */
    SANDBOX(const toml::value& inputdata) : Simulation<D, SANDBOXEngine>(inputdata) {}
    // SANDBOX(const SANDBOX<D>&) = delete;
    ~SANDBOX()                 = default;

    /**
     * @brief Run the simulation (calling initialize, verify, mainloop, etc).
     */
    void Run();
  };

}    // namespace ntt

#endif
