#ifndef FRAMEWORK_SIM_PARAMS_H
#define FRAMEWORK_SIM_PARAMS_H

#include "global.h"
#include "particles.h"

#include <toml/toml.hpp>

#include <vector>

namespace ntt {
  /**
   * Storage class for user-defined & implied parameter values.
   */
  class SimulationParams {
    // User defined simualation title
    std::string m_title;
    // User defined CFL
    real_t m_cfl;
    // User defined correction to the speed of light
    real_t m_correction;
    // User defined total runtime in physical units
    real_t m_total_runtime;
    // Independent simulation parameters.
    real_t m_ppc0, m_larmor0, m_skindepth0;
    // Deduced simulation parameters.
    real_t m_sigma0, m_charge0, m_B0;
    // Vector of user-defined species parameters.
    std::vector<ParticleSpecies> m_species;
    bool                         m_enable_fieldsolver;
    /*
     * Extent of the whole domain in physical units
     * { x1_min, x1_max, x2_min, x2_max, x3_min, x3_max }.
     *
     * @warning Size of the vector is 2*D (dimension).
     */
    std::vector<real_t> m_extent;
    // User-defined resolution.
    std::vector<unsigned int> m_resolution;
    // User-defined boundary conditions.
    std::vector<BoundaryCondition> m_boundaries;
    // User-defined metric.
    std::string m_metric;
    // User-defined real-valued parameters for the metric [10 max].
    real_t m_metric_parameters[10];

    // Container with data from the parsed input file.
    toml::value m_inputdata;

  public:
    /**
     * Constructor for simulation parameters class.
     *
     * @param inputdata toml-object with parsed toml parameters.
     * @param dim Dimension.
     */
    SimulationParams(const toml::value& inputdata, Dimension dim);
    ~SimulationParams() = default;

    /**
     * Getters.
     */
    [[nodiscard]] auto title() const -> const std::string& { return m_title; }
    [[nodiscard]] auto cfl() const -> const real_t& { return m_cfl; }
    [[nodiscard]] auto correction() const -> const real_t& { return m_correction; }
    [[nodiscard]] auto total_runtime() const -> const real_t& { return m_total_runtime; }
    [[nodiscard]] auto ppc0() const -> const real_t& { return m_ppc0; }
    [[nodiscard]] auto larmor0() const -> const real_t& { return m_larmor0; }
    [[nodiscard]] auto skindepth0() const -> const real_t& { return m_skindepth0; }
    [[nodiscard]] auto sigma0() const -> const real_t& { return m_sigma0; }
    [[nodiscard]] auto charge0() const -> const real_t& { return m_charge0; }
    [[nodiscard]] auto B0() const -> const real_t& { return m_B0; }
    [[nodiscard]] auto species() const -> const std::vector<ParticleSpecies>& { return m_species; }
    [[nodiscard]] auto extent() const -> const std::vector<real_t>& { return m_extent; }
    [[nodiscard]] auto resolution() const -> const std::vector<unsigned int>& { return m_resolution; }
    [[nodiscard]] auto boundaries() const -> const std::vector<BoundaryCondition>& { return m_boundaries; }
    [[nodiscard]] auto metric() const -> const std::string& { return m_metric; }
    [[nodiscard]] auto metric_parameters() const -> const real_t* { return m_metric_parameters; }
    [[nodiscard]] auto inputdata() const -> const toml::value& { return m_inputdata; }
    [[nodiscard]] auto enable_fieldsolver() const -> const bool& { return m_enable_fieldsolver; }
  };

} // namespace ntt

#endif
