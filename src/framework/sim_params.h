#ifndef FRAMEWORK_SIM_PARAMS_H
#define FRAMEWORK_SIM_PARAMS_H

#include "global.h"
#include "particles.h"

#include <toml/toml.hpp>

#include <vector>

namespace ntt {
  /**
   * @brief Storage class for user-defined & implied parameter values.
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
    real_t m_sigma0;
    // Vector of user-defined species parameters.
    std::vector<ParticleSpecies> m_species;

    // Enable/disable algorithms
    bool m_enable_fieldsolver;
    bool m_enable_deposit;

    // current filtering passes
    unsigned short m_current_filters;

    /**
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

    // Output parameters
    std::string m_output_format;
    int         m_output_interval;

    // Container with data from the parsed input file.
    toml::value m_inputdata;

  public:
    /**
     * @brief Constructor for simulation parameters class.
     * @param inputdata toml-object with parsed toml parameters.
     * @param dim Dimension.
     */
    SimulationParams(const toml::value&, Dimension);
    ~SimulationParams() = default;

    /**
     * @brief Get the simulation title.
     */
    [[nodiscard]] auto title() const -> const std::string& { return m_title; }
    /**
     * @brief Get the CFL.
     */
    [[nodiscard]] auto cfl() const -> const real_t& { return m_cfl; }
    /**
     * @brief Get the correction to the speed of light.
     */
    [[nodiscard]] auto correction() const -> const real_t& { return m_correction; }
    /**
     * @brief Get the total runtime in physical units.
     */
    [[nodiscard]] auto totalRuntime() const -> const real_t& { return m_total_runtime; }
    /**
     * @brief Get the fiducial number of particles per cell.
     */
    [[nodiscard]] auto ppc0() const -> const real_t& { return m_ppc0; }
    /**
     * @brief Get the fiducial Larmor radius.
     */
    [[nodiscard]] auto larmor0() const -> const real_t& { return m_larmor0; }
    /**
     * @brief Get the fiducial skin depth.
     */
    [[nodiscard]] auto skindepth0() const -> const real_t& { return m_skindepth0; }
    /**
     * @brief Get the fiducial sigma.
     */
    [[nodiscard]] auto sigma0() const -> const real_t& { return m_sigma0; }
    /**
     * @brief Get the vector of user-defined species parameters.
     */
    [[nodiscard]] auto species() const -> const std::vector<ParticleSpecies>& {
      return m_species;
    }
    /**
     * @brief Get the extent of the simulation box.
     */
    [[nodiscard]] auto extent() const -> const std::vector<real_t>& { return m_extent; }
    /**
     * @brief Get the resolution of the simulation box.
     */
    [[nodiscard]] auto resolution() const -> const std::vector<unsigned int>& {
      return m_resolution;
    }
    /**
     * @brief Get the boundary conditions of the simulation box.
     */
    [[nodiscard]] auto boundaries() const -> const std::vector<BoundaryCondition>& {
      return m_boundaries;
    }
    /**
     * @brief Get the metric.
     */
    [[nodiscard]] auto metric() const -> const std::string& { return m_metric; }
    /**
     * @brief Get the metric parameters.
     */
    [[nodiscard]] auto metricParameters() const -> const real_t* {
      return m_metric_parameters;
    }
    /**
     * @brief Get the input params in toml format.
     */
    [[nodiscard]] auto inputdata() const -> const toml::value& { return m_inputdata; }
    /**
     * @brief Get the enable_fieldsolver flag.
     */
    [[nodiscard]] auto fieldsolverEnabled() const -> const bool& {
      return m_enable_fieldsolver;
    }
    /**
     * @brief number of current filter passes
     */
    [[nodiscard]] auto currentFilters() const -> const unsigned short& {
      return m_current_filters;
    }
    /**
     * @brief Get the enable_deposit flag.
     */
    [[nodiscard]] auto depositEnabled() const -> const bool& { return m_enable_deposit; }
    /**
     * @brief Get the output format.
     */
    [[nodiscard]] auto outputFormat() const -> const std::string& { return m_output_format; }
    /**
     * @brief Get the output interval.
     */
    [[nodiscard]] auto outputInterval() const -> const int& { return m_output_interval; }
  };

} // namespace ntt

#endif
