#ifndef FRAMEWORK_SIM_PARAMS_H
#define FRAMEWORK_SIM_PARAMS_H

#include "wrapper.h"

#include "io/input.h"
#include "meshblock/particles.h"
#include "utils/utils.h"

#include <toml.hpp>

#include <vector>

namespace ntt {
  /**
   * @brief Storage class for user-defined & implied parameter values.
   */
  class SimulationParams {
    // User defined simulation title
    std::string                                 m_title;
    // User defined CFL
    real_t                                      m_cfl;
    // User enforced timestep
    real_t                                      m_dt;
    // User defined correction to the speed of light
    real_t                                      m_correction;
    // User defined total runtime in physical units
    real_t                                      m_total_runtime;
    // Independent simulation parameters.
    real_t                                      m_ppc0, m_larmor0, m_skindepth0;
    // Deduced simulation parameters.
    real_t                                      m_sigma0;
    // Vector of user-defined species parameters.
    std::vector<ParticleSpecies>                m_species;

    // GR specific
    real_t                                      m_gr_pusher_epsilon;
    int                                         m_gr_pusher_niter;

    // Use particle weights
    bool                                        m_use_weights;

    // Particle shuffle interval.
    int                                         m_shuffle_interval;
    double                                      m_max_dead_frac;

    // Enable/disable algorithms
    bool                                        m_enable_fieldsolver;
    bool                                        m_enable_deposit;

    // current filtering passes
    unsigned short                              m_current_filters;

    /**
     * Extent of the whole domain in physical units
     * { x1_min, x1_max, x2_min, x2_max, x3_min, x3_max }.
     *
     * @warning Size of the vector is 2*D (dimension).
     */
    std::vector<real_t>                         m_extent;
    // User-defined resolution.
    std::vector<unsigned int>                   m_resolution;
    // User-defined boundary conditions.
    std::vector<std::vector<BoundaryCondition>> m_boundaries;
    // User-defined metric.
    std::string                                 m_metric;
    std::string                                 m_coordinates;
    // User-defined real-valued parameters for the metric [10 max].
    real_t                                      m_metric_parameters[10];

    // Output parameters
    std::string                                 m_output_format;
    int                                         m_output_interval;
    real_t                                      m_output_interval_time;
    std::vector<std::string>                    m_output_fields;
    std::vector<std::string>                    m_output_particles;
    int                                         m_output_mom_smooth;
    std::size_t                                 m_output_prtl_stride;

    // Diagnostic parameters
    int                                         m_diag_interval;
    bool                                        m_blocking_timers;

    // Container with data from the parsed input file.
    toml::value                                 m_inputdata;

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
    [[nodiscard]] auto title() const -> const std::string& {
      return m_title;
    }
    /**
     * @brief Get the CFL.
     */
    [[nodiscard]] auto cfl() const -> const real_t& {
      return m_cfl;
    }
    /**
     * @brief Get the dt.
     */
    [[nodiscard]] auto dt() const -> const real_t& {
      return m_dt;
    }
    /**
     * @brief Get the correction to the speed of light.
     */
    [[nodiscard]] auto correction() const -> const real_t& {
      return m_correction;
    }
    /**
     * @brief Get the GR pusher epsilon.
     */
    [[nodiscard]] auto grPusherEpsilon() const -> const real_t& {
      return m_gr_pusher_epsilon;
    }
    /**
     * @brief Get the GR pusher niter.
     */
    [[nodiscard]] auto grPusherNiter() const -> const int& {
      return m_gr_pusher_niter;
    }
    /**
     * @brief Get the total runtime in physical units.
     */
    [[nodiscard]] auto totalRuntime() const -> const real_t& {
      return m_total_runtime;
    }
    /**
     * @brief Get the fiducial number of particles per cell.
     */
    [[nodiscard]] auto ppc0() const -> const real_t& {
      return m_ppc0;
    }
    /**
     * @brief Get the fiducial Larmor radius.
     */
    [[nodiscard]] auto larmor0() const -> const real_t& {
      return m_larmor0;
    }
    /**
     * @brief Get the fiducial skin depth.
     */
    [[nodiscard]] auto skindepth0() const -> const real_t& {
      return m_skindepth0;
    }
    /**
     * @brief Get the fiducial sigma.
     */
    [[nodiscard]] auto sigma0() const -> const real_t& {
      return m_sigma0;
    }
    /**
     * @brief Get the vector of user-defined species parameters.
     */
    [[nodiscard]] auto species() const -> const std::vector<ParticleSpecies>& {
      return m_species;
    }
    /**
     * @brief Get the particle shuffle interval.
     */
    [[nodiscard]] auto shuffleInterval() const -> const int& {
      return m_shuffle_interval;
    }
    /**
     * @brief Get maximum number of dead particles (as a fraction of current active particles).
     */
    [[nodiscard]] auto maxDeadFraction() const -> const double& {
      return m_max_dead_frac;
    }
    [[nodiscard]] auto useWeights() const -> const bool& {
      return m_use_weights;
    }
    /**
     * @brief Get the extent of the simulation box.
     */
    [[nodiscard]] auto extent() const -> const std::vector<real_t>& {
      return m_extent;
    }
    /**
     * @brief Get the resolution of the simulation box.
     */
    [[nodiscard]] auto resolution() const -> const std::vector<unsigned int>& {
      return m_resolution;
    }
    /**
     * @brief Get the boundary conditions of the simulation box.
     */
    [[nodiscard]] auto boundaries() const
      -> const std::vector<std::vector<BoundaryCondition>>& {
      return m_boundaries;
    }
    /**
     * @brief Get the metric label.
     */
    [[nodiscard]] auto metric() const -> const std::string& {
      return m_metric;
    }
    /**
     * @brief Get the coordinates label.
     */
    [[nodiscard]] auto coordinates() const -> const std::string& {
      return m_coordinates;
    }
    /**
     * @brief Get the metric parameters.
     */
    [[nodiscard]] auto metricParameters() const -> const real_t* {
      return m_metric_parameters;
    }
    /**
     * @brief Get the input params in toml format.
     */
    [[nodiscard]] auto inputdata() const -> const toml::value& {
      return m_inputdata;
    }
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
    [[nodiscard]] auto depositEnabled() const -> const bool& {
      return m_enable_deposit;
    }
    /**
     * @brief Get the output format.
     */
    [[nodiscard]] auto outputFormat() const -> const std::string& {
      return m_output_format;
    }
    /**
     * @brief Get the output interval.
     */
    [[nodiscard]] auto outputInterval() const -> const int& {
      return m_output_interval;
    }
    /**
     * @brief Get the output interval in physical time units.
     */
    [[nodiscard]] auto outputIntervalTime() const -> const real_t& {
      return m_output_interval_time;
    }
    /**
     * @brief Get output field labels.
     */
    [[nodiscard]] auto outputFields() const -> const std::vector<std::string>& {
      return m_output_fields;
    }
    /**
     * @brief Get output particles labels.
     */
    [[nodiscard]] auto outputParticles() const -> const std::vector<std::string>& {
      return m_output_particles;
    }
    /**
     * @brief Get the smoothing size for moments.
     */
    [[nodiscard]] auto outputMomSmooth() const -> const int& {
      return m_output_mom_smooth;
    }
    /**
     * @brief Get the particle stride for the output.
     */
    [[nodiscard]] auto outputPrtlStride() const -> const std::size_t& {
      return m_output_prtl_stride;
    }
    /**
     * @brief Get the diagnostic printout interval.
     */
    [[nodiscard]] auto diagInterval() const -> const int& {
      return m_diag_interval;
    }
    /**
     * @brief Get the blocking timers flag.
     */
    [[nodiscard]] auto blockingTimers() const -> const bool& {
      return m_blocking_timers;
    }

    /**
     * @brief Get parameters read from the input (with default value if not found)
     */
    template <typename T>
    [[nodiscard]] auto get(const std::string& block,
                           const std::string& key,
                           const T&           defval) const -> T {
      return readFromInput<T>(m_inputdata, block, key, defval);
    }

    /**
     * @brief Get parameters read from the input (no default)
     * @overload
     */
    template <typename T>
    [[nodiscard]] auto get(const std::string& block, const std::string& key) const -> T {
      return readFromInput<T>(m_inputdata, block, key);
    }

    /**
     * @brief Get parameters read from the input (with valid values, no default)
     * @overload
     */
    template <typename T>
    [[nodiscard]] auto get(const std::string&    block,
                           const std::string&    key,
                           const std::vector<T>& valid) const -> T {
      auto val = readFromInput<T>(m_inputdata, block, key);
      TestValidOption(val, valid);
      return val;
    }

    /**
     * @brief Get parameters read from the input (with valid values, and with default)
     * @overload
     */
    template <typename T>
    [[nodiscard]] auto get(const std::string&    block,
                           const std::string&    key,
                           const T&              defval,
                           const std::vector<T>& valid) const -> T {
      auto val = readFromInput<T>(m_inputdata, block, key, defval);
      TestValidOption(val, valid);
      return val;
    }
  };

}    // namespace ntt

#endif
