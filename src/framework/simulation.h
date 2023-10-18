#ifndef FRAMEWORK_SIMULATION_H
#define FRAMEWORK_SIMULATION_H

#include "wrapper.h"

#include "sim_params.h"

#include "communications/metadomain.h"
#include "io/output.h"
#include "io/writer.h"
#include "meshblock/meshblock.h"
#include "utils/timer.h"

#include <toml.hpp>

#include <string>

namespace ntt {
  namespace {
    enum DiagFlags_ {
      DiagFlags_None     = 0,
      DiagFlags_Progress = 1 << 0,
      DiagFlags_Timers   = 1 << 1,
      DiagFlags_Species  = 2 << 2,
      DiagFlags_Default = DiagFlags_Progress | DiagFlags_Timers | DiagFlags_Species,
    };
  } // namespace

  typedef int DiagFlags;

  namespace {
    enum CommTags_ {
      Comm_None = 0,
      Comm_E    = 1 << 0,
      Comm_B    = 1 << 1,
      Comm_J    = 1 << 2,
      Comm_Prtl = 1 << 3,
      Comm_D    = 1 << 4,
      Comm_D0   = 1 << 5,
      Comm_B0   = 1 << 6,
      Comm_H    = 1 << 7,
    };
  } // namespace

  typedef int CommTags;

  /**
   * @brief Main class of the simulation containing all the necessary methods
   * and configurations.
   * @tparam D dimension.
   * @tparam S simulation engine.
   */
  template <Dimension D, SimulationEngine S>
  class Simulation {
  protected:
    // user-defined and inferred simulation parameters
    SimulationParams m_params;
    // time in physical units
    real_t           m_time { 0.0 };
    // time in iteration timesteps
    std::size_t      m_tstep { 0 };

    Metadomain<D> m_metadomain;

  public:
    static inline constexpr SimulationEngine engine { S };
    // meshblock with all the fields / metric / and particles
    Meshblock<D, S>                          meshblock;
    // writer
    Writer<D, S>                             writer;
    // random number pool
    RandomNumberPool_t                       random_pool;

    /**
     * @brief Constructor for simulation class.
     * @param inputdata toml-object with parsed toml parameters.
     */
    Simulation(const toml::value& inputdata);
    ~Simulation();

    /* -------------------------------------------------------------------------- */
    /*                                Main routines */
    /* -------------------------------------------------------------------------- */

    /**
     * @brief Verify that all the specified parameters are compatible before
     * beginning the simulation.
     */
    void Verify();

    /**
     * @brief Print all the simulation details using `plog`.
     */
    void PrintDetails();

    /**
     * @brief Diagnostic logging.
     * @param os output stream.
     */
    void PrintDiagnostics(const std::size_t&        step,
                          const real_t&             time,
                          const timer::Timers&      timer,
                          std::vector<long double>& tstep_durations,
                          const DiagFlags           diag_flags,
                          std::ostream&             os = std::cout);

    /* -------------------------------------------------------------------------- */
    /*                       Inter-meshblock communications */
    /* -------------------------------------------------------------------------- */

    /**
     * @brief Synchronize ghost zones between the meshblocks.
     * @param f tags identifying what quantities are synchronized.
     */
    void Communicate(CommTags comm);
    /**
     * @brief Synchronize currents between the blocks by accumulating.
     */
    void CurrentsSynchronize();

    /* -------------------------------------------------------------------------- */
    /*                                   Getters */
    /* -------------------------------------------------------------------------- */
    /**
     * @brief Get pointer to `m_params`.
     */
    [[nodiscard]]
    auto params() -> SimulationParams* {
      return &m_params;
    }

    [[nodiscard]]
    auto metadomain() -> Metadomain<D>* {
      return &m_metadomain;
    }

    /**
     * @brief Get the physical time
     */
    [[nodiscard]]
    auto time() const -> real_t {
      return m_time;
    }

    /**
     * @brief Get the current timestep
     */
    [[nodiscard]]
    auto tstep() const -> std::size_t {
      return m_tstep;
    }

    /**
     * @brief Loop over all active cells (disregard ghost cells).
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto rangeActiveCells() -> range_t<D>;

    /**
     * @brief Loop over all cells.
     * @returns Kokkos range policy with proper min/max indices and dimension.
     */
    auto rangeAllCells() -> range_t<D>;
  };

} // namespace ntt

#endif
