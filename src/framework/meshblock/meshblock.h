#ifndef FRAMEWORK_MESHBLOCK_H
#define FRAMEWORK_MESHBLOCK_H

#include "wrapper.h"

#include "sim_params.h"

#include "meshblock/fields.h"
#include "meshblock/mesh.h"
#include "meshblock/particles.h"
#include "meshblock/species.h"

#include <vector>

namespace ntt {
  namespace {
    enum PrepareOutputFlags_ {
      PrepareOutput_None                        = 0,
      PrepareOutput_InterpToCellCenterFromEdges = 1 << 0,
      PrepareOutput_InterpToCellCenterFromFaces = 1 << 1,
      PrepareOutput_ConvertToHat                = 1 << 2,
      PrepareOutput_ConvertToPhysCntrv          = 1 << 3,
      PrepareOutput_ConvertToPhysCov            = 1 << 4,
    };

    enum CheckNaNFlags_ {
      CheckNaN_None      = 0,
      CheckNaN_Particles = 1 << 0,
      CheckNaN_Fields    = 1 << 1,
      CheckNaN_Currents  = 1 << 2,
    };
  } // namespace

  typedef int PrepareOutputFlags;
  typedef int CheckNaNFlags;

  /**
   * @brief Container for the fields, particles and coordinate system.
   * @tparam D Dimension.
   * @tparam S Simulation engine.
   */
  template <Dimension D, SimulationEngine S>
  class Meshblock : public Mesh<D>,
                    public Fields<D, S> {
  private:
    // Timestep duration in physical units defined at the meshblock.
    real_t m_timestep;
    // Effective minimum cell size of the meshblock.
    real_t m_min_cell_size;

  public:
    // Vector of particles species.
    std::vector<Particles<D, S>> particles;
    RandomNumberPool_t*          random_pool_ptr;

    /**
     * @brief Constructor for the meshblock.
     * @param res resolution vector of size D (dimension).
     * @param ext extent vector of size 2 * D.
     * @param params metric-/domain-specific parameters (max: 10).
     * @param species vector of particle species parameters.
     */
    Meshblock(const std::vector<unsigned int>&    res,
              const std::vector<real_t>&          ext,
              const real_t*                       params,
              const std::vector<ParticleSpecies>& species);
    ~Meshblock() = default;

    /**
     * @brief Get the timestep.
     */
    [[nodiscard]]
    auto timestep() const -> real_t {
      return m_timestep;
    }

    /**
     * @brief Get the minimum cell size.
     */
    [[nodiscard]]
    auto minCellSize() const -> real_t {
      return this->metric.dxMin();
    }

    /**
     * @brief Set the timestep of the meshblock.
     * @param timestep timestep in physical units.
     */
    void setTimestep(const real_t& timestep) {
      m_timestep = timestep;
    }

    /**
     * @brief Verify that all the specified parameters are valid.
     */
    void Verify();

    /* ----------------- Additional conversions and computations ---------------- */

    /**
     * @brief Interpolate and convert fields to prepare for output.
     * @brief Details provided using the `PrepareOutputFlags`.
     * @brief The result is stored inside the buffer.
     */
    template <int N, int M>
    void PrepareFieldsForOutput(const ndfield_t<D, N>&    field,
                                ndfield_t<D, M>&          buffer,
                                const int&                fx1,
                                const int&                fx2,
                                const int&                fx3,
                                const PrepareOutputFlags& flags);

    /**
     * @brief Compute A3 vector potential (for GRPIC 2D).
     * @brief The result is stored inside the buffer(i1, i2, buff_ind).
     */
    void ComputeVectorPotential(ndfield_t<D, 6>&, int) {}

    /**
     * @brief Compute the divergence of the E/D-field.
     * @param buffer Buffer to store the result in.
     * @param buff_ind Component of the buffer to store the result in.
     */
    void ComputeDivergenceED(ndfield_t<D, 3>&, int);

    /**
     * @brief Compute particle moment for output or other usage.
     * @param params SimulationParams object.
     * @param field FieldID for the moment to compute.
     * @param components Components of the field to compute (if applicable).
     * @param prtl_species Particle species to compute the moment for.
     * @param buff_ind Buffer index to store the result in (`meshblock::buff` array).
     * @param window Smoothing window (default: 2).
     */
    void ComputeMoments(const SimulationParams& params,
                        const FieldID&          field,
                        const std::vector<int>& components,
                        const std::vector<int>& prtl_species,
                        int                     buff_ind,
                        short                   window = 2);

    /**
     * @brief Check for NaNs in the fields, currents and/or particles.
     * @param msg Message to print if NaNs are found.
     * @param flags Pick which quantities to check using the `CheckNaNFlags`.
     */
    void CheckNaNs(const std::string&, CheckNaNFlags);

    /**
     * @brief Compute the charge density.
     * @param params SimulationParams object.
     * @param buffer Buffer to store the result in.
     * @param prtl_species Particle species to compute the charge density for.
     * @param buff_ind Buffer index to store the result in (`meshblock::buff` array).
     */
    void ComputeChargeDensity(const SimulationParams&,
                              ndfield_t<D, 3>&,
                              const std::vector<int>&,
                              int);

    /**
     * @brief Check for particles out of bounds.
     * @param msg Message to print if particles are out of bounds.
     * @param only_on_debug Only run when DEBUG enabled.
     */
    void CheckOutOfBounds(const std::string&, bool = true);
  };
} // namespace ntt

#endif
