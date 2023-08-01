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
  }    // namespace
  typedef int PrepareOutputFlags;

  /**
   * @brief Container for the fields, particles and coordinate system.
   * @tparam D Dimension.
   * @tparam S Simulation engine.
   */
  template <Dimension D, SimulationEngine S>
  class Meshblock : public Mesh<D>, public Fields<D, S> {
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
    [[nodiscard]] auto timestep() const -> const real_t& {
      return m_timestep;
    }
    /**
     * @brief Get the minimum cell size.
     */
    [[nodiscard]] auto minCellSize() const -> const real_t& {
      return m_min_cell_size;
    }

    /**
     * @brief Set the timestep of the meshblock.
     * @param timestep timestep in physical units.
     */
    void setTimestep(const real_t& timestep) {
      m_timestep = timestep;
    }
    /**
     * @brief Set the minimum cell size of the meshblock.
     * @param min_cell_size minimum cell size in physical units.
     */
    void setMinCellSize(const real_t& min_cell_size) {
      m_min_cell_size = min_cell_size;
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
     * @brief The result is stored inside the buffer(i1, i2, buffer_comp).
     */
    void ComputeVectorPotential(ndfield_t<D, 6>&, const int&) {}

    /**
     * @brief Compute the divergence of the E/D-field.
     * @param buffer Buffer to store the result in.
     * @param buffer_comp Component of the buffer to store the result in.
     */
    void ComputeDivergenceED(ndfield_t<D, 3>&, const int&);

    /**
     * @brief Compute particle moment for output or other usage.
     * @param params SimulationParams object.
     * @param field FieldID for the moment to compute.
     * @param components Components of the field to compute (if applicable).
     * @param prtl_species Particle species to compute the moment for.
     * @param buff_ind Buffer index to store the result in (`meshblock::buff` array).
     * @param smooth Smoothing order (default: 2).
     */
    void ComputeMoments(const SimulationParams& params,
                        const FieldID&          field,
                        const std::vector<int>& components,
                        const std::vector<int>& prtl_species,
                        const int&              buff_ind,
                        const short&            smooth = 2);
  };
}    // namespace ntt

#endif
