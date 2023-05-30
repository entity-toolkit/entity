#ifndef PIC_PIC_H
#define PIC_PIC_H

#include "wrapper.h"

#include "simulation.h"

#include PGEN_HEADER

#include <toml.hpp>

namespace ntt {
  /**
   * @brief Class for PIC simulations, inherits from `Simulation<D, PICEngine>`.
   * @tparam D dimension.
   */
  template <Dimension D>
  struct PIC : public Simulation<D, PICEngine> {
    // problem setup generator
    ProblemGenerator<D, PICEngine> problem_generator;

    /**
     * @brief Constructor for PIC class.
     * @param inputdata toml-object with parsed toml parameters.
     */
    PIC(const toml::value& inputdata)
      : Simulation<D, PICEngine>(inputdata), problem_generator { this->m_params } {}
    PIC(const PIC<D>&) = delete;
    ~PIC()             = default;

    /**
     * @brief Advance the simulation forward for one timestep.
     */
    void StepForward();

    /**
     * @brief Advance the simulation forward for one timestep.
     */
    void StepBackward();

    /**
     * @brief Run the simulation (calling initialize, verify, mainloop, etc).
     */
    void Run();

    /**
     * @brief Initialize the setup with a problem generator.
     */
    void InitializeSetup();

    /**
     * @brief Initial step performed before the main loop.
     */
    void InitialStep();

    /* ---------------------------------- Reset --------------------------------- */
    /**
     * @brief Reset field arrays.
     */
    void ResetFields();
    /**
     * @brief Reset current arrays.
     */
    void ResetCurrents();
    /**
     * @brief Reset particles.
     */
    void ResetParticles();
    /**
     * @brief Reset whole simulation.
     */
    void ResetSimulation();

    /* --------------------------------- Fields --------------------------------- */
    /**
     * @brief Advance B-field using Faraday's law.
     * @param f coefficient that gets multiplied by the timestep (def. 0.5).
     */
    void Faraday(const real_t& f = 0.5);
    /**
     * @brief Advance E-field using Ampere's law (without currents).
     * @param f coefficient that gets multiplied by the timestep (def. 1.0).
     */
    void Ampere(const real_t& f = 1.0);
    /**
     * @brief Add computed and filtered currents to the E-field.
     */
    void AmpereCurrents();
    /**
     * @brief Apply special boundary conditions for fields.
     */
    void FieldsBoundaryConditions();
    /**
     * @brief Synchronize ghost zones between the meshblocks.
     * @param f flag to synchronize fields, currents or particles.
     */
    void Exchange(const GhostCells&);

    /* -------------------------------- Currents -------------------------------- */
    /**
     * @brief Spatially filter all the deposited currents.
     */
    void CurrentsFilter();
    /**
     * @brief Deposit currents from particles.
     */
    void CurrentsDeposit();
    /**
     * @brief Apply boundary conditions for currents.
     */
    void CurrentsBoundaryConditions();
    /**
     * @brief Synchronize currents deposited in different meshblocks.
     */
    void CurrentsSynchronize();

    /* -------------------------------- Particles ------------------------------- */
    /**
     * @brief Advance particle positions and velocities.
     * @param f coefficient that gets multiplied by the timestep (def. 1.0).
     */
    void ParticlesPush(const real_t& f = 1.0);
    /**
     * @brief Apply boundary conditions for particles.
     */
    void ParticlesBoundaryConditions();

    /**
     * @brief Benchmarking step.
     */
    void Benchmark();
  };

}    // namespace ntt

#endif
