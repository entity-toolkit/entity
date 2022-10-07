#ifndef PIC_PIC_H
#define PIC_PIC_H

#include "wrapper.h"
#include "simulation.h"

#include <toml/toml.hpp>

namespace ntt {
  /**
   * @brief Class for PIC simulations, inherits from `Simulation<D, TypePIC>`.
   * @tparam D dimension.
   */
  template <Dimension D>
  struct PIC : public Simulation<D, TypePIC> {
    /**
     * @brief Constructor for PIC class.
     * @param inputdata toml-object with parsed toml parameters.
     */
    PIC(const toml::value& inputdata) : Simulation<D, TypePIC>(inputdata) {}
    ~PIC() = default;

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
     * @brief Dummy function to match with GRPIC
     */
    void InitialStep() {}

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
     */
    void FieldsExchange();

    /* -------------------------------- Currents -------------------------------- */
    /**
     * @brief Spatially filter all the deposited currents.
     * @param t time in physical units.
     */
    void FilterCurrentsSubstep(const real_t& t);
    /**
     * @brief Deposit currents from particles.
     */
    void CurrentsDeposit();
    /**
     * @brief Apply boundary conditions for currents.
     */
    void CurrentsBoundaryConditions();
    /**
     * @brief Synchronize currents in ghost zones between the meshblocks.
     */
    void CurrentsExchange();

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
     * @brief Exchange particles between the meshblocks (incl. periodic BCs).
     */
    void ParticlesExchange();

    /**
     * @brief Benchmarking step.
     */
    void Benchmark();
  };

} // namespace ntt

#endif
