#ifndef GRPIC_GRPIC_H
#define GRPIC_GRPIC_H

#include "wrapper.h"

#include "simulation.h"

#include "problem_generator.hpp"

#include <toml.hpp>

namespace ntt {
  enum class gr_bc { Dfield, Efield, Bfield, Hfield };
  enum class gr_faraday { aux, main };
  enum class gr_ampere { init, aux, main };
  enum class gr_getE { D0_B, D_B0 };
  enum class gr_getH { D_B0, D0_B0 };

  /**
   * Class for GRPIC simulations, inherits from `Simulation<D, GRPICEngine>`.
   * @tparam D dimension.
   */
  template <Dimension D>
  struct GRPIC : public Simulation<D, GRPICEngine> {
  public:
    // problem setup generator
    ProblemGenerator<D, GRPICEngine> problem_generator;

    /**
     * Constructor for GRPIC class.
     *
     * @param inputdata toml-object with parsed toml parameters.
     */
    GRPIC(const toml::value& inputdata)
      : Simulation<D, GRPICEngine>(inputdata), problem_generator { this->m_params } {}
    GRPIC(const GRPIC<D>&) = delete;
    ~GRPIC()               = default;

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
     * @param g indicator for either intermediate substep or the main one:
     * [`gr_faraday::aux`, `gr_faraday::main`].
     */
    void Faraday(const real_t& f, const gr_faraday&);
    /**
     * @brief Advance E-field using Ampere's law (without currents).
     * @param f coefficient that gets multiplied by the timestep (def. 1.0).
     * @param g indicator for either initial, intermediate or the main substep:
     * [`gr_ampere::init`, `gr_ampere::aux`, `gr_ampere::main`].
     */
    void Ampere(const real_t& f, const gr_ampere&);
    /**
     * @brief Add computed and filtered currents to the E-field.
     * !ADD: #GR
     */
    void AmpereCurrents() {}
    /**
     * @brief Apply special boundary conditions for fields.
     * @param g select field to apply boundary conditions to:
     * [`gr_bc::Dfield`, `gr_bc::Bfield`].
     */
    void FieldsBoundaryConditions(const gr_bc&);
    /**
     * @brief Synchronize ghost zones between the meshblocks.
     * @param f flag to synchronize fields, currents or particles.
     */
    void Exchange(const GhostCells&);

    /* ----------------------------- Aux fields --------------------------------- */
    /**
     * @brief Compute E field.
     * @param g flag to use D0 and B or D and B0 [`gr_getE::D0_B`, `gr_getE::D_B0`].
     */
    void ComputeAuxE(const gr_getE&);
    /**
     * @brief Compute H field.
     * @param g flat to use D0 and B0 or D and B0 [`gr_getH::D_B0`, `gr_getH::D0_B0`].
     */
    void ComputeAuxH(const gr_getH&);
    /**
     * @brief Apply special boundary conditions for auxiliary fields.
     * @param g select field to apply boundary conditions to:
     * [`gr_bc::Efield`, `gr_bc::Hfield`].
     */
    void AuxFieldsBoundaryConditions(const gr_bc&);
    /**
     * @brief Time average EM fields.
     */
    void TimeAverageDB();

    /* -------------------------------- Currents -------------------------------- */
    /**
     * !ADD: #GR
     */
    /**
     * @brief Spatially filter all the deposited currents.
     */
    void CurrentsFilter() {}
    /**
     * @brief Deposit currents from particles.
     */
    void CurrentsDeposit() {}
    /**
     * @brief Apply boundary conditions for currents.
     */
    void CurrentsBoundaryConditions() {}
    /**
     * @brief Synchronize currents deposited in different meshblocks.
     */
    void CurrentsSynchronize() {}
    /**
     * @brief Time average J currents.
     */
    void TimeAverageJ() {}

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
     * @brief Swaps em and em0 fields, cur and cur0 currents.
     */
    void SwapFields() {
      auto& mblock = this->meshblock;
      std::swap(mblock.em, mblock.em0);
      std::swap(mblock.cur, mblock.buff);
    }
    /**
     * @brief Copies em fields into em0
     */
    void CopyFields() {
      auto& mblock = this->meshblock;
      Kokkos::deep_copy(mblock.em0, mblock.em);
    }
    // /**
    //  * @brief Computes Aphi
    //  */
    // void ComputeVectorPotential();
  };

  /**
   * Computes Aphi from integration of local Br
   *
   * @tparam D Dimension.
   */

  // template <Dimension D>
  // class Compute_Aphi {
  //   Meshblock<D, SimulationType::GRPIC> m_mblock;
  //   real_t                              m_eps;
  //   int                                 i2_min;

  // public:
  //   Compute_Aphi(const Meshblock<D, SimulationType::GRPIC>& mblock, real_t eps)
  //     : m_mblock(mblock), m_eps(eps), i2_min(mblock.i2_min()) {}

  //   Inline void operator()(index_t, index_t) const;
  // };

}    // namespace ntt

#endif
