#ifndef GRPIC_GRPIC_H
#define GRPIC_GRPIC_H

#include "global.h"
#include "simulation.h"

#include <toml/toml.hpp>

namespace ntt {
  enum class gr_bc { Dfield, Efield, Bfield, Hfield };
  enum class gr_faraday { aux, main };
  enum class gr_ampere { init, aux, main };
  enum class gr_getE { D0_B, D_B0 };
  enum class gr_getH { D_B0, D0_B0 };

  /**
   * Class for GRPIC simulations, inherits from `Simulation<D, SimulationType::GRPIC>`.
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class GRPIC : public Simulation<D, SimulationType::GRPIC> {
  public:
    /**
     * Constructor for GRPIC class.
     *
     * @param inputdata toml-object with parsed toml parameters.
     */
    GRPIC(const toml::value& inputdata) : Simulation<D, SimulationType::GRPIC>(inputdata) {}
    ~GRPIC() = default;

    /**
     * Advance the simulation forward for one timestep.
     *
     * @param t time in physical units
     */
    void step_forward(const real_t&);

    // /**
    //  * Advance the simulation forward for one timestep.
    //  *
    //  * @param t time in physical units
    //  */
    // void step_backward(const real_t&);

    /**
     * From the initial fields, advances the first time steps.
     *
     * @param t time in physical units
     */
    void initial_step(const real_t&);

    /**
     * Advance the simulation forward for a specified amount of timesteps, keeping track of time.
     */
    void mainloop();

    /**
     * Process the simulation (calling initialize, verify, mainloop, etc).
     */
    void process();

    /**
     * Advance B-field using Faraday's law.
     *
     * @param t time in physical units.
     * @param c coefficient that gets multiplied by the timestep (e.g., 0.5).
     * @param f either intermediate substep or the main one [`gr_faraday::aux`, `gr_faraday::main`].
     */
    void faradaySubstep(const real_t&, const real_t&, const gr_faraday&);

    /**
     * Advance D-field using Ampere's law.
     *
     * @param t time in physical units.
     * @param c coefficient that gets multiplied by the timestep (e.g., 0.5).
     * @param f either initial, intermediate or the main substep [`gr_ampere::init`, `gr_ampere::aux`,
     * `gr_ampere::main`].
     */
    void ampereSubstep(const real_t&, const real_t&, const gr_ampere&);

    /**
     * Compute E field.
     *
     * @param t time in physical units.
     * @param f flag to use D0 and B or D and B0 [`gr_getE::D0_B`, `gr_getE::D_B0`].
     */
    void computeAuxESubstep(const real_t&, const gr_getE&);

    /**
     * Compute H field.
     *
     * @param t time in physical units.
     * @param f flat to use D0 and B0 or D and B0 [`gr_getH::D_B0`, `gr_getH::D0_B0`].
     */
    void computeAuxHSubstep(const real_t&, const gr_getH&);

    /**
     * Time average EM fields.
     *
     * @param t time in physical units.
     */
    void timeAverageDBSubstep(const real_t&);

    /**
     * Time average currents.
     *
     * @param t time in physical units.
     */
    void timeAverageJSubstep(const real_t&);

    /**
     * Apply boundary conditions for fields.
     *
     * @param t time in physical units.
     * @param f select field to apply boundary conditions to [`gr_bc::Dfield`, `gr_bc::Bfield`].
     */
    void fieldBoundaryConditions(const real_t&, const gr_bc&);

    /**
     * Apply boundary conditions for auxiliary fields.
     *
     * @param t time in physical units.
     * @param f select field to apply boundary conditions to [`gr_bc::Efield`, `gr_bc::Hfield`].
     */
    void auxFieldBoundaryConditions(const real_t&, const gr_bc&);

    /**
     * @brief Swaps em and em0 fields, cur and cur0 currents.
     */
    void swapFieldsGR() {
      std::swap((this->m_mblock).em, (this->m_mblock).em0);
      std::swap((this->m_mblock).cur, (this->m_mblock).cur0);
    }
    /**
     * @brief Copies em fields into em0
     *
     */
    void copyFieldsGR() { Kokkos::deep_copy((this->m_mblock).em0, (this->m_mblock).em); }
  };

} // namespace ntt

#endif