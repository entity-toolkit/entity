#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"
#include "particle_macros.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {

    Kokkos::parallel_for(
      "userInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)},
          j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
        real_t                    ex2_hat {0.1}, bx3_hat {1.0};
        vec_t<Dimension::THREE_D> e_cntrv, b_cntrv;
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_}, {ZERO, ex2_hat, ZERO}, e_cntrv);
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bx3_hat}, b_cntrv);
        mblock.em(i, j, em::ex1) = ZERO;
        mblock.em(i, j, em::ex2) = ZERO; // e_cntrv[1];
        mblock.em(i, j, em::ex3) = ZERO;
        mblock.em(i, j, em::bx1) = ZERO;
        mblock.em(i, j, em::bx2) = ZERO;
        mblock.em(i, j, em::bx3) = ZERO; // b_cntrv[2];
      });
  }

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {

    Kokkos::parallel_for(
      "userInitPrtls", CreateRangePolicy<Dimension::ONE_D>({0}, {1}), Lambda(index_t p) {
        PICPRTL_XYZ_2D(&mblock, 0, p, 0.1, 0.12, 1.0, 0.0, 0.0);
        PICPRTL_XYZ_2D(&mblock, 1, p, 0.1, 0.12, -1.0, 0.0, 0.0);
        // PICPRTL_XYZ_2D(&mblock, 2, p, 0.1, 0.12, 1.0, 0.0, 0.0);
        // PICPRTL_XYZ_2D(&mblock, 3, p, 0.1, 0.12, 1.0, 0.0, 0.0);
      });
    mblock.particles[0].set_npart(1);
    mblock.particles[1].set_npart(1);
    // mblock.particles[2].set_npart(1);
    // mblock.particles[3].set_npart(1);
  }
  // 1D
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::PIC>&) {}
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::PIC>&) {}

  // 3D
  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::PIC>&) {}
  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::PIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
