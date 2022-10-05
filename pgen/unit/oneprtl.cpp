#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"
#include "particle_macros.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  template <>
  void ProblemGenerator<Dim2, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim2, SimulationType::PIC>& mblock) {

    Kokkos::parallel_for(
      "userInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)},
          j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
        real_t      ex2_hat {0.1}, bx3_hat {1.0};
        vec_t<Dim3> e_cntrv, b_cntrv;
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_}, {ZERO, ex2_hat, ZERO}, e_cntrv);
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bx3_hat}, b_cntrv);
        mblock.em(i, j, em::ex1) = ZERO;
        mblock.em(i, j, em::ex2) = e_cntrv[1];
        mblock.em(i, j, em::ex3) = ZERO;
        mblock.em(i, j, em::bx1) = ZERO;
        mblock.em(i, j, em::bx2) = ZERO;
        mblock.em(i, j, em::bx3) = b_cntrv[2];
      });
  }

  template <>
  void ProblemGenerator<Dim2, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dim2, SimulationType::PIC>& mblock) {
    auto electrons = mblock.particles[0];
    auto positrons = mblock.particles[1];
    auto random_pool = (*mblock.random_pool_ptr);
    Kokkos::parallel_for(
      "userInitPrtls", CreateRangePolicy<Dim1>({0}, {1}), Lambda(index_t p) {
        typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();
        real_t rx = rand_gen.frand((real_t)(-2.0), (real_t)(2.0));
        real_t ry = rand_gen.frand((real_t)(-2.0), (real_t)(2.0));
        PICPRTL_XYZ_2D(mblock, electrons, p, rx, ry, 1.0, 0.0, 0.0);
        PICPRTL_XYZ_2D(mblock, positrons, p, rx, ry, 1.0, 0.0, 0.0);
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
  void ProblemGenerator<Dim1, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim1, SimulationType::PIC>&) {}
  template <>
  void ProblemGenerator<Dim1, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dim1, SimulationType::PIC>&) {}

  // 3D
  template <>
  void ProblemGenerator<Dim3, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim3, SimulationType::PIC>&) {}
  template <>
  void ProblemGenerator<Dim3, SimulationType::PIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dim3, SimulationType::PIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dim1, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::SimulationType::PIC>;
