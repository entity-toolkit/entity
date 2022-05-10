#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    Kokkos::parallel_for(
      "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)}, j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
        real_t ex2_hat {0.1}, bx3_hat {1.0};
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
    using index_t = const std::size_t;
    Kokkos::parallel_for(
      "userInitPrtls", NTTRange<Dimension::ONE_D>({0}, {1}), Lambda(index_t p) {
        coord_t<Dimension::TWO_D> x {0.1, 0.12}, x_CU;
        mblock.metric.x_Cart2Code(x, x_CU);
        auto [i1, dx1] = mblock.metric.CU_to_Idi(x_CU[0]);
        auto [i2, dx2] = mblock.metric.CU_to_Idi(x_CU[1]);
        // electron
        mblock.particles[0].i1(p) = i1;
        mblock.particles[0].i2(p) = i2;
        mblock.particles[0].dx1(p) = dx1;
        mblock.particles[0].dx2(p) = dx2;
        mblock.particles[0].ux3(p) = 1.0;
        // positron
        mblock.particles[1].i1(p) = i1;
        mblock.particles[1].i2(p) = i2;
        mblock.particles[1].dx1(p) = dx1;
        mblock.particles[1].dx2(p) = dx2;
        mblock.particles[1].ux3(p) = 1.0;
        // photon
        mblock.particles[2].i1(p) = i1;
        mblock.particles[2].i2(p) = i2;
        mblock.particles[2].dx1(p) = dx1;
        mblock.particles[2].dx2(p) = dx2;
        mblock.particles[2].ux1(p) = 1.0;
      });
    mblock.particles[0].set_npart(1);
    mblock.particles[1].set_npart(1);
    mblock.particles[2].set_npart(0);
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