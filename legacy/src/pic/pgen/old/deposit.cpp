#include "wrapper.h"
#include "io/input.h"
#include "sim_params.h"
#include "meshblock/meshblock.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  template <>
  void ProblemGenerator<Dim2, TypePIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim2, TypePIC>& mblock) {

    Kokkos::parallel_for(
      "userInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)},
          j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
        // real_t ex2_hat {0.1}, bx3_hat {1.0};
        // vec_t<Dim3> e_cntrv, b_cntrv;
        // mblock.metric.v_Hat2Cntrv({i_ + HALF, j_}, {ZERO, ex2_hat, ZERO}, e_cntrv);
        // mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bx3_hat}, b_cntrv);
        mblock.em(i, j, em::ex1) = ZERO;
        mblock.em(i, j, em::ex2) = ZERO; // e_cntrv[1];
        mblock.em(i, j, em::ex3) = ZERO;
        mblock.em(i, j, em::bx1) = ZERO;
        mblock.em(i, j, em::bx2) = ZERO;
        mblock.em(i, j, em::bx3) = ZERO; // b_cntrv[2];
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dim2, TypePIC>& mblock) {

    Kokkos::parallel_for(
      "userInitPrtls", CreateRangePolicy<Dim1>({0}, {1}), Lambda(index_t p) {
        coord_t<Dim2> x {0.0, 0.0}, x_CU;
        mblock.metric.x_Cart2Code(x, x_CU);
        auto [i1, dx1] = mblock.metric.CU_to_Idi(x_CU[0]);
        auto [i2, dx2] = mblock.metric.CU_to_Idi(x_CU[1]);
        // electron
        mblock.particles[0].i1(p)  = i1;
        mblock.particles[0].i2(p)  = i2;
        mblock.particles[0].dx1(p) = dx1;
        mblock.particles[0].dx2(p) = dx2;
        // mblock.particles[0].ux1(p) = -2.0;
        // mblock.particles[0].ux2(p) = -5.0;
        mblock.particles[0].ux1(p) = 12.0;
        // positron
        mblock.particles[1].i1(p)  = i1;
        mblock.particles[1].i2(p)  = i2;
        mblock.particles[1].dx1(p) = dx1;
        mblock.particles[1].dx2(p) = dx2;
        // mblock.particles[0].ux1(p) = -2.0;
        // mblock.particles[0].ux2(p) = 6.0;
        // mblock.particles[1].ux3(p) = 0.0;
      });
    mblock.particles[0].setNpart(1);
    mblock.particles[1].setNpart(1);
  }
  // 1D
  template <>
  void ProblemGenerator<Dim1, TypePIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim1, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim1, TypePIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dim1, TypePIC>&) {}

  // 3D
  template <>
  void ProblemGenerator<Dim3, TypePIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim3, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim3, TypePIC>::userInitParticles(
    const SimulationParams&, Meshblock<Dim3, TypePIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dim1, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::TypePIC>;