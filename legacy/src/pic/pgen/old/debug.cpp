#include "wrapper.h"
#include "io/input.h"
#include "sim_params.h"
#include "meshblock/meshblock.h"
#include "particle_macros.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim2, TypePIC>& mblock) {

    Kokkos::parallel_for(
      "UserInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)},
          j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
        real_t      ex2_hat {0.1}, bx3_hat {1.0};
        vec_t<Dim3> e_cntrv, b_cntrv;
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_}, {ZERO, ex2_hat, ZERO}, e_cntrv);
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bx3_hat}, b_cntrv);
        mblock.em(i, j, em::ex1) = ZERO;
        mblock.em(i, j, em::ex2) = ZERO;
        mblock.em(i, j, em::ex3) = ZERO;
        mblock.em(i, j, em::bx1) = ZERO;
        mblock.em(i, j, em::bx2) = ZERO;
        mblock.em(i, j, em::bx3) = ZERO;
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&,
                                                          Meshblock<Dim2, TypePIC>& mblock) {
    auto& electrons   = mblock.particles[0];
    auto& positrons   = mblock.particles[1];
    auto  random_pool = *(mblock.random_pool_ptr);
    Kokkos::parallel_for(
      "UserInitPrtls", CreateRangePolicy<Dim1>({0}, {1}), Lambda(index_t p) {
        typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();
        real_t rx = rand_gen.frand((real_t)(-2.0), (real_t)(2.0));
        real_t ry = rand_gen.frand((real_t)(-2.0), (real_t)(2.0));
        init_prtl_2d_XYZ(mblock, electrons, p, rx, ry, 1.0, 0.0, 0.0);
        init_prtl_2d_XYZ(mblock, positrons, p, rx, ry, 1.0, 0.0, 0.0);
        // init_prtl_2d_XYZ(&mblock, 2, p, 0.1, 0.12, 1.0, 0.0, 0.0);
        // init_prtl_2d_XYZ(&mblock, 3, p, 0.1, 0.12, 1.0, 0.0, 0.0);
      });
    electrons.setNpart(1);
    positrons.setNpart(1);
    // mblock.particles[2].setNpart(1);
    // mblock.particles[3].setNpart(1);
  }
  // 1D
  template <>
  void ProblemGenerator<Dim1, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim1, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim1, TypePIC>::UserInitParticles(const SimulationParams&,
                                                          Meshblock<Dim1, TypePIC>&) {}

  // 3D
  template <>
  void ProblemGenerator<Dim3, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim3, TypePIC>&) {}
  template <>
  void ProblemGenerator<Dim3, TypePIC>::UserInitParticles(const SimulationParams&,
                                                          Meshblock<Dim3, TypePIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dim1, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::TypePIC>;
