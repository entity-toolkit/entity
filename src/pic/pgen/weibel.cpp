#include "wrapper.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"
#include "particle_macros.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {
  template <Dimension D, SimulationType S>
  ProblemGenerator<D, S>::ProblemGenerator(const SimulationParams& params) {
    udrift_1 = readFromInput<real_t>(params.inputdata(), "problem", "udrift_1", 1.0);
    udrift_2 = readFromInput<real_t>(params.inputdata(), "problem", "udrift_2", -udrift_1);
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&   params,
                                                       Meshblock<Dim2, TypePIC>& mblock) {
    real_t Ymin     = params.extent()[2];
    real_t Ymax     = params.extent()[3];
    real_t sY       = Ymax - Ymin;
    real_t cs_width = 20.0;
    real_t cY1      = Ymin + 0.25 * sY;
    real_t cY2      = Ymin + 0.75 * sY;
    Kokkos::parallel_for(
      "userInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {static_cast<int>(i) - N_GHOSTS};
        real_t j_ {static_cast<int>(j) - N_GHOSTS};
        mblock.em(i, j, em::ex1) = ZERO;
        mblock.em(i, j, em::ex2) = ZERO;
        mblock.em(i, j, em::ex3) = ZERO;

        mblock.em(i, j, em::bx1) = ZERO;
        mblock.em(i, j, em::bx2) = ZERO;
        mblock.em(i, j, em::bx3) = ZERO;
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&   params,
                                                          Meshblock<Dim2, TypePIC>& mblock) {
    std::size_t npart = (std::size_t)(
      (double)(params.resolution()[0] * params.resolution()[1] * params.ppc0() * 0.5));
    auto&  electrons   = mblock.particles[0];
    auto&  positrons   = mblock.particles[1];
    auto   random_pool = *(mblock.random_pool_ptr);
    real_t Xmin = params.extent()[0], Xmax = params.extent()[1];
    real_t Ymin = params.extent()[2], Ymax = params.extent()[3];
    auto   u1 = udrift_1;
    auto   u2 = udrift_2;
    electrons.setNpart(npart);
    positrons.setNpart(npart);
    Kokkos::parallel_for(
      "userInitPrtls", CreateRangePolicy<Dim1>({0}, {(int)npart}), Lambda(index_t p) {
        typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();

        real_t rx = rand_gen.frand(Xmin, Xmax);
        real_t ry = rand_gen.frand(Ymin, Ymax);
        init_prtl_2d_XYZ(mblock, electrons, p, rx, ry, u1, 0.0, 0.0);
        init_prtl_2d_XYZ(mblock, positrons, p, rx, ry, u2, 0.0, 0.0);

        random_pool.free_state(rand_gen);
      });
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