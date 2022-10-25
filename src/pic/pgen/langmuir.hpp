#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"
#include "particle_macros.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&);
  };

  template <>
  inline void
  ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&   params,
                                                     Meshblock<Dim2, TypePIC>& mblock) {
    auto        ncells      = mblock.Ni1() * mblock.Ni2() * mblock.Ni3();
    std::size_t npart       = (std::size_t)((double)(ncells * params.ppc0() * 0.5));
    auto&       electrons   = mblock.particles[0];
    auto        random_pool = *(mblock.random_pool_ptr);
    real_t      Xmin        = mblock.metric.x1_min;
    real_t      Xmax        = mblock.metric.x1_max;
    real_t      Ymin        = mblock.metric.x2_min;
    real_t      Ymax        = mblock.metric.x2_max;
    electrons.setNpart(npart);
    Kokkos::parallel_for(
      "userInitPrtls", CreateRangePolicy<Dim1>({0}, {(int)npart}), Lambda(index_t p) {
        typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();

        real_t rx = rand_gen.frand(Xmin, Xmax);
        real_t ry = rand_gen.frand(Ymin, Ymax);
        real_t u1
          = (real_t)(0.01) * math::sin((real_t)(2.0) * constant::TWO_PI * rx / (Xmax - Xmin));

        init_prtl_2d_XYZ(mblock, electrons, p, rx, ry, u1, 0.0, 0.0);
        random_pool.free_state(rand_gen);
      });
  }

  template <>
  inline void ProblemGenerator<Dim1, TypePIC>::UserInitParticles(const SimulationParams&,
                                                                 Meshblock<Dim1, TypePIC>&) {}
  template <>
  inline void ProblemGenerator<Dim3, TypePIC>::UserInitParticles(const SimulationParams&,
                                                                 Meshblock<Dim3, TypePIC>&) {}

} // namespace ntt

#endif
