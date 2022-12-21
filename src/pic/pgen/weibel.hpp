#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "sim_params.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) {
      udrift_1 = readFromInput<real_t>(params.inputdata(), "problem", "udrift_1", 1.0);
      udrift_2 = readFromInput<real_t>(params.inputdata(), "problem", "udrift_2", -udrift_1);
    }
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&);

  private:
    real_t udrift_1, udrift_2;
  };

  template <>
  inline void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, TypePIC>& mblock) {
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
      "userInitPrtls", CreateRangePolicy<Dim1>({ 0 }, { (int)npart }), Lambda(index_t p) {
        typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();

        real_t                                      rx       = rand_gen.frand(Xmin, Xmax);
        real_t                                      ry       = rand_gen.frand(Ymin, Ymax);
        init_prtl_2d(mblock, electrons, p, rx, ry, u1, 0.0, 0.0);
        init_prtl_2d(mblock, positrons, p, rx, ry, u2, 0.0, 0.0);

        random_pool.free_state(rand_gen);
      });
  }

  template <>
  inline void ProblemGenerator<Dim1, TypePIC>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim1, TypePIC>& mblock) {}

  template <>
  inline void ProblemGenerator<Dim3, TypePIC>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim3, TypePIC>& mblock) {}

}    // namespace ntt

#endif
