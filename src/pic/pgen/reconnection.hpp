#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "input.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"
#include "field_macros.h"
#include "particle_macros.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) {
      m_cs_width = readFromInput<real_t>(params.inputdata(), "problem", "cs_width");
    }

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&);
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&);

  private:
    real_t m_cs_width;
  };

  Inline void reconnectionField(const coord_t<Dim2>& x_ph,
                                vec_t<Dim3>&         e_out,
                                vec_t<Dim3>&         b_out,
                                real_t               csW,
                                real_t               c1,
                                real_t               c2) {
    b_out[1] = math::tanh((x_ph[0] - c1) / csW) - math::tanh((x_ph[0] - c2) / csW) - ONE;
  }

  template <>
  inline void
  ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&   params,
                                                  Meshblock<Dim2, TypePIC>& mblock) {
    real_t Xmin     = mblock.metric.x1_min;
    real_t Xmax     = mblock.metric.x1_max;
    real_t sX       = Xmax - Xmin;
    auto   cs_width = m_cs_width;
    real_t cX1      = Xmin + 0.25 * sX;
    real_t cX2      = Xmin + 0.75 * sX;
    Kokkos::parallel_for(
      "userInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, reconnectionField, cs_width, cX1, cX2);
      });
  }

  template <>
  inline void
  ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&   params,
                                                     Meshblock<Dim2, TypePIC>& mblock) {
    auto   npart = (std::size_t)((double)(mblock.Ni1() * mblock.Ni2() * params.ppc0() * 0.5));
    auto&  electrons   = mblock.particles[0];
    auto&  positrons   = mblock.particles[1];
    auto   random_pool = *(mblock.random_pool_ptr);
    real_t Xmin = mblock.metric.x1_min, Xmax = mblock.metric.x1_max;
    real_t Ymin = mblock.metric.x2_min, Ymax = mblock.metric.x2_max;
    electrons.setNpart(npart);
    positrons.setNpart(npart);
    Kokkos::parallel_for(
      "userInitPrtls", CreateRangePolicy<Dim1>({0}, {(int)npart}), Lambda(index_t p) {
        typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();

        real_t rx = rand_gen.frand(Xmin, Xmax);
        real_t ry = rand_gen.frand(Ymin, Ymax);
        init_prtl_2d(mblock, electrons, p, rx, ry, 0.0, 0.0, 0.0);
        init_prtl_2d(mblock, positrons, p, rx, ry, 0.0, 0.0, 0.0);

        random_pool.free_state(rand_gen);
      });
  }
} // namespace ntt

#endif
