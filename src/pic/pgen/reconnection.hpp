#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "input.h"
#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) {
      m_cs_width = readFromInput<real_t>(params.inputdata(), "problem", "cs_width");
    }

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}

  private:
    real_t m_cs_width;
  };    // struct ProblemGenerator

  Inline void reconnectionField(const coord_t<Dim2>& x_ph,
                                vec_t<Dim3>&         e_out,
                                vec_t<Dim3>&         b_out,
                                real_t               csW,
                                real_t               c1,
                                real_t               c2) {
    b_out[1] = math::tanh((x_ph[0] - c1) / csW) - math::tanh((x_ph[0] - c2) / csW) - ONE;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    real_t Xmin     = mblock.metric.x1_min;
    real_t Xmax     = mblock.metric.x1_max;
    real_t sX       = Xmax - Xmin;
    auto   cs_width = m_cs_width;
    real_t cX1      = Xmin + 0.25 * sX;
    real_t cX2      = Xmin + 0.75 * sX;
    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, reconnectionField, cs_width, cX1, cX2);
      });
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    InjectUniform<Dim2, PICEngine>(params, mblock, { 1, 2 }, params.ppc0() * 0.5);
  }
}    // namespace ntt

#endif
