#include "wrapper.h"
#include "sim_params.h"
#include "meshblock.h"
#include "field_macros.h"
#include "particle_macros.h"

#include "problem_generator.hpp"

#include <cmath>
#include <functional>

namespace ntt {
  
  Inline void reconnectionField(const coord_t<Dim2>& x_ph,
                                vec_t<Dim3>&         e_out,
                                vec_t<Dim3>&         b_out,
                                real_t               csW,
                                real_t               cY1,
                                real_t               cY2) {
    b_out[1] = math::tanh((x_ph[0] - cY1) / csW) - math::tanh((x_ph[0] - cY2) / csW) - ONE;
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
        init_em_fields_2d(mblock, i, j, reconnectionField, cs_width, cY1, cY2);
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&   params,
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
        init_prtl_2d_XYZ(mblock, electrons, p, rx, ry, 0.0, 0.0, 0.0);
        init_prtl_2d_XYZ(mblock, positrons, p, rx, ry, 0.0, 0.0, 0.0);

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