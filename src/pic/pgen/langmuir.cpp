#include "wrapper.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"
#include "particle_macros.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {
  template <Dimension D, SimulationType S>
  ProblemGenerator<D, S>::ProblemGenerator(const SimulationParams& params) {}

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim2, TypePIC>&) {
    // real_t Ymin     = params.extent()[2];
    // real_t Ymax     = params.extent()[3];
    // real_t sY       = Ymax - Ymin;
    // real_t cs_width = 20.0;
    // real_t cY1      = Ymin + 0.25 * sY;
    // real_t cY2      = Ymin + 0.75 * sY;
    // Kokkos::parallel_for(
    //   "userInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
    //     real_t i_ {static_cast<int>(i) - N_GHOSTS};
    //     real_t j_ {static_cast<int>(j) - N_GHOSTS};
    //     mblock.em(i, j, em::ex1) = ZERO;
    //     mblock.em(i, j, em::ex2) = ZERO;
    //     mblock.em(i, j, em::ex3) = ZERO;

    //     mblock.em(i, j, em::bx1) = ZERO;
    //     mblock.em(i, j, em::bx2) = ZERO;
    //     mblock.em(i, j, em::bx3) = ZERO;
    //   });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&   params,
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
          = (real_t)(0.01) * math::sin((real_t)(5.0) * constant::TWO_PI * rx / (Xmax - Xmin));

        init_prtl_2d_XYZ(mblock, electrons, p, rx, ry, u1, 0.0, 0.0);
        // init_prtl_2d_XYZ(mblock, positrons, p, rx, ry, u2, 0.0, 0.0);
        random_pool.free_state(rand_gen);
      });
  }
  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserDriveParticles(const real_t&,
                                                           const SimulationParams&,
                                                           Meshblock<Dim2, TypePIC>&) {}

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserBCFields(const real_t&,
                                                     const SimulationParams&,
                                                     Meshblock<Dim2, TypePIC>&) {}
  template <>
  Inline auto ProblemGenerator<Dim2, TypePIC>::UserTargetField_br_hat(
    const Meshblock<Dim2, TypePIC>&, const coord_t<Dim2>&) const -> real_t {
    return ZERO;
  }

  // clang-format off
  @PgenPlaceholder1D@
  @PgenPlaceholder3D@
  // clang-format on

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dim1, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::TypePIC>;