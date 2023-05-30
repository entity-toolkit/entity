#include "wrapper.h"
#include "io/input.h"
#include "sim_params.h"
#include "meshblock/meshblock.h"
#include "particle_macros.h"

#include "problem_generator.hpp"

#include <cmath>

namespace ntt {

  template <>
  ProblemGenerator<Dim2, TypePIC>::ProblemGenerator(const SimulationParams&) {}

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim2, TypePIC>& mblock) {
    Kokkos::parallel_for(
      "UserInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)},
          j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
        // real_t      ex2_hat {0.1}, bx3_hat {1.0};
        vec_t<Dim3> e_cntrv;
        mblock.metric.v_Hat2Cntrv({i_, j_}, {1e-4, ZERO, ZERO}, e_cntrv);
        mblock.em(i, j, em::ex1) = e_cntrv[0];
        // mblock.em(i, j, em::ex2) = e_cntrv[1];
        // mblock.em(i, j, em::ex3) = ZERO;
        // mblock.em(i, j, em::bx1) = ZERO;
        // mblock.em(i, j, em::bx2) = ZERO;
        // mblock.em(i, j, em::bx3) = b_cntrv[2];
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&,
                                                          Meshblock<Dim2, TypePIC>& mblock) {
    auto& electrons = mblock.particles[0];
    // auto& positrons = mblock.particles[1];
    electrons.setNpart(1);
    // positrons.setNpart(1);
    Kokkos::parallel_for(
      "UserInitPrtls", CreateRangePolicy<Dim1>({0}, {1}), Lambda(index_t p) {
        real_t rx = 0.0, ry = 0.0;
        init_prtl_2d_XYZ(mblock, electrons, p, rx, ry, 0.0, 0.0, 0.0);
        // init_prtl_2d_XYZ(mblock, positrons, p, rx, ry, 1.0, 0.0, 0.0);
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
