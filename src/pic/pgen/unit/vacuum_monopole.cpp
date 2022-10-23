#include "wrapper.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"
#include "field_macros.h"

#include "problem_generator.hpp"

#include <cmath>
#include <iostream>

namespace ntt {

  template <>
  ProblemGenerator<Dim2, TypePIC>::ProblemGenerator(const SimulationParams& params) {
    spinup_time = readFromInput<real_t>(params.inputdata(), "problem", "spinup_time");
    omega_max   = readFromInput<real_t>(params.inputdata(), "problem", "omega_max");
#ifdef MINKOWSKI_METRIC
    NTTHostError("Vacuum monopole not supported in Minkowski metric");
#endif
  }

  Inline void monopoleField(const coord_t<Dim2>& x_ph,
                            vec_t<Dim3>&         e_out,
                            vec_t<Dim3>&         b_out,
                            real_t               rmin) {
    b_out[0] = SQR(rmin / x_ph[0]);
  }

  Inline void surfaceRotationField(const coord_t<Dim2>& x_ph,
                                   vec_t<Dim3>&         e_out,
                                   vec_t<Dim3>&         b_out,
                                   real_t               rmin,
                                   real_t               omega) {
    b_out[0] = SQR(rmin / x_ph[0]);
    e_out[1] = omega * math::sin(x_ph[1]);
    e_out[2] = 0.0;
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&,
                                                       Meshblock<Dim2, TypePIC>& mblock) {
    auto r_min = mblock.metric.x1_min;
    Kokkos::parallel_for(
      "UserInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, monopoleField, r_min);
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserBCFields(const real_t& time,
                                                     const SimulationParams&,
                                                     Meshblock<Dim2, TypePIC>& mblock) {
    real_t omega;
    auto   r_min = mblock.metric.x1_min;
    if (time < spinup_time) {
      omega = omega_max * time / spinup_time;
    } else {
      omega = omega_max;
    }
    Kokkos::parallel_for(
      "UserBcFlds_rmin",
      CreateRangePolicy<Dim2>({mblock.i1_min(), mblock.i2_min()},
                              {mblock.i1_min() + 1, mblock.i2_max()}),
      Lambda(index_t i, index_t j) {
        // set_em_fields_2d(mblock, i, j, surfaceRotationField, r_min, omega);
        set_ex2_2d(mblock, i, j, surfaceRotationField, r_min, omega);
        set_ex3_2d(mblock, i, j, surfaceRotationField, r_min, omega);
        set_bx1_2d(mblock, i, j, surfaceRotationField, r_min, omega);
      });

    Kokkos::parallel_for(
      "UserBcFlds_rmax",
      CreateRangePolicy<Dim2>({mblock.i1_max(), mblock.i2_min()},
                              {mblock.i1_max() + 1, mblock.i2_max()}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::ex3) = 0.0;
        mblock.em(i, j, em::ex2) = 0.0;
        mblock.em(i, j, em::bx1) = 0.0;
      });
  }

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&,
                                                          Meshblock<Dim2, TypePIC>&) {}

  template <>
  void ProblemGenerator<Dim2, TypePIC>::UserDriveParticles(const real_t&,
                                                           const SimulationParams&,
                                                           Meshblock<Dim2, TypePIC>&) {}

  // clang-format off
  @PgenPlaceholder1D@
  @PgenPlaceholder3D@
  // clang-format on

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dim1, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::TypePIC>;