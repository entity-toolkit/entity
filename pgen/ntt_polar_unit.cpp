#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "ntt_polar_unit.hpp"

#include <cmath>

namespace ntt {

  template<Dimension D>
  ProblemGenerator<D>::ProblemGenerator(SimulationParams& sim_params) {
    UNUSED(sim_params);
  }

  // * * * * * * * * * * * * * * * * * * * * * * * *
  // Field initializers
  // . . . . . . . . . . . . . . . . . . . . . . . .
  template <>
  void ProblemGenerator<ONE_D>::userInitFields(SimulationParams&,
                                               Meshblock<ONE_D>&) {}

  template <>
  void ProblemGenerator<TWO_D>::userInitFields(SimulationParams& sim_params,
                                               Meshblock<TWO_D>& mblock) {
    UNUSED(sim_params);
    using index_t = NTTArray<real_t**>::size_type;
    Kokkos::deep_copy(mblock.em_fields, 0.0);
    real_t r_min {mblock.grid->x1_min};
    Kokkos::parallel_for(
      "userInitFlds",
      mblock.loopActiveCells(),
      Lambda(index_t i, index_t j) {
        auto i_ {static_cast<real_t>(i - N_GHOSTS)};
        auto j_ {static_cast<real_t>(j - N_GHOSTS)};

        auto [r_, th_] = mblock.grid->coord_CU_to_Sph(i_, j_ + HALF);

        auto br_hat {ONE * r_min * r_min / (r_ * r_)};
        mblock.em_fields(i, j, fld::bx1) = mblock.grid->vec_HAT_to_CNT_x1(br_hat, i_, j_ + HALF);
    });
  }

  template <>
  void ProblemGenerator<THREE_D>::userInitFields(SimulationParams&,
                                                 Meshblock<THREE_D>&) {}

  // * * * * * * * * * * * * * * * * * * * * * * * *
  // Field boundary conditions
  // . . . . . . . . . . . . . . . . . . . . . . . .
  template <>
  void ProblemGenerator<ONE_D>::userBCFields(const real_t&,
                                             SimulationParams&,
                                             Meshblock<ONE_D>&) {}

  template <>
  void ProblemGenerator<TWO_D>::userBCFields(const real_t& time,
                                             SimulationParams& sim_params,
                                             Meshblock<TWO_D>& mblock) {
    UNUSED(sim_params);
    using index_t = NTTArray<real_t**>::size_type;
    real_t omega;
    if (time < 0.5) {
      omega = time / 10.0;
    } else {
      omega = 0.05;
    }
    Kokkos::parallel_for(
      "userBcFlds_rmin",
      NTT2DRange({mblock.i_min, mblock.j_min}, {mblock.i_min + 1, mblock.j_max}),
      Lambda(index_t i, index_t j) {
        auto i_ {static_cast<real_t>(i - N_GHOSTS)};
        auto j_ {static_cast<real_t>(j - N_GHOSTS)};

        auto [r_, th_] = mblock.grid->coord_CU_to_Sph(i_, j_ + HALF);
        auto etheta_hat = omega * std::sin(th_);

        mblock.em_fields(i, j, fld::ex3) = 0.0;
        mblock.em_fields(i, j, fld::ex2) = mblock.grid->vec_HAT_to_CNT_x2(etheta_hat, i_, j_ + HALF);

        auto br_hat {ONE};
        mblock.em_fields(i, j, fld::bx1) = mblock.grid->vec_HAT_to_CNT_x1(br_hat, i_, j_ + HALF);
      });

    Kokkos::parallel_for(
      "userBcFlds_rmax",
      NTT2DRange({mblock.i_max, mblock.j_min}, {mblock.i_max + 1, mblock.j_max}),
      Lambda(index_t i, index_t j) {
        mblock.em_fields(i, j, fld::ex3) = 0.0;
        mblock.em_fields(i, j, fld::ex2) = 0.0;
        mblock.em_fields(i, j, fld::bx1) = 0.0;
    });
  }

  template <>
  void ProblemGenerator<THREE_D>::userBCFields(const real_t&,
                                               SimulationParams&,
                                               Meshblock<THREE_D>&) {}

}
template struct ntt::ProblemGenerator<ntt::ONE_D>;
template struct ntt::ProblemGenerator<ntt::TWO_D>;
template struct ntt::ProblemGenerator<ntt::THREE_D>;
