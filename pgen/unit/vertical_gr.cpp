#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"
#include "init_fields.hpp"

#include <cmath>
#include <iostream>

namespace ntt {

  template <Dimension D, SimulationType S>
  ProblemGenerator<D, S>::ProblemGenerator(const SimulationParams&) {
  }

  // * * * * * * * * * * * * * * * * * * * * * * * *
  // Field initializers
  // . . . . . . . . . . . . . . . . . . . . . . . .
  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::GRPIC>&) {}

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::GRPIC>& mblock) {
    
    // RealFieldND<Dimension::TWO_D, 1> Bru0 ("Br0", mblock.res()[0] + 2 * N_GHOSTS, mblock.res()[1] + 2 * N_GHOSTS);
    // Kokkos::parallel_for(
    //   "userInitFlds", mblock.loopActiveCells(), 
    //   // init_fields_potential<Dimension::TWO_D>(mblock, epsilon, Aphi, Ar, At, Bru0)
    //   init_fields_potential<Dimension::TWO_D>(mblock, epsilon, Aphi, Ar, At)
    //   );

    Kokkos::parallel_for(
      "userInitFlds",
       mblock.loopActiveCells(),
       init_fields_potential<Dimension::TWO_D>(mblock, epsilon, A0, A1, A3)
      );
  }

  // template <>
  // void ProblemGenerator<Dimension::TWO_D, SimulationType::GRPIC>::userInitFields(
  //   const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::GRPIC>& mblock) {
  //   using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
  //   Kokkos::deep_copy(mblock.em, 0.0);
  //   Kokkos::parallel_for(
  //     "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
  //       real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
  //       real_t j_ {static_cast<real_t>(j - N_GHOSTS)};

  //       coord_t<Dimension::TWO_D> rth_;
  //       mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth_);

  //       // Vertical field
  //       real_t br_hat {std::cos(rth_[1])};
  //       real_t bth_hat {std::sin(rth_[1])};
  //       vec_t<Dimension::THREE_D> b_cntr;
  //       mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {br_hat, bth_hat, ZERO}, b_cntr);
  //       mblock.em(i, j, em::bx1) = b_cntr[0];        
  //       mblock.em(i, j, em::bx2) = b_cntr[1];
  //     });
  // }

  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::GRPIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::GRPIC>&) {}

  // * * * * * * * * * * * * * * * * * * * * * * * *
  // Field boundary conditions
  // . . . . . . . . . . . . . . . . . . . . . . . .

  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::GRPIC>::userBCFields(
    const real_t&, const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::GRPIC>&) {}

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::GRPIC>::userBCFields(
  const real_t& time, const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::GRPIC>& mblock) {
   using index_t = NTTArray<real_t**>::size_type;
    (void) time;
    Kokkos::parallel_for(
      "userBcFlds_rmin",
      NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_min() + 1, mblock.j_max()}),
      Lambda(index_t i, index_t j) {
        
        real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
        real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
        real_t br_target  {userTargetField_br_cntrv(mblock, {i_, j_ + HALF})};

        mblock.em(i, j, em::ex3) = ZERO; //mblock.em(i + 1, j, em::ex3); //ZERO;
        mblock.em(i, j, em::ex2) = ZERO; //mblock.em(i + 1, j, em::ex2); //ZERO;
        mblock.em(i, j, em::bx1) = ZERO; //mblock.em(i + 1, j, em::bx1); //br_target;
      });

    Kokkos::parallel_for(
      "userBcFlds_rmax",
      NTTRange<Dimension::TWO_D>({mblock.i_max(), mblock.j_min()}, {mblock.i_max() + 1, mblock.j_max()}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::ex3) = ZERO;
        mblock.em(i, j, em::ex2) = ZERO;
        mblock.em(i, j, em::bx1) = ZERO;
      });
  }

  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::GRPIC>::userBCFields(
    const real_t&, const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::GRPIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dimension::ONE_D, ntt::SimulationType::GRPIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::TWO_D, ntt::SimulationType::GRPIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::THREE_D, ntt::SimulationType::GRPIC>;
