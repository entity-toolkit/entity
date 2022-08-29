#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

#include <cmath>
#include <iostream>

namespace ntt {

  template <Dimension D, SimulationType S>
  ProblemGenerator<D, S>::ProblemGenerator(const SimulationParams&) {}

  // * * * * * * * * * * * * * * * * * * * * * * * *
  // Field initializers
  // . . . . . . . . . . . . . . . . . . . . . . . .

  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    
    Kokkos::deep_copy(mblock.em, 0.0);
    real_t r_min {mblock.metric.x1_min};
    Kokkos::parallel_for(
      "userInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
        real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

        coord_t<Dimension::TWO_D> rth_;
        mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth_);

        real_t                    br_hat {ONE * r_min * r_min / (rth_[0] * rth_[0])};
        vec_t<Dimension::THREE_D> br_cntr;
        mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {br_hat, ZERO, ZERO}, br_cntr);
        mblock.em(i, j, em::bx1) = br_cntr[0];
      });
  }

  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::PIC>&) {}

  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userInitFields(
    const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::PIC>&) {}

  // * * * * * * * * * * * * * * * * * * * * * * * *
  // Field boundary conditions
  // . . . . . . . . . . . . . . . . . . . . . . . .
  template <>
  void ProblemGenerator<Dimension::TWO_D, SimulationType::PIC>::userBCFields(
    const real_t& time, const SimulationParams&, Meshblock<Dimension::TWO_D, SimulationType::PIC>& mblock) {
    
    real_t omega;
    if (time < 5) {
      omega = time / 100.0;
    } else {
      omega = 0.05;
    }
    Kokkos::parallel_for(
      "userBcFlds_rmin",
      CreateRangePolicy<Dimension::TWO_D>({mblock.i1_min(), mblock.i2_min()}, {mblock.i1_min() + 1, mblock.i2_max()}),
      Lambda(index_t i, index_t j) {
        real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
        real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

        coord_t<Dimension::TWO_D> rth1_;
        mblock.metric.x_Code2Sph({i_, j_ + HALF}, rth1_);

        real_t                    etheta_hat {omega * math::sin(rth1_[1])};
        vec_t<Dimension::THREE_D> etheta_cntr, br_cntr;
        mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {ZERO, etheta_hat, ZERO}, etheta_cntr);

        mblock.em(i, j, em::ex3) = 0.0;
        mblock.em(i, j, em::ex2) = etheta_cntr[1];

        real_t br_hat {ONE};

        mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {br_hat, ZERO, ZERO}, br_cntr);
        mblock.em(i, j, em::bx1) = br_cntr[0];
      });

    Kokkos::parallel_for(
      "userBcFlds_rmax",
      CreateRangePolicy<Dimension::TWO_D>({mblock.i1_max(), mblock.i2_min()}, {mblock.i1_max() + 1, mblock.i2_max()}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::ex3) = 0.0;
        mblock.em(i, j, em::ex2) = 0.0;
        mblock.em(i, j, em::bx1) = 0.0;
      });
  }

  template <>
  void ProblemGenerator<Dimension::ONE_D, SimulationType::PIC>::userBCFields(
    const real_t&, const SimulationParams&, Meshblock<Dimension::ONE_D, SimulationType::PIC>&) {}

  template <>
  void ProblemGenerator<Dimension::THREE_D, SimulationType::PIC>::userBCFields(
    const real_t&, const SimulationParams&, Meshblock<Dimension::THREE_D, SimulationType::PIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;