#include "wrapper.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

#include <cmath>
#include <iostream>

namespace ntt {

  template <Dimension D, SimulationType S>
  ProblemGenerator<D, S>::ProblemGenerator(const SimulationParams& sim_params) {
    m_nx1       = readFromInput<int>(sim_params.inputdata(), "problem", "nx1", 1);
    m_nx2       = readFromInput<int>(sim_params.inputdata(), "problem", "nx2", 1);
    m_amplitude = readFromInput<real_t>(sim_params.inputdata(), "problem", "amplitude", 1.0);
  }

  template <>
  void ProblemGenerator<Dim1, TypePIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim1, TypePIC>&) {}

  template <>
  void ProblemGenerator<Dim2, TypePIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim2, TypePIC>& mblock) {

    real_t sx {mblock.metric.x1_max - mblock.metric.x1_min};
    real_t sy {mblock.metric.x2_max - mblock.metric.x2_min};

    auto   kx {(real_t)(constant::TWO_PI)*m_nx1 / sx};
    auto   ky {(real_t)(constant::TWO_PI)*m_nx2 / sy};
    real_t ex_ampl, ey_ampl, bz_ampl {m_amplitude};
    ex_ampl = -ky;
    ey_ampl = kx;
    ex_ampl = m_amplitude * ex_ampl / math::sqrt(ex_ampl * ex_ampl + ey_ampl * ey_ampl);
    ey_ampl = m_amplitude * ey_ampl / math::sqrt(ex_ampl * ex_ampl + ey_ampl * ey_ampl);
    Kokkos::parallel_for(
      "userInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        // index to code units
        real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)},
          j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};

        // code units to cartesian (physical units)
        coord_t<Dim2> xy_, xy_half;
        mblock.metric.x_Code2Cart({i_, j_}, xy_);
        mblock.metric.x_Code2Cart({i_ + HALF, j_ + HALF}, xy_half);

        // hatted fields
        real_t ex_hat {ex_ampl * math::sin(kx * xy_half[0] + ky * xy_[1])};
        real_t ey_hat {ey_ampl * math::sin(kx * xy_[0] + ky * xy_half[1])};
        real_t bz_hat {bz_ampl * math::sin(kx * xy_half[0] + ky * xy_half[1])};

        vec_t<Dim3> ex_cntr, ey_cntr, bz_cntr;
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_}, {ex_hat, ZERO, ZERO}, ex_cntr);
        mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {ZERO, ey_hat, ZERO}, ey_cntr);
        mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bz_hat}, bz_cntr);

        mblock.em(i, j, em::ex1) = ex_cntr[0];
        mblock.em(i, j, em::ex2) = ey_cntr[1];
        mblock.em(i, j, em::bx3) = bz_cntr[2];
      });
  }

  template <>
  void ProblemGenerator<Dim3, TypePIC>::userInitFields(
    const SimulationParams&, Meshblock<Dim3, TypePIC>&) {}

} // namespace ntt

template struct ntt::ProblemGenerator<ntt::Dim1, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::TypePIC>;
