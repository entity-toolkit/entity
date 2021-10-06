#include "global.h"
#include "constants.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include <cmath>

namespace ntt {

ProblemGenerator::ProblemGenerator(SimulationParams& sim_params) {
  UNUSED(sim_params);
  m_nx1 = readFromInput<int>(sim_params.m_inputdata, "problem", "nx1", 1);
  m_nx2 = readFromInput<int>(sim_params.m_inputdata, "problem", "nx2", 1);
  m_amplitude = readFromInput<real_t>(sim_params.m_inputdata, "problem", "amplitude", 1.0);
}

template <>
void ProblemGenerator::userInitFields<One_D>(SimulationParams& sim_params,
                                             Meshblock<One_D>& mblock) {
  UNUSED(sim_params);
  using size_type = NTTArray<real_t*>::size_type;
  real_t sx1 = mblock.get_x1max() - mblock.get_x1min();
  real_t dx1_half = mblock.get_dx1() * 0.5;
  Kokkos::parallel_for(
      "userInit", loopActiveCells(mblock), Lambda(size_type i) {
        real_t x1 = convert_iTOx1(mblock, i);
        mblock.ex2(i) = std::sin(TWO_PI * x1 / sx1);
        mblock.bx3(i) = std::sin(TWO_PI * (x1 + dx1_half) / sx1);
      });
}

template <>
void ProblemGenerator::userInitFields<Two_D>(SimulationParams& sim_params,
                                             Meshblock<Two_D>& mblock) {
  UNUSED(sim_params);
  using size_type = NTTArray<real_t**>::size_type;
  real_t dx1_half = mblock.get_dx1() * 0.5;
  real_t dx2_half = mblock.get_dx2() * 0.5;
  auto kx1 {TWO_PI * m_nx1 / (mblock.get_x1max() - mblock.get_x1min())};
  auto kx2 {TWO_PI * m_nx2 / (mblock.get_x2max() - mblock.get_x2min())};
  real_t ex1_ampl, ex2_ampl, bx3_ampl {m_amplitude};
  ex1_ampl = -kx2;
  ex2_ampl = kx1;
  ex1_ampl = m_amplitude * ex1_ampl / std::sqrt(ex1_ampl * ex1_ampl + ex2_ampl * ex2_ampl);
  ex2_ampl = m_amplitude * ex2_ampl / std::sqrt(ex1_ampl * ex1_ampl + ex2_ampl * ex2_ampl);
  Kokkos::parallel_for(
      "userInit", loopActiveCells(mblock), Lambda(size_type i, size_type j) {
        real_t x1 = convert_iTOx1(mblock, i);
        real_t x2 = convert_jTOx2(mblock, j);
        mblock.ex1(i, j) = ex1_ampl * std::sin(kx1 * (x1 + dx1_half) + kx2 * x2);
        mblock.ex2(i, j) = ex2_ampl * std::sin(kx1 * x1 + kx2 * (x2 + dx2_half));
        mblock.bx3(i, j) = bx3_ampl * std::sin(kx1 * (x1 + dx1_half) + kx2 * (x2 + dx2_half));
      });
}
template <>
void ProblemGenerator::userInitFields<Three_D>(SimulationParams&, Meshblock<Three_D>&) {}

} // namespace ntt