#include "global.h"
#include "constants.h"
#include "sim_params.h"
#include "meshblock.h"

#include <cmath>

namespace ntt {

ProblemGenerator::ProblemGenerator(SimulationParams &sim_params) {
  UNUSED(sim_params);
  // auto timestep = readFromInput<real_t>(sim_params.m_inputdata, "algorithm",
  // "timestep"); PLOGD << timestep << "\n";
}

template <> void ProblemGenerator::userInitFields<One_D>(SimulationParams &sim_params, Meshblock<One_D> &mblock) {
  using size_type = NTTArray<real_t *>::size_type;
  real_t sx1 = mblock.get_x1max() - mblock.get_x1min();
  real_t dx1_half = mblock.get_dx1() * 0.5;
  Kokkos::parallel_for(
      "userInit", loopActiveCells(mblock), Lambda(size_type i) {
        real_t x1 = convert_iTOx1(mblock, i);
        mblock.ex2(i) = std::sin(TWO_PI * x1 / sx1);
        mblock.bx3(i) = std::sin(TWO_PI * (x1 + dx1_half) / sx1);
      });
}

template <> void ProblemGenerator::userInitFields<Two_D>(SimulationParams &sim_params, Meshblock<Two_D> &mblock) {
  using size_type = NTTArray<real_t **>::size_type;
  real_t sx1 = mblock.get_x1max() - mblock.get_x1min();
  real_t dx1_half = mblock.get_dx1() * 0.5;
  auto range = NTT2DRange({mblock.get_imin(), mblock.get_jmin()}, {mblock.get_imax(), mblock.get_jmax()});
  Kokkos::parallel_for(
      "userInit", loopActiveCells(mblock), Lambda(size_type i, size_type j) {
        real_t x1 = convert_iTOx1(mblock, i);
        mblock.ex2(i, j) = std::sin(TWO_PI * x1 / sx1);
        mblock.bx3(i, j) = std::sin(TWO_PI * (x1 + dx1_half) / sx1);
      });
}
template <> void ProblemGenerator::userInitFields<Three_D>(SimulationParams &, Meshblock<Three_D> &) {}

// TODO: this has to be done better
template void ProblemGenerator::userInitFields<One_D>(SimulationParams &, Meshblock<One_D> &);
template void ProblemGenerator::userInitFields<Two_D>(SimulationParams &, Meshblock<Two_D> &);
template void ProblemGenerator::userInitFields<Three_D>(SimulationParams &, Meshblock<Three_D> &);

} // namespace ntt
