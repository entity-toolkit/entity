#include "global.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"

#include "input.h"

#include <plog/Log.h>

namespace ntt {

ProblemGenerator::ProblemGenerator(SimulationParams &sim_params) {
  UNUSED(sim_params);
  // auto timestep = readFromInput<real_t>(sim_params.m_inputdata, "algorithm", "timestep");
  // PLOGD << timestep << "\n";
  MyStuff mystuff();
}

template <> void ProblemGenerator::userInitFields<One_D>(SimulationParams &sim_params, Meshblock<One_D> &mblock) {
  // Kokkos::parallel_for("userInit", sim_params.m_resolution[0],
  //   Lambda (index_t i) {
  //     mblock.ex1(i) = 1.0 / static_cast<real_t>(i + 1);
  //     mblock.bx2(i) = 1.0 / static_cast<real_t>(i + 1);
  //   }
  // );
}

// TODO: this has to be done better
template <> void ProblemGenerator::userInitFields<Two_D>(SimulationParams &, Meshblock<Two_D> &) {}
template <> void ProblemGenerator::userInitFields<Three_D>(SimulationParams &, Meshblock<Three_D> &) {}

template void ProblemGenerator::userInitFields<One_D>(SimulationParams &, Meshblock<One_D> &);
template void ProblemGenerator::userInitFields<Two_D>(SimulationParams &, Meshblock<Two_D> &);
template void ProblemGenerator::userInitFields<Three_D>(SimulationParams &, Meshblock<Three_D> &);

} // namespace ntt
