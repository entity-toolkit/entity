#include "global.h"
#include "constants.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

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
  Kokkos::parallel_for(
    "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
      // real_t rr {}
      // real_t r0 {mblock.m_coord_system->m_parameters[0]};
      // real_t xi {mblock.convert_iTOx1(i)};
      // real_t rr {r0 * std::exp(xi / r0)};
      real_t rr {mblock.convert_iTOx1(i)};
      real_t r0 {mblock.convert_iTOx1(N_GHOSTS)};
      mblock.em_fields(i, j, fld::bx1) = ONE * r0 * r0 / (rr * rr);
  });
}

template <>
void ProblemGenerator<THREE_D>::userInitFields(SimulationParams&,
                                               Meshblock<THREE_D>&) {}

// * * * * * * * * * * * * * * * * * * * * * * * *
// Field boundary conditions
// . . . . . . . . . . . . . . . . . . . . . . . .
template <>
void ProblemGenerator<ONE_D>::userBCFields_x1min(SimulationParams&,
                                                 Meshblock<ONE_D>&) {}

template <>
void ProblemGenerator<TWO_D>::userBCFields_x1min(SimulationParams& sim_params,
                                                 Meshblock<TWO_D>& mblock) {
  UNUSED(sim_params);
  using index_t = NTTArray<real_t**>::size_type;
  Kokkos::parallel_for(
    "userBcFlds",
    NTT2DRange({mblock.get_imin(), mblock.get_jmin()}, {mblock.get_imin() + 1, mblock.get_jmax()}),
    Lambda(index_t i, index_t j) {
      mblock.em_fields(i, j, fld::ex3) = 0.0;

      real_t theta {mblock.convert_jTOx2(j)};
      real_t dtheta {(mblock.m_extent[3] - mblock.m_extent[2]) / static_cast<real_t>(mblock.m_resolution[1])};
      mblock.em_fields(i, j, fld::ex2) = 0.1 * std::sin(theta + dtheta * 0.5);

      mblock.em_fields(i, j, fld::bx1) = 1.0;
  });
}

template <>
void ProblemGenerator<THREE_D>::userBCFields_x1min(SimulationParams&,
                                                   Meshblock<THREE_D>&) {}

template <>
void ProblemGenerator<ONE_D>::userBCFields_x1max(SimulationParams&,
                                                 Meshblock<ONE_D>&) {}

template <>
void ProblemGenerator<TWO_D>::userBCFields_x1max(SimulationParams& sim_params,
                                                 Meshblock<TWO_D>& mblock) {
  UNUSED(sim_params);
  using index_t = NTTArray<real_t**>::size_type;
  Kokkos::parallel_for(
    "userBcFlds",
    NTT2DRange({mblock.get_imax(), mblock.get_jmin()}, {mblock.get_imax() + 1, mblock.get_jmax()}),
    Lambda(index_t i, index_t j) {
      mblock.em_fields(i, j, fld::ex3) = 0.0;
      mblock.em_fields(i, j, fld::ex2) = 0.0;
      mblock.em_fields(i, j, fld::bx1) = 0.0;
  });
}

template <>
void ProblemGenerator<THREE_D>::userBCFields_x1max(SimulationParams&,
                                                  Meshblock<THREE_D>&) {}


}
template struct ntt::ProblemGenerator<ntt::ONE_D>;
template struct ntt::ProblemGenerator<ntt::TWO_D>;
template struct ntt::ProblemGenerator<ntt::THREE_D>;
