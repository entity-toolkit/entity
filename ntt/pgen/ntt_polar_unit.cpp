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
    real_t dx1 {mblock.get_dx1()};
    real_t dx2 {mblock.get_dx2()};
    Kokkos::parallel_for(
      "userInitFlds",
      mblock.loopActiveCells(),
      Lambda(index_t i, index_t j) {
        auto x1 {mblock.convert_iTOx1(i)};
        auto x2 {mblock.convert_jTOx2(j)};

        auto r0 {mblock.m_coord_system->getSpherical_r(mblock.convert_iTOx1(N_GHOSTS), ZERO)};
        auto rr {mblock.m_coord_system->getSpherical_r(x1, ZERO)};

        auto bx1 {ONE * r0 * r0 / (rr * rr)};
        bx1 = mblock.m_coord_system->convert_LOC_to_CNT_x1(bx1, x1, x2 + 0.5 * dx1);
        mblock.em_fields(i, j, fld::bx1) = bx1;
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
    real_t dx1 {mblock.get_dx1()};
    real_t dx2 {mblock.get_dx2()};
    real_t omega {0.1 * std::sin(time)};
    // if (time < 0.5) {
    //   omega = time / 10.0;
    // } else {
    //   omega = 0.05;
    // }
    Kokkos::parallel_for(
      "userBcFlds_rmax",
      NTT2DRange({mblock.get_imin(), mblock.get_jmin()}, {mblock.get_imin() + 1, mblock.get_jmax()}),
      Lambda(index_t i, index_t j) {
        auto x1 {mblock.convert_iTOx1(i)};
        auto x2 {mblock.convert_jTOx2(j)};

        mblock.em_fields(i, j, fld::ex3) = 0.0;

        auto theta {mblock.m_coord_system->getSpherical_theta(ZERO, x2 + 0.5 * dx2)};
        auto ex2 {omega * std::sin(theta)};
        ex2 = mblock.m_coord_system->convert_LOC_to_CNT_x2(ex2, x1, x2 + 0.5 * dx2);
        mblock.em_fields(i, j, fld::ex2) = ex2;

        auto bx1 {1.0};
        bx1 = mblock.m_coord_system->convert_LOC_to_CNT_x1(bx1, x1, x2 + 0.5 * dx1);
        mblock.em_fields(i, j, fld::bx1) = bx1;
    });

    Kokkos::parallel_for(
      "userBcFlds_rmin",
      NTT2DRange({mblock.get_imax(), mblock.get_jmin()}, {mblock.get_imax() + 1, mblock.get_jmax()}),
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
