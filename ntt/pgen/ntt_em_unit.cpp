#include "global.h"
#include "input.h"
#include "sim_params.h"
#include "meshblock.h"

#include "ntt_em_unit.hpp"

#include <cmath>

namespace ntt {

  template<Dimension D>
  ProblemGenerator<D>::ProblemGenerator(SimulationParams& sim_params) {
    UNUSED(sim_params);
    m_nx1 = readFromInput<int>(sim_params.m_inputdata, "problem", "nx1", 1);
    m_nx2 = readFromInput<int>(sim_params.m_inputdata, "problem", "nx2", 1);
    m_amplitude = readFromInput<real_t>(sim_params.m_inputdata, "problem", "amplitude", 1.0);
  }

  template <>
  void ProblemGenerator<ONE_D>::userInitFields(SimulationParams& sim_params,
                                               Meshblock<ONE_D>& mblock) {
    UNUSED(sim_params);
    using index_t = NTTArray<real_t*>::size_type;
    real_t sx = mblock.m_coord_system->x1max_PHU() - mblock.m_coord_system->x1min_PHU();
    Kokkos::parallel_for(
        "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i) {
          auto i_ {(real_t)(i)}, i_half {(real_t)(i + 0.5)};

          real_t x_ {mblock.m_coord_system->coord_CU_to_Cart(i_)};
          real_t x_half {mblock.m_coord_system->coord_CU_to_Cart(i_half)};

          auto ey_hat {std::sin(TWO_PI * x_ / sx)};
          auto bz_hat {std::sin(TWO_PI * x_half / sx)};

          mblock.em_fields(i, fld::ex2) = mblock.m_coord_system->vec_HAT_to_CNT_x2(ey_hat, i_);
          mblock.em_fields(i, fld::bx3) = mblock.m_coord_system->vec_HAT_to_CNT_x3(bz_hat, i_half);
        });
  }

  template <>
  void ProblemGenerator<TWO_D>::userInitFields(SimulationParams& sim_params,
                                               Meshblock<TWO_D>& mblock) {
    UNUSED(sim_params);
    using index_t = NTTArray<real_t**>::size_type;
    real_t sx = mblock.m_coord_system->x1max_PHU() - mblock.m_coord_system->x1min_PHU();
    real_t sy = mblock.m_coord_system->x2max_PHU() - mblock.m_coord_system->x2min_PHU();

    auto kx {TWO_PI * m_nx1 / sx};
    auto ky {TWO_PI * m_nx2 / sy};
    real_t ex_ampl, ey_ampl, bz_ampl {m_amplitude};
    ex_ampl = -ky;
    ey_ampl = kx;
    ex_ampl = m_amplitude * ex_ampl / std::sqrt(ex_ampl * ex_ampl + ey_ampl * ey_ampl);
    ey_ampl = m_amplitude * ey_ampl / std::sqrt(ex_ampl * ex_ampl + ey_ampl * ey_ampl);
    Kokkos::parallel_for(
        "userInitFlds", mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
          auto i_ {(real_t)(i)}, j_ {(real_t)(j)};
          auto i_half {(real_t)(i + 0.5)}, j_half {(real_t)(j + 0.5)};

          auto [x_, y_] = mblock.m_coord_system->coord_CU_to_Cart(i_, j_);
          auto [x_half, y_half] = mblock.m_coord_system->coord_CU_to_Cart(i_half, j_half);

          auto ex_hat = ex_ampl * std::sin(kx * x_half + ky * y_);
          auto ey_hat = ey_ampl * std::sin(kx * x_ + ky * y_half);
          auto bz_hat = bz_ampl * std::sin(kx * x_half + ky * y_half);

          mblock.em_fields(i, j, fld::ex1) = mblock.m_coord_system->vec_HAT_to_CNT_x1(ex_hat, i_half, j_);
          mblock.em_fields(i, j, fld::ex2) = mblock.m_coord_system->vec_HAT_to_CNT_x2(ex_hat, i_, j_half);
          mblock.em_fields(i, j, fld::bx3) = mblock.m_coord_system->vec_HAT_to_CNT_x3(bz_hat, i_half, j_half);
        });
  }

  template <>
  void ProblemGenerator<THREE_D>::userInitFields(SimulationParams&,
                                                 Meshblock<THREE_D>&) {}

  template <>
  void ProblemGenerator<ONE_D>::userInitParticles(SimulationParams&,
                                                  Meshblock<ONE_D>&) {}

  template <>
  void ProblemGenerator<TWO_D>::userInitParticles(SimulationParams&,
                                                  Meshblock<TWO_D>&) {}

  template <>
  void ProblemGenerator<THREE_D>::userInitParticles(SimulationParams&,
                                                    Meshblock<THREE_D>&) {}
                                                    
} // namespace ntt

template struct ntt::ProblemGenerator<ntt::ONE_D>;
template struct ntt::ProblemGenerator<ntt::TWO_D>;
template struct ntt::ProblemGenerator<ntt::THREE_D>;
