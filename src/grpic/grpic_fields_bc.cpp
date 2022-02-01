#include "global.h"
#include "grpic.h"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {
  /**
   * @brief 2d periodic field bc.
   *
   */
  template <>
  void GRPIC<Dimension::TWO_D>::fieldBoundaryConditions(const real_t& t) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;

#if (METRIC == KERR_SCHILD_METRIC) 
    // * * * * * * * * * * * * * * * *
    // axisymmetric spherical grid
    // * * * * * * * * * * * * * * * *
    // r = rmin boundary
    if (m_mblock.boundaries[0] == BoundaryCondition::USER) {
      m_pGen.userBCFields(t, m_sim_params, m_mblock);
    } else {
      NTTError("2d non-user boundary condition not implemented for curvilinear");
    }
    auto mblock {this->m_mblock};
    // theta = 0 boundary
    Kokkos::parallel_for(
      "2d_bc_theta0",
      NTTRange<Dimension::TWO_D>({0, 0}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_min() + 1}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
        mblock.em0(i, j, em::bx2) = 0.0;
        mblock.em0(i, j, em::ex3) = 0.0;
        mblock.aux(i, j, em::bx2) = 0.0;
        mblock.aux(i, j, em::ex3) = 0.0;
      });
    // theta = pi boundary
    Kokkos::parallel_for(
      "2d_bc_thetaPi",
      NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() + N_GHOSTS}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
        mblock.em0(i, j, em::bx2) = 0.0;
        mblock.em0(i, j, em::ex3) = 0.0;
        mblock.aux(i, j, em::bx2) = 0.0;
        mblock.aux(i, j, em::ex3) = 0.0;
      });

    auto r_absorb {m_sim_params.metric_parameters()[2]};
    auto r_max {m_mblock.metric.x1_max};
    auto pGen {this->m_pGen};
    Kokkos::parallel_for(
      "2d_absorbing bc", m_mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {static_cast<real_t>(i)};
        real_t j_ {static_cast<real_t>(j)};

        // i
        vec_t<Dimension::TWO_D> rth_;
        mblock.metric.x_Code2Sph({i_, j_}, rth_);
        real_t delta_r1 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
        real_t sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
        // i + 1/2
        mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
        real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
        real_t sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};

        mblock.em(i, j, em::ex1) = (ONE - sigma_r1) * mblock.em(i, j, em::ex1);
        mblock.em(i, j, em::bx2) = (ONE - sigma_r1) * mblock.em(i, j, em::bx2);
        mblock.em(i, j, em::bx3) = (ONE - sigma_r1) * mblock.em(i, j, em::bx3);

        real_t br_target_hat {pGen.userTargetField_br_hat(mblock, {i_, j_ + HALF})};
        real_t bx1_source_cntr {mblock.em(i, j, em::bx1)};
        vec_t<Dimension::THREE_D> br_source_hat;
        mblock.metric.v_Cntrv2Hat({i_, j_ + HALF}, {bx1_source_cntr, ZERO, ZERO}, br_source_hat);
        real_t br_interm_hat {(ONE - sigma_r2) * br_source_hat[0] + sigma_r2 * br_target_hat};
        vec_t<Dimension::THREE_D> br_interm_cntr;
        mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {br_interm_hat, ZERO, ZERO}, br_interm_cntr);
        mblock.em(i, j, em::bx1) = br_interm_cntr[0];
        mblock.em(i, j, em::ex2) = (ONE - sigma_r2) * mblock.em(i, j, em::ex2);
        mblock.em(i, j, em::ex3) = (ONE - sigma_r2) * mblock.em(i, j, em::ex3);
      });
#else
    (void)(index_t {});
    NTTError("2d boundary condition for Cartesian Kerr-Schild metric not implemented");
#endif
  }

  /**
   * @brief 3d periodic field bc.
   *
   */
  template <>
  void GRPIC<Dimension::THREE_D>::fieldBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt