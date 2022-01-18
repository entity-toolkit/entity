#include "global.h"
#include "pic.h"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {
  /**
   * @brief 1d periodic field bc.
   *
   */
  template <>
  void PIC<Dimension::ONE_D>::fieldBoundaryConditions(const real_t&) {
    using index_t = typename RealFieldND<Dimension::ONE_D, 6>::size_type;
#if METRIC == MINKOWSKI_METRIC
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      auto mblock {this->m_mblock};
      auto range_m {NTTRange<Dimension::ONE_D>({0}, {m_mblock.i_min()})};
      auto range_p {NTTRange<Dimension::ONE_D>({m_mblock.i_max()}, {m_mblock.i_max() + N_GHOSTS})};
      auto ni {m_mblock.Ni()};
      // in x1_min
      Kokkos::parallel_for(
        "1d_bc_x1m", range_m, Lambda(index_t i) {
          mblock.em(i, em::ex1) = mblock.em(i + ni, em::ex1);
          mblock.em(i, em::ex2) = mblock.em(i + ni, em::ex2);
          mblock.em(i, em::ex3) = mblock.em(i + ni, em::ex3);
          mblock.em(i, em::bx1) = mblock.em(i + ni, em::bx1);
          mblock.em(i, em::bx2) = mblock.em(i + ni, em::bx2);
          mblock.em(i, em::bx3) = mblock.em(i + ni, em::bx3);
        });
      // in x1_max
      Kokkos::parallel_for(
        "1d_bc_x1p", range_p, Lambda(index_t i) {
          mblock.em(i, em::ex1) = mblock.em(i - ni, em::ex1);
          mblock.em(i, em::ex2) = mblock.em(i - ni, em::ex2);
          mblock.em(i, em::ex3) = mblock.em(i - ni, em::ex3);
          mblock.em(i, em::bx1) = mblock.em(i - ni, em::bx1);
          mblock.em(i, em::bx2) = mblock.em(i - ni, em::bx2);
          mblock.em(i, em::bx3) = mblock.em(i - ni, em::bx3);
        });
    } else {
      NTTError("boundary condition not implemented");
    }
#else
    (void)(index_t{});
    NTTError("only minkowski possible in 1d");
#endif
  }

  /**
   * @brief 2d periodic field bc.
   *
   */
  template <>
  void PIC<Dimension::TWO_D>::fieldBoundaryConditions(const real_t& t) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
#if METRIC == MINKOWSKI_METRIC
    (void)(t); // ignore warning about unused parameter
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      // non-periodic
      auto range_m {NTTRange<Dimension::TWO_D>({0, m_mblock.j_min()}, {m_mblock.i_min(), m_mblock.j_max()})};
      auto range_p {NTTRange<Dimension::TWO_D>({m_mblock.i_max(), m_mblock.j_min()},
                                               {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max()})};
      auto ni {m_mblock.Ni()};
      auto mblock {this->m_mblock};
      Kokkos::parallel_for(
        "2d_bc_x1m", range_m, Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i + ni, j, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i + ni, j, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i + ni, j, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i + ni, j, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i + ni, j, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i + ni, j, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x1p", range_p, Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i - ni, j, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i - ni, j, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i - ni, j, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i - ni, j, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i - ni, j, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i - ni, j, em::bx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
    // corners are included in x2
    if (m_mblock.boundaries[1] == BoundaryCondition::PERIODIC) {
      RangeND<Dimension::TWO_D> range_m, range_p;
      if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
        // double periodic boundaries
        range_m = NTTRange<Dimension::TWO_D>({0, 0}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_min()});
        range_p = NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()},
                                             {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() + N_GHOSTS});
      } else {
        // single periodic (only x2-periodic)
        range_m = NTTRange<Dimension::TWO_D>({m_mblock.i_min(), 0}, {m_mblock.i_max(), m_mblock.j_min()});
        range_p = NTTRange<Dimension::TWO_D>({m_mblock.i_min(), m_mblock.j_max()},
                                             {m_mblock.i_max(), m_mblock.j_max() + N_GHOSTS});
      }
      auto nj {m_mblock.Nj()};
      auto mblock {this->m_mblock};
      Kokkos::parallel_for(
        "2d_bc_x2m", range_m, Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i, j + nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i, j + nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i, j + nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i, j + nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i, j + nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i, j + nj, em::bx3);
        });
      Kokkos::parallel_for(
        "2d_bc_x2p", range_p, Lambda(index_t i, index_t j) {
          mblock.em(i, j, em::ex1) = mblock.em(i, j - nj, em::ex1);
          mblock.em(i, j, em::ex2) = mblock.em(i, j - nj, em::ex2);
          mblock.em(i, j, em::ex3) = mblock.em(i, j - nj, em::ex3);
          mblock.em(i, j, em::bx1) = mblock.em(i, j - nj, em::bx1);
          mblock.em(i, j, em::bx2) = mblock.em(i, j - nj, em::bx2);
          mblock.em(i, j, em::bx3) = mblock.em(i, j - nj, em::bx3);
        });
    } else {
      // non-periodic
      NTTError("2d boundary condition for minkowski not implemented");
    }
#elif (METRIC == SPHERICAL_METRIC) || (METRIC == QSPHERICAL_METRIC)
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
      });
    // theta = pi boundary
    Kokkos::parallel_for(
      "2d_bc_thetaPi",
      NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() + N_GHOSTS}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::bx2) = 0.0;
        mblock.em(i, j, em::ex3) = 0.0;
      });

    auto r_absorb {m_sim_params.metric_parameters(2)};
    auto r_max {m_mblock.metric->x1_max};
    auto pGen {this->m_pGen};
    Kokkos::parallel_for(
      "2d_absorbing bc", m_mblock.loopActiveCells(), Lambda(index_t i, index_t j) {
        real_t i_ {static_cast<real_t>(i)};
        real_t j_ {static_cast<real_t>(j)};

        // i
        vec_t<Dimension::TWO_D> rth_;
        mblock.metric->x_Code2Sph({i_, j_}, rth_);
        real_t delta_r1 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
        real_t sigma_r1 {HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1};
        // i + 1/2
        mblock.metric->x_Code2Sph({i_ + HALF, j_}, rth_);
        real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
        real_t sigma_r2 {HEAVISIDE(delta_r2) * delta_r2 * delta_r2 * delta_r2};

        mblock.em(i, j, em::ex1) = (ONE - sigma_r1) * mblock.em(i, j, em::ex1);
        mblock.em(i, j, em::bx2) = (ONE - sigma_r1) * mblock.em(i, j, em::bx2);
        mblock.em(i, j, em::bx3) = (ONE - sigma_r1) * mblock.em(i, j, em::bx3);

        real_t br_target_hat {pGen.userTargetField_br_hat(mblock, {i_, j_ + HALF})};
        real_t bx1_source_cntr {mblock.em(i, j, em::bx1)};
        vec_t<Dimension::THREE_D> br_source_hat;
        mblock.metric->v_Cntrv2Hat({i_, j_ + HALF}, {bx1_source_cntr, ZERO, ZERO}, br_source_hat);
        real_t br_interm_hat {(ONE - sigma_r2) * br_source_hat[0] + sigma_r2 * br_target_hat};
        vec_t<Dimension::THREE_D> br_interm_cntr;
        mblock.metric->v_Hat2Cntrv({i_, j_ + HALF}, {br_interm_hat, ZERO, ZERO}, br_interm_cntr);
        mblock.em(i, j, em::bx1) = br_interm_cntr[0];
        mblock.em(i, j, em::ex2) = (ONE - sigma_r2) * mblock.em(i, j, em::ex2);
        mblock.em(i, j, em::ex3) = (ONE - sigma_r2) * mblock.em(i, j, em::ex3);
      });
#else
    (void)(index_t{});
    NTTError("2d boundary condition for metric not implemented");
#endif
  }

  /**
   * @brief 3d periodic field bc.
   *
   */
  template <>
  void PIC<Dimension::THREE_D>::fieldBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt
