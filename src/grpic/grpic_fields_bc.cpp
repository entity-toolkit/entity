#include "global.h"
#include "grpic.h"
#include "grpic_fields_bc.hpp"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

  template <>
  void GRPIC<Dimension::TWO_D>::fieldBoundaryConditions(const real_t& t, const gr_bc& f) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    (void)t;
    auto mblock {this->m_mblock};
    if (f == gr_bc::Dfield) {
      // r = rmin boundary
      auto i_min {mblock.i_min()};
      Kokkos::parallel_for(
        "2d_bc_rmin", NTTRange<Dimension::ONE_D>({mblock.j_min()}, {mblock.j_max() + 1}), Lambda(index_t j) {
          mblock.em0(i_min - 1, j, em::ex1) = mblock.em0(i_min, j, em::ex1);
          mblock.em0(i_min, j, em::ex2)     = mblock.em0(i_min + 1, j, em::ex2);
          mblock.em0(i_min - 1, j, em::ex2) = mblock.em0(i_min, j, em::ex2);
          mblock.em0(i_min, j, em::ex3)     = mblock.em0(i_min + 1, j, em::ex3);
          mblock.em0(i_min - 1, j, em::ex3) = mblock.em0(i_min, j, em::ex3);

          mblock.em(i_min - 1, j, em::ex1) = mblock.em(i_min, j, em::ex1);
          mblock.em(i_min, j, em::ex2)     = mblock.em(i_min + 1, j, em::ex2);
          mblock.em(i_min - 1, j, em::ex2) = mblock.em(i_min, j, em::ex2);
          mblock.em(i_min, j, em::ex3)     = mblock.em(i_min + 1, j, em::ex3);
          mblock.em(i_min - 1, j, em::ex3) = mblock.em(i_min, j, em::ex3);
        });
      // Absorbing boundary
      auto r_absorb {m_sim_params.metric_parameters()[2]};
      auto absorb_coeff {m_sim_params.metric_parameters()[3]};
      auto absorb_norm {ONE / (ONE - math::exp(absorb_coeff))};
      auto r_max {m_mblock.metric.x1_max};
      // auto pGen {this->m_pGen};
      Kokkos::parallel_for(
        "2d_absorbing bc",
        NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_max() + 1, mblock.j_max() + 1}),
        Lambda(index_t i, index_t j) {
          real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
          real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

          // i
          vec_t<Dimension::TWO_D> rth_;
          mblock.metric.x_Code2Sph({i_, j_}, rth_);
          if (rth_[0] > r_absorb) {
            real_t delta_r1 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
            real_t sigma_r1 {absorb_norm * (ONE - math::exp(absorb_coeff * HEAVISIDE(delta_r1) * CUBE(delta_r1)))};

            mblock.em0(i, j, em::ex2) = (ONE - sigma_r1) * mblock.em0(i, j, em::ex2);
            mblock.em0(i, j, em::ex3) = (ONE - sigma_r1) * mblock.em0(i, j, em::ex3);

            mblock.em(i, j, em::ex2) = (ONE - sigma_r1) * mblock.em(i, j, em::ex2);
            mblock.em(i, j, em::ex3) = (ONE - sigma_r1) * mblock.em(i, j, em::ex3);
          }
          // i + 1/2
          mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
          if (rth_[0] > r_absorb) {
            real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
            real_t sigma_r2 {absorb_norm * (ONE - math::exp(absorb_coeff * HEAVISIDE(delta_r2) * CUBE(delta_r2)))};

            mblock.em0(i, j, em::ex1) = (ONE - sigma_r2) * mblock.em0(i, j, em::ex1);
            mblock.em(i, j, em::ex1)  = (ONE - sigma_r2) * mblock.em(i, j, em::ex1);
          }
        });
      // r = rmax
      auto i_max {mblock.i_max()};
      Kokkos::parallel_for(
        "2d_bc_rmax", NTTRange<Dimension::ONE_D>({mblock.j_min()}, {mblock.j_max()}), Lambda(index_t j) {
          mblock.em0(i_max, j, em::ex2) = mblock.em0(i_max - 1, j, em::ex2);
          mblock.em0(i_max, j, em::ex3) = mblock.em0(i_max - 1, j, em::ex3);

          mblock.em(i_max, j, em::ex2) = mblock.em(i_max - 1, j, em::ex2);
          mblock.em(i_max, j, em::ex3) = mblock.em(i_max - 1, j, em::ex3);
        });
    } else if (f == gr_bc::Bfield) {
      // theta = 0 boundary
      auto j_min {mblock.j_min()};
      Kokkos::parallel_for(
        "2d_bc_theta0", NTTRange<Dimension::ONE_D>({mblock.i_min() - 1}, {mblock.i_max()}), Lambda(index_t i) {
          mblock.em0(i, j_min, em::bx2) = ZERO;
          mblock.em(i, j_min, em::bx2)  = ZERO;
        });

      // theta = pi boundary
      auto j_max {mblock.j_max()};
      Kokkos::parallel_for(
        "2d_bc_thetaPi", NTTRange<Dimension::ONE_D>({mblock.i_min() - 1}, {m_mblock.i_max()}), Lambda(index_t i) {
          mblock.em0(i, j_max, em::bx2) = ZERO;
          mblock.em(i, j_max, em::bx2)  = ZERO;
        });

      // r = rmin boundary
      auto i_min {mblock.i_min()};
      Kokkos::parallel_for(
        "2d_bc_rmin", NTTRange<Dimension::ONE_D>({mblock.j_min()}, {mblock.j_max() + 1}), Lambda(index_t j) {
          mblock.em0(i_min, j, em::bx1)     = mblock.em0(i_min + 1, j, em::bx1);
          mblock.em0(i_min - 1, j, em::bx1) = mblock.em0(i_min, j, em::bx1);
          mblock.em0(i_min - 1, j, em::bx2) = mblock.em0(i_min, j, em::bx2);
          mblock.em0(i_min - 1, j, em::bx3) = mblock.em0(i_min, j, em::bx3);

          mblock.em(i_min, j, em::bx1)     = mblock.em(i_min + 1, j, em::bx1);
          mblock.em(i_min - 1, j, em::bx1) = mblock.em(i_min, j, em::bx1);
          mblock.em(i_min - 1, j, em::bx2) = mblock.em(i_min, j, em::bx2);
          mblock.em(i_min - 1, j, em::bx3) = mblock.em(i_min, j, em::bx3);
        });

      auto r_absorb {m_sim_params.metric_parameters()[2]};
      auto r_max {m_mblock.metric.x1_max};
      auto absorb_coeff {m_sim_params.metric_parameters()[3]};
      auto absorb_norm {ONE / (ONE - math::exp(absorb_coeff))};
      Kokkos::parallel_for(
        "2d_absorbing bc",
        NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_max() + 1, mblock.j_max() + 1}),
        GRFieldBC_rmax<Dimension::TWO_D>(m_mblock, this->m_pGen, r_absorb, r_max, absorb_coeff, absorb_norm));

      // r = rmax
      auto i_max {mblock.i_max()};
      Kokkos::parallel_for(
        "2d_bc_rmax", NTTRange<Dimension::ONE_D>({mblock.j_min()}, {mblock.j_max()}), Lambda(index_t j) {
          mblock.em0(i_max, j, em::bx1) = mblock.em0(i_max - 1, j, em::bx1);
          mblock.em(i_max, j, em::bx1)  = mblock.em(i_max - 1, j, em::bx1);
        });
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dimension::THREE_D>::fieldBoundaryConditions(const real_t&, const gr_bc&) {
    NTTError("3D GRPIC not implemented yet");
  }

  template <>
  void GRPIC<Dimension::TWO_D>::auxFieldBoundaryConditions(const real_t&, const gr_bc& f) {
    using index_t = typename RealFieldND<Dimension::TWO_D, 6>::size_type;
    auto mblock {this->m_mblock};
    auto i_min {mblock.i_min()};
    auto range {NTTRange<Dimension::ONE_D>({mblock.j_min()}, {mblock.j_max() + 1})};
    if (f == gr_bc::Efield) {
      // r = rmin boundary
      Kokkos::parallel_for(
        "2d_bc_rmin", range, Lambda(index_t j) {
          mblock.aux(i_min - 1, j, em::ex1) = mblock.aux(i_min, j, em::ex1);
          mblock.aux(i_min - 1, j, em::ex2) = mblock.aux(i_min, j, em::ex2);
          mblock.aux(i_min - 1, j, em::ex3) = mblock.aux(i_min, j, em::ex3);
        });
    } else if (f == gr_bc::Hfield) {
      // r = rmin boundary
      Kokkos::parallel_for(
        "2d_bc_rmin", range, Lambda(index_t j) {
          mblock.aux(i_min - 1, j, em::bx1) = mblock.aux(i_min, j, em::bx1);
          mblock.aux(i_min - 1, j, em::bx2) = mblock.aux(i_min, j, em::bx2);
          mblock.aux(i_min - 1, j, em::bx3) = mblock.aux(i_min, j, em::bx3);
        });
    } else {
      NTTError("Wrong option for `f`");
    }
  }

  template <>
  void GRPIC<Dimension::THREE_D>::auxFieldBoundaryConditions(const real_t&, const gr_bc&) {
    NTTError("3D GRPIC not implemented yet");
  }

} // namespace ntt

// !LEGACY
// // theta = 0 boundary
// Kokkos::parallel_for(
//   "2d_bc_theta0",
//   NTTRange<Dimension::TWO_D>({0, 0}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_min() + 1}),
//   Lambda(index_t i, index_t j) {
//     // mblock.em0(i, j, em::ex3) = ZERO;
//     // mblock.em(i, j, em::ex3) = ZERO;
//     });

// // theta = pi boundary
// Kokkos::parallel_for(
//   "2d_bc_thetaPi",
//   NTTRange<Dimension::TWO_D>({0, m_mblock.j_max()}, {m_mblock.i_max() + N_GHOSTS, m_mblock.j_max() +
//   N_GHOSTS}), Lambda(index_t i, index_t j) {
//     mblock.em0(i, j, em::ex3) = ZERO;

//     mblock.em(i, j, em::ex3) = ZERO;
//     });

// auto j_min {mblock.j_min()};
// // Kokkos::parallel_for(
// //   "2d_bc_theta0", NTTRange<Dimension::ONE_D>({mblock.i_min() - 1}, {mblock.i_max()}), Lambda(index_t i) {
// //     // mblock.em0(i, j_min, em::ex3) = ZERO;
// //     // mblock.em(i, j_min, em::ex3) = ZERO;
// //     mblock.em0(i, j_min - 1, em::ex2) = -mblock.em0(i, j_min, em::ex2);
// //     mblock.em(i, j_min - 1, em::ex2) = -mblock.em(i, j_min, em::ex2);
// //   });

// // // theta = pi boundary
// // auto j_max {mblock.j_max()};
// // Kokkos::parallel_for(
// //   "2d_bc_thetaPi", NTTRange<Dimension::ONE_D>({mblock.i_min() - 1}, {m_mblock.i_max()}), Lambda(index_t i) {
// //     // mblock.em0(i, j_max, em::ex3) = ZERO;
// //     // mblock.em(i, j_max, em::ex3) = ZERO;
// //     mblock.em0(i, j_max, em::ex2) = mblock.em0(i, j_max - 1, em::ex2);
// //     mblock.em(i, j_max, em::ex2) = mblock.em(i, j_max - 1, em::ex2);
// //   });

// auto pGen {this->m_pGen};
// auto br_func {&(this->m_pGen.userTargetField_br_cntrv)};
// Kokkos::parallel_for(
//   "2d_absorbing bc",
//   NTTRange<Dimension::TWO_D>({mblock.i_min(), mblock.j_min()}, {mblock.i_max() + 1, mblock.j_max() + 1}),
//   Lambda(index_t i, index_t j) {
//     real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
//     real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

//     // i
//     vec_t<Dimension::TWO_D> rth_;
//     mblock.metric.x_Code2Sph({i_, j_}, rth_);
//     if (rth_[0] > r_absorb) {
//       real_t delta_r1 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
//       real_t sigma_r1 {absorb_norm * (ONE - math::exp(absorb_coeff * HEAVISIDE(delta_r1) * CUBE(delta_r1)))};
//       // !HACK
//       // real_t br_target = pGen.userTargetField_br_cntrv(mblock, {i_, j_ + HALF});
//       real_t br_target = br_func(mblock, {i_, j_ + HALF});
//       // real_t br_target {ZERO};
//       mblock.em0(i, j, em::bx1) = (ONE - sigma_r1) * mblock.em0(i, j, em::bx1) + sigma_r1 * br_target;
//       mblock.em(i, j, em::bx1)  = (ONE - sigma_r1) * mblock.em(i, j, em::bx1) + sigma_r1 * br_target;
//     }
//     // i + 1/2
//     mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
//     if (rth_[0] > r_absorb) {
//       real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
//       real_t sigma_r2 {absorb_norm * (ONE - math::exp(absorb_coeff * HEAVISIDE(delta_r2) * CUBE(delta_r2)))};
//       // !HACK
//       // real_t bth_target {pGen->userTargetField_bth_cntrv(mblock, {i_ + HALF, j_})};
//       real_t bth_target {ZERO};
//       mblock.em0(i, j, em::bx2) = (ONE - sigma_r2) * mblock.em0(i, j, em::bx2) + sigma_r2 * bth_target;
//       mblock.em(i, j, em::bx2)  = (ONE - sigma_r2) * mblock.em(i, j, em::bx2) + sigma_r2 * bth_target;
//       mblock.em0(i, j, em::bx3) = (ONE - sigma_r2) * mblock.em0(i, j, em::bx3);
//       mblock.em(i, j, em::bx3)  = (ONE - sigma_r2) * mblock.em(i, j, em::bx3);
//     }
//   });
