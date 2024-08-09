
// !LEGACY
// // theta = 0 boundary
// Kokkos::parallel_for(
//   "2d_bc_theta0",
//   CreateRangePolicy<Dim2>({0, 0}, {m_mblock.i1_max() + N_GHOSTS, m_mblock.i2_min() + 1}),
//   Lambda(index_t i, index_t j) {
//     // mblock.em0(i, j, em::ex3) = ZERO;
//     // mblock.em(i, j, em::ex3) = ZERO;
//     });

// // theta = pi boundary
// Kokkos::parallel_for(
//   "2d_bc_thetaPi",
//   CreateRangePolicy<Dim2>({0, m_mblock.i2_max()}, {m_mblock.i1_max() + N_GHOSTS,
//   m_mblock.i2_max() + N_GHOSTS}), Lambda(index_t i, index_t j) {
//     mblock.em0(i, j, em::ex3) = ZERO;

//     mblock.em(i, j, em::ex3) = ZERO;
//     });

// auto j_min {mblock.i2_min()};
// // Kokkos::parallel_for(
// //   "2d_bc_theta0", CreateRangePolicy<Dim1>({mblock.i1_min() - 1}, {mblock.i1_max()}),
// Lambda(index_t i) {
// //     // mblock.em0(i, j_min, em::ex3) = ZERO;
// //     // mblock.em(i, j_min, em::ex3) = ZERO;
// //     mblock.em0(i, j_min - 1, em::ex2) = -mblock.em0(i, j_min, em::ex2);
// //     mblock.em(i, j_min - 1, em::ex2) = -mblock.em(i, j_min, em::ex2);
// //   });

// // // theta = pi boundary
// // auto j_max {mblock.i2_max()};
// // Kokkos::parallel_for(
// //   "2d_bc_thetaPi", CreateRangePolicy<Dim1>({mblock.i1_min() - 1}, {m_mblock.i1_max()}),
// Lambda(index_t i) {
// //     // mblock.em0(i, j_max, em::ex3) = ZERO;
// //     // mblock.em(i, j_max, em::ex3) = ZERO;
// //     mblock.em0(i, j_max, em::ex2) = mblock.em0(i, j_max - 1, em::ex2);
// //     mblock.em(i, j_max, em::ex2) = mblock.em(i, j_max - 1, em::ex2);
// //   });

// auto pGen {this->m_pGen};
// auto br_func {&(this->m_pGen.userTargetField_br_cntrv)};
// Kokkos::parallel_for(
//   "2d_absorbing bc",
//   CreateRangePolicy<Dim2>({mblock.i1_min(), mblock.i2_min()}, {mblock.i1_max() + 1,
//   mblock.i2_max() + 1}), Lambda(index_t i, index_t j) {
//     real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
//     real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};

//     // i
//     vec_t<Dim2> rth_;
//     mblock.metric.x_Code2Sph({i_, j_}, rth_);
//     if (rth_[0] > r_absorb) {
//       real_t delta_r1 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
//       real_t sigma_r1 {absorb_norm * (ONE - math::exp(absorb_coeff * HEAVISIDE(delta_r1) *
//       CUBE(delta_r1)))};
//       // !HACK
//       // real_t br_target = pGen.userTargetField_br_cntrv(mblock, {i_, j_ + HALF});
//       real_t br_target = br_func(mblock, {i_, j_ + HALF});
//       // real_t br_target {ZERO};
//       mblock.em0(i, j, em::bx1) = (ONE - sigma_r1) * mblock.em0(i, j, em::bx1) + sigma_r1
//       * br_target; mblock.em(i, j, em::bx1)  = (ONE - sigma_r1) * mblock.em(i, j, em::bx1)
//       + sigma_r1 * br_target;
//     }
//     // i + 1/2
//     mblock.metric.x_Code2Sph({i_ + HALF, j_}, rth_);
//     if (rth_[0] > r_absorb) {
//       real_t delta_r2 {(rth_[0] - r_absorb) / (r_max - r_absorb)};
//       real_t sigma_r2 {absorb_norm * (ONE - math::exp(absorb_coeff * HEAVISIDE(delta_r2) *
//       CUBE(delta_r2)))};
//       // !HACK
//       // real_t bth_target {pGen->userTargetField_bth_cntrv(mblock, {i_ + HALF, j_})};
//       real_t bth_target {ZERO};
//       mblock.em0(i, j, em::bx2) = (ONE - sigma_r2) * mblock.em0(i, j, em::bx2) + sigma_r2
//       * bth_target; mblock.em(i, j, em::bx2)  = (ONE - sigma_r2) * mblock.em(i, j,
//       em::bx2) + sigma_r2 * bth_target; mblock.em0(i, j, em::bx3) = (ONE - sigma_r2) *
//       mblock.em0(i, j, em::bx3); mblock.em(i, j, em::bx3)  = (ONE - sigma_r2) *
//       mblock.em(i, j, em::bx3);
//     }
//   });
