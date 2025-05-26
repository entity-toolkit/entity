#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "kernels/currents_deposit.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

Inline auto equal(real_t a, real_t b, const char* msg = "", real_t acc = ONE)
  -> bool {
  const auto eps = epsilon * acc;
  if (not cmp::AlmostEqual(a, b, eps)) {
    printf("%.12e != %.12e %s\n", a, b, msg);
    return false;
  }
  return true;
}

template <typename T>
void put_value(array_t<T*> arr, T value, int i) {
  auto arr_h = Kokkos::create_mirror_view(arr);
  arr_h(i)   = value;
  Kokkos::deep_copy(arr, arr_h);
}

template <typename M, ntt::SimEngine::type S>
void testDeposit(const std::vector<std::size_t>&      res,
                 const boundaries_t<real_t>&          ext,
                 const std::map<std::string, real_t>& params = {},
                 const real_t                         acc    = ONE) {
  static_assert(M::Dim == 2);
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");
  using namespace ntt;

  M metric { res, ext, params };

  const auto nx1 = res[0];
  const auto nx2 = res[1];

  ndfield_t<M::Dim, 3> J { "J", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
  array_t<int*>        i1 { "i1", 10 };
  array_t<int*>        i2 { "i2", 10 };
  array_t<int*>        i3 { "i3", 10 };
  array_t<int*>        i1_prev { "i1_prev", 10 };
  array_t<int*>        i2_prev { "i2_prev", 10 };
  array_t<int*>        i3_prev { "i3_prev", 10 };
  array_t<prtldx_t*>   dx1 { "dx1", 10 };
  array_t<prtldx_t*>   dx2 { "dx2", 10 };
  array_t<prtldx_t*>   dx3 { "dx3", 10 };
  array_t<prtldx_t*>   dx1_prev { "dx1_prev", 10 };
  array_t<prtldx_t*>   dx2_prev { "dx2_prev", 10 };
  array_t<prtldx_t*>   dx3_prev { "dx3_prev", 10 };
  array_t<real_t*>     ux1 { "ux1", 10 };
  array_t<real_t*>     ux2 { "ux2", 10 };
  array_t<real_t*>     ux3 { "ux3", 10 };
  array_t<real_t*>     phi { "phi", 10 };
  array_t<real_t*>     weight { "weight", 10 };
  array_t<short*>      tag { "tag", 10 };
  const real_t         charge { 1.0 }, inv_dt { 1.0 };

  const int i0 = 3, j0 = 3;
  const int i0f = 3, j0f = 3;
  const real_t uz = 0.5;

  //   const prtldx_t dxi = 0.53, dxf = 0.47;
  //   const prtldx_t dyi = 0.34, dyf = 0.52;
  const prtldx_t dxi = 0.65, dxf = 0.65;
  const prtldx_t dyi = 0.65, dyf = 0.65;
  const real_t   xi = (real_t)i0 + (real_t)dxi, xf = (real_t)i0f + (real_t)dxf;
  const real_t   yi = (real_t)j0 + (real_t)dyi, yf = (real_t)j0f + (real_t)dyf;

  const real_t xr = 0.5 * (xi + xf);
  const real_t yr = 0.5 * (yi + yf);

  const real_t Wx1 = 0.5 * (xi + xr) - (real_t)i0;
  const real_t Wx2 = 0.5 * (xf + xr) - (real_t)i0;

  const real_t Wy1 = 0.5 * (yi + yr) - (real_t)j0;
  const real_t Wy2 = 0.5 * (yf + yr) - (real_t)j0;

  const real_t Fx1 = (xr - xi);
  const real_t Fx2 = (xf - xr);

  const real_t Fy1 = (yr - yi);
  const real_t Fy2 = (yf - yr);

  const real_t Fz1 = HALF * uz / math::sqrt(1.0 + uz * uz);
  const real_t Fz2 = HALF * uz / math::sqrt(1.0 + uz * uz);

  const real_t Jx1 = Fx1 * (1 - Wy1) + Fx2 * (1 - Wy2);
  const real_t Jx2 = Fx1 * Wy1 + Fx2 * Wy2;

  const real_t Jy1 = Fy1 * (1 - Wx1) + Fy2 * (1 - Wx2);
  const real_t Jy2 = Fy1 * Wx1 + Fy2 * Wx2;

  const real_t Jz = Fz1 * (1 - Wx1) + Fz2 * (1 - Wy1) +
                    Fz1 * Wx1 * (1 - Wy1) +
                    Fz1 * (1 - Wx1) * Wy1 +
                    Fz1 * Wx1 * Wy1 +
                    Fz2 * (1 - Wx2) * (1 - Wy2) +
                    Fz2 * Wx2 * (1 - Wy2) +
                    Fz2 * (1 - Wx2) * Wy2 +
                    Fz2 * Wx2 * Wy2;

  put_value<int>(i1, i0f, 0);
  put_value<int>(i2, j0f, 0);
  put_value<int>(i1_prev, i0, 0);
  put_value<int>(i2_prev, j0, 0);
  put_value<prtldx_t>(dx1, dxf, 0);
  put_value<prtldx_t>(dx2, dyf, 0);
  put_value<prtldx_t>(dx1_prev, dxi, 0);
  put_value<prtldx_t>(dx2_prev, dyi, 0);
  put_value<prtldx_t>(ux3, uz, 0);
  put_value<real_t>(weight, 1.0, 0);
  put_value<short>(tag, ParticleTag::alive, 0);

  auto J_scat = Kokkos::Experimental::create_scatter_view(J);

  // clang-format off
  Kokkos::parallel_for("CurrentsDeposit", 10,
                       kernel::DepositCurrents_kernel<S, M, 1u>(J_scat,
                                                            i1, i2, i3,
                                                            i1_prev, i2_prev, i3_prev,
                                                            dx1, dx2, dx3,
                                                            dx1_prev, dx2_prev, dx3_prev,
                                                            ux1, ux2, ux3,
                                                            phi, weight, tag,
                                                            metric, charge, inv_dt));
  // clang-format on

  Kokkos::Experimental::contribute(J, J_scat);

  const auto range = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
    { N_GHOSTS, N_GHOSTS },
    { nx1 + N_GHOSTS, nx2 + N_GHOSTS });

  real_t SumDivJ = ZERO, SumJx = ZERO, SumJy = ZERO, SumJz = ZERO;
  Kokkos::parallel_reduce(
    "SumDivJ",
    range,
    Lambda(const int i, const int j, real_t& sum) {
      sum += J(i, j, cur::jx1) - J(i - 1, j, cur::jx1) + J(i, j, cur::jx2) -
             J(i, j - 1, cur::jx2);
    },
    SumDivJ);

  Kokkos::parallel_reduce(
    "SumJx",
    range,
    Lambda(const int i, const int j, real_t& sum) { sum += J(i, j, cur::jx1); },
    SumJx);

  Kokkos::parallel_reduce(
    "SumJy",
    range,
    Lambda(const int i, const int j, real_t& sum) { sum += J(i, j, cur::jx2); },
    SumJy);

  Kokkos::parallel_reduce(
    "SumJy",
    range,
    Lambda(const int i, const int j, real_t& sum) { sum += J(i, j, cur::jx3); },
    SumJz);

  auto J_h = Kokkos::create_mirror_view(J);
  Kokkos::deep_copy(J_h, J);

  if (not cmp::AlmostZero(SumDivJ)) {
    throw std::logic_error("DepositCurrents_kernel::SumDivJ != 0");
  }

  std::cout << "SumJx: " << SumJx << " expected " << Jx1 + Jx2 << std::endl;
  std::cout << "SumJy: " << SumJy << " expected " << Jy1 + Jy2 << std::endl;
  std::cout << "SumJz: " << SumJz << " expected " << Jz << std::endl;
  // errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + N_GHOSTS, cur::jx1), Jx1, "", acc),
  //         "DepositCurrents_kernel::Jx1 is incorrect");
  // errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + 1 + N_GHOSTS, cur::jx1), Jx2, "", acc),
  //         "DepositCurrents_kernel::Jx2 is incorrect");
  // errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + N_GHOSTS, cur::jx2), Jy1, "", acc),
  //         "DepositCurrents_kernel::Jy1 is incorrect");
  // errorIf(not equal(J_h(i0 + 1 + N_GHOSTS, j0 + N_GHOSTS, cur::jx2), Jy2, "", acc),
  //         "DepositCurrents_kernel::Jy2 is incorrect");
}

// void ind_pond(real_t Rcoord, int* Iindices, real_t* Rpond) {

//   // Assuming interp_order is an integer and Rcoord is a double
//   int i_min = std::floor(Rcoord - HALF);

//   // Populate Iindices
//   for (int i = 0; i < 3; ++i) {
//     Iindices[i] = i_min + i;
//   }

//   // Eq. 24
//   Rpond[0] = 0.5 * std::pow(0.5 + (static_cast<double>(Iindices[1]) - Rcoord), 2);
//   Rpond[1] = 0.75 - std::pow(static_cast<double>(Iindices[1]) - Rcoord, 2);
//   Rpond[2] = 0.5 * std::pow(0.5 - (static_cast<double>(Iindices[1]) - Rcoord), 2);
// }

// template <typename M, ntt::SimEngine::type S>
// void testDeposit_2nd(const std::vector<std::size_t>&      res,
//                      const boundaries_t<real_t>&          ext,
//                      const std::map<std::string, real_t>& params = {},
//                      const real_t                         acc    = ONE) {
//   static_assert(M::Dim == 2);
//   errorIf(res.size() != M::Dim, "res.size() != M::Dim");
//   using namespace ntt;

//   M metric { res, ext, params };

//   const auto nx1 = res[0];
//   const auto nx2 = res[1];

//   ndfield_t<M::Dim, 3> J { "J", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
//   array_t<int*>        i1 { "i1", 10 };
//   array_t<int*>        i2 { "i2", 10 };
//   array_t<int*>        i3 { "i3", 10 };
//   array_t<int*>        i1_prev { "i1_prev", 10 };
//   array_t<int*>        i2_prev { "i2_prev", 10 };
//   array_t<int*>        i3_prev { "i3_prev", 10 };
//   array_t<prtldx_t*>   dx1 { "dx1", 10 };
//   array_t<prtldx_t*>   dx2 { "dx2", 10 };
//   array_t<prtldx_t*>   dx3 { "dx3", 10 };
//   array_t<prtldx_t*>   dx1_prev { "dx1_prev", 10 };
//   array_t<prtldx_t*>   dx2_prev { "dx2_prev", 10 };
//   array_t<prtldx_t*>   dx3_prev { "dx3_prev", 10 };
//   array_t<real_t*>     ux1 { "ux1", 10 };
//   array_t<real_t*>     ux2 { "ux2", 10 };
//   array_t<real_t*>     ux3 { "ux3", 10 };
//   array_t<real_t*>     phi { "phi", 10 };
//   array_t<real_t*>     weight { "weight", 10 };
//   array_t<short*>      tag { "tag", 10 };
//   const real_t         charge { 1.0 }, inv_dt { 1.0 };

//   const int i0 = 4, j0 = 4;

//   // initial and final positions
//   const prtldx_t dxi = 0.53, dxf = 0.47;
//   const prtldx_t dyi = 0.34, dyf = 0.52;
//   const real_t   xi = (real_t)i0 + (real_t)dxi, xf = (real_t)i0 + (real_t)dxf;
//   const real_t   yi = (real_t)j0 + (real_t)dyi, yf = (real_t)j0 + (real_t)dyf;

//   // const real_t xr = 0.5 * (xi + xf);
//   // const real_t yr = 0.5 * (yi + yf);

//   // const real_t Wx1 = 0.5 * (xi + xr) - (real_t)i0;
//   // const real_t Wx2 = 0.5 * (xf + xr) - (real_t)i0;

//   // const real_t Wy1 = 0.5 * (yi + yr) - (real_t)j0;
//   // const real_t Wy2 = 0.5 * (yf + yr) - (real_t)j0;

//   // const real_t Fx1 = (xr - xi);
//   // const real_t Fx2 = (xf - xr);

//   // const real_t Fy1 = (yr - yi);
//   // const real_t Fy2 = (yf - yr);

//   // const real_t Jx1 = Fx1 * (1 - Wy1) + Fx2 * (1 - Wy2);
//   // const real_t Jx2 = Fx1 * Wy1 + Fx2 * Wy2;

//   // const real_t Jy1 = Fy1 * (1 - Wx1) + Fy2 * (1 - Wx2);
//   // const real_t Jy2 = Fy1 * Wx1 + Fy2 * Wx2;

//   // Define interp_order
//   constexpr int interp_order = 2;
//   const real_t aux_jx = 1.0;
//   const real_t aux_jy = 1.0;
//   const real_t aux_jz = 1.0;

//   // Arrays with size (interp_order + 1)
//   std::array<int, interp_order + 1>    ISx1, ISx2;
//   std::array<double, interp_order + 1> PondSx1, PondSx2;
//   std::array<int, interp_order + 1>    ISy1, ISy2;
//   std::array<double, interp_order + 1> PondSy1, PondSy2;

//   // 2D arrays with size (interp_order + 2) x (interp_order + 2)
//   std::array<std::array<double, interp_order + 2>, interp_order + 2> WEsirkx,
//     WEsirky, WEsirkz;
//   std::array<std::array<double, interp_order + 2>, interp_order + 2> jx_local,
//     jy_local;

//   std::array<std::array<double, 10>, 10> jx, jy, jz;
//   std::fill(jx.begin(), jx.end(), 0.0);
//   std::fill(jy.begin(), jy.end(), 0.0);
//   std::fill(jz.begin(), jz.end(), 0.0);
//   // 1D arrays with size (interp_order + 2)
//   std::array<double, interp_order + 2> Sx2, Sx1, Sy2, Sy1;

//   // Interpolation coefficients
//   ind_pond(xi, &ISx1, &PondSx1);
//   ind_pond(xf, &ISx2, &PondSx2);
//   ind_pond(yi, &ISy1, &PondSy1);
//   ind_pond(yf, &ISy2, &PondSy2);

//   int min_x, max_x;
//   int min_y, max_y;

//   // Esirkepov coefficients W
//   int shift_Ix = ISx2[0] - ISx1[0];
//   std::fill(Sx2.begin(), Sx2.end(), 0.0);
//   std::fill(Sx1.begin(), Sx1.end(), 0.0);

//   if (shift_Ix == 0) {
//     std::copy(PondSx2.begin(), PondSx2.end(), Sx2.begin());
//     std::copy(PondSx1.begin(), PondSx1.end(), Sx1.begin());
//     min_x = ISx2[0];
//     max_x = ISx2[interp_order];
//   } else if (shift_Ix == 1) {
//     std::copy(PondSx2.begin(), PondSx2.end(), Sx2.begin() + 1);
//     std::copy(PondSx1.begin(), PondSx1.end(), Sx1.begin());
//     min_x = ISx1[0];
//     max_x = ISx2[interp_order];
//   } else if (shift_Ix == -1) {
//     std::copy(PondSx2.begin(), PondSx2.end(), Sx2.begin());
//     std::copy(PondSx1.begin(), PondSx1.end(), Sx1.begin() + 1);
//     min_x = ISx2[0];
//     max_x = ISx1[interp_order];
//   }

//   int shift_Iy = ISy2[0] - ISy1[0];
//   std::fill(Sy2.begin(), Sy2.end(), 0.0);
//   std::fill(Sy1.begin(), Sy1.end(), 0.0);

//   if (shift_Iy == 0) {
//     std::copy(PondSy2.begin(), PondSy2.end(), Sy2.begin());
//     std::copy(PondSy1.begin(), PondSy1.end(), Sy1.begin());
//     min_y = ISy2[0];
//     max_y = ISy2[interp_order];
//   } else if (shift_Iy == 1) {
//     std::copy(PondSy2.begin(), PondSy2.end(), Sy2.begin() + 1);
//     std::copy(PondSy1.begin(), PondSy1.end(), Sy1.begin());
//     min_y = ISy1[0];
//     max_y = ISy2[interp_order];
//   } else if (shift_Iy == -1) {
//     std::copy(PondSy2.begin(), PondSy2.end(), Sy2.begin());
//     std::copy(PondSy1.begin(), PondSy1.end(), Sy1.begin() + 1);
//     min_y = ISy2[0];
//     max_y = ISy1[interp_order];
//   }

//   for (int i = 0; i < interp_order + 2; ++i) {
//     for (int j = 0; j < interp_order + 2; ++j) {
//       WEsirkx[i][j] = 0.5 * (Sx2[i] - Sx1[i]) * (Sy2[j] + Sy1[j]);
//       WEsirky[i][j] = 0.5 * (Sx2[i] + Sx1[i]) * (Sy2[j] - Sy1[j]);
//       WEsirkz[i][j] = THIRD * (Sy2[j] * (0.5 * Sx1[i] + Sx2[i]) +
//                                Sy1[j] * (0.5 * Sx2[i] + Sx1[i]));
//     }
//   }

//   // Current deposition jx
//   for (int j = 0; j < interp_order + 2; ++j) {
//     jx_local[0][j] = -aux_jx * WEsirkx[0][j];
//   }
//   for (int i = 1; i < interp_order + 2; ++i) {
//     for (int j = 0; j < interp_order + 2; ++j) {
//       jx_local[i][j] = jx_local[i - 1][j] - aux_jx * WEsirkx[i][j];
//     }
//   }
//   for (int i = 0; i < max_x - min_x; ++i) {
//     for (int j = 0; j < max_y - min_y + 1; ++j) {
//       jx[min_x + i][min_y + j] += jx_local[i][j];
//     }
//   }

//   // Current deposition jy
//   for (int i = 0; i < interp_order + 2; ++i) {
//     jy_local[i][0] = -aux_jy * WEsirky[i][0];
//   }
//   for (int j = 1; j < interp_order + 2; ++j) {
//     for (int i = 0; i < interp_order + 2; ++i) {
//       jy_local[i][j] = jy_local[i][j - 1] - aux_jy * WEsirky[i][j];
//     }
//   }
//   for (int i = 0; i < max_x - min_x + 1; ++i) {
//     for (int j = 0; j < max_y - min_y; ++j) {
//       jy[min_x + i][min_y + j] += jy_local[i][j];
//     }
//   }

//   // Current deposition jz
//   for (int i = 0; i < max_x - min_x + 1; ++i) {
//     for (int j = 0; j < max_y - min_y + 1; ++j) {
//       jz[min_x + i][min_y + j] += aux_jz * WEsirkz[i][j];
//     }
//   }

//   // define particle positions
//   put_value<int>(i1, i0, 0);
//   put_value<int>(i2, j0, 0);
//   put_value<int>(i1_prev, i0, 0);
//   put_value<int>(i2_prev, j0, 0);
//   put_value<prtldx_t>(dx1, dxf, 0);
//   put_value<prtldx_t>(dx2, dyf, 0);
//   put_value<prtldx_t>(dx1_prev, dxi, 0);
//   put_value<prtldx_t>(dx2_prev, dyi, 0);
//   put_value<real_t>(weight, 1.0, 0);
//   put_value<short>(tag, ParticleTag::alive, 0);

//   auto J_scat = Kokkos::Experimental::create_scatter_view(J);

//   // clang-format off
//   Kokkos::parallel_for("CurrentsDeposit", 10,
//                        kernel::DepositCurrents_kernel<S, M, 2u>(J_scat,
//                                                             i1, i2, i3,
//                                                             i1_prev, i2_prev, i3_prev,
//                                                             dx1, dx2, dx3,
//                                                             dx1_prev, dx2_prev, dx3_prev,
//                                                             ux1, ux2, ux3,
//                                                             phi, weight, tag,
//                                                             metric, charge, inv_dt));
//   // clang-format on

//   Kokkos::Experimental::contribute(J, J_scat);

//   const auto range = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
//     { N_GHOSTS, N_GHOSTS },
//     { nx1 + N_GHOSTS, nx2 + N_GHOSTS });

//   real_t SumDivJ = ZERO, SumJx = ZERO, SumJy = ZERO;
//   Kokkos::parallel_reduce(
//     "SumDivJ",
//     range,
//     Lambda(const int i, const int j, real_t& sum) {
//       sum += J(i, j, cur::jx1) - J(i - 1, j, cur::jx1) + J(i, j, cur::jx2) -
//              J(i, j - 1, cur::jx2);
//     },
//     SumDivJ);

//   Kokkos::parallel_reduce(
//     "SumJx",
//     range,
//     Lambda(const int i, const int j, real_t& sum) { sum += J(i, j, cur::jx1); },
//     SumJx);

//   Kokkos::parallel_reduce(
//     "SumJy",
//     range,
//     Lambda(const int i, const int j, real_t& sum) { sum += J(i, j, cur::jx2); },
//     SumJy);

//   auto J_h = Kokkos::create_mirror_view(J);
//   Kokkos::deep_copy(J_h, J);

//   if (not cmp::AlmostZero(SumDivJ)) {
//     throw std::logic_error("DepositCurrents_kernel::SumDivJ != 0");
//   }

//   // std::cout << "SumJx: " << SumJx << " expected " << Jx1 + Jx2 << std::endl;
//   // std::cout << "SumJy: " << SumJy << " expected " << Jy1 + Jy2 << std::endl;
//   errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + N_GHOSTS, cur::jx1), jx[i0][j0], "", acc),
//           "DepositCurrents_kernel::Jx1 is incorrect");
//   errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + 1 + N_GHOSTS, cur::jx1), jx[i0][j0+1], "", acc),
//           "DepositCurrents_kernel::Jx2 is incorrect");
//   errorIf(not equal(J_h(i0 + N_GHOSTS, j0 + N_GHOSTS, cur::jx2), jy[i0][j0], "", acc),
//           "DepositCurrents_kernel::Jy1 is incorrect");
//   errorIf(not equal(J_h(i0 + 1 + N_GHOSTS, j0 + N_GHOSTS, cur::jx2), jy[i0][j0+1], "", acc),
//           "DepositCurrents_kernel::Jy2 is incorrect");
// }

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    using namespace metric;

    testDeposit<Minkowski<Dim::_2D>, SimEngine::SRPIC>(
      {
        10,
        10
    },
      { { 0.0, 55.0 }, { 0.0, 55.0 } },
      {},
      500);

    // testDeposit<Spherical<Dim::_2D>, SimEngine::SRPIC>(
    //   {
    //     10,
    //     10
    // },
    //   { { 1.0, 100.0 } },
    //   {},
    //   500);

    // testDeposit<QSpherical<Dim::_2D>, SimEngine::SRPIC>(
    //   {
    //     10,
    //     10
    // },
    //   { { 1.0, 100.0 } },
    //   { { "r0", 0.0 }, { "h", 0.25 } },
    //   500);

    // testDeposit<KerrSchild<Dim::_2D>, SimEngine::GRPIC>(
    //   {
    //     10,
    //     10
    // },
    //   { { 1.0, 100.0 } },
    //   { { "a", 0.9 } },
    //   500);

    // testDeposit<QKerrSchild<Dim::_2D>, SimEngine::GRPIC>(
    //   {
    //     10,
    //     10
    // },
    //   { { 1.0, 100.0 } },
    //   { { "r0", 0.0 }, { "h", 0.25 }, { "a", 0.9 } },
    //   500);

    // testDeposit<KerrSchild0<Dim::_2D>, SimEngine::GRPIC>(
    //   {
    //     10,
    //     10
    // },
    //   { { 1.0, 100.0 } },
    //   { { "a", 0.9 } },
    //   500);

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
