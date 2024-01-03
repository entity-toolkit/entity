#include "wrapper.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

#include METRIC_HEADER

#if defined(PIC_ENGINE)
  #define SIMENGINE ntt::SimulationEngine::PIC
#else
  #define SIMENGINE ntt::SimulationEngine::GRPIC
#endif

#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"

template <typename T>
void put_value(ntt::array_t<T*> arr, T value, int i) {
  auto arr_h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(arr_h, arr);
  arr_h(i) = value;
  Kokkos::deep_copy(arr, arr_h);
}

auto dummy_metric(const unsigned int nx1, const unsigned int nx2)
  -> ntt::Metric<ntt::Dim2> {
  const auto resolution = std::vector<unsigned int>({ nx1, nx2 });
#ifdef MINKOWSKI_METRIC
  const auto extent = std::vector<real_t>({ 0.0, 55.0, 0.0, 55.0 });
#else
  const auto extent = std::vector<real_t>({ 1.0, 100.0, ZERO, ntt::constant::PI });
#endif
  // optional for GR
  const auto spin    = (real_t)(0.9);
  const auto rh      = ONE + std::sqrt(ONE - SQR(spin));
  // optional for Qspherical
  const auto qsph_r0 = (real_t)(0.0);
  const auto qsph_h  = (real_t)(0.25);

  auto params = new real_t[6];
  params[0]   = qsph_r0;
  params[1]   = qsph_h;
  params[4]   = spin;
  params[5]   = rh;
  ntt::Metric<ntt::Dim2> metric(resolution, extent, params);
  delete[] params;
  return metric;
}

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    constexpr auto nx1 = 10, nx2 = 10;
    auto           metric = dummy_metric(nx1, nx2);
    {
      /* --------------------------------- deposit -------------------------------- */
      ntt::ndfield_t<ntt::Dim2, 3> J { "J", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
      ntt::array_t<int*>      i1 { "i1", 10 };
      ntt::array_t<int*>      i2 { "i2", 10 };
      ntt::array_t<int*>      i3 { "i3", 10 };
      ntt::array_t<int*>      i1_prev { "i1_prev", 10 };
      ntt::array_t<int*>      i2_prev { "i2_prev", 10 };
      ntt::array_t<int*>      i3_prev { "i3_prev", 10 };
      ntt::array_t<prtldx_t*> dx1 { "dx1", 10 };
      ntt::array_t<prtldx_t*> dx2 { "dx2", 10 };
      ntt::array_t<prtldx_t*> dx3 { "dx3", 10 };
      ntt::array_t<prtldx_t*> dx1_prev { "dx1_prev", 10 };
      ntt::array_t<prtldx_t*> dx2_prev { "dx2_prev", 10 };
      ntt::array_t<prtldx_t*> dx3_prev { "dx3_prev", 10 };
      ntt::array_t<real_t*>   ux1 { "ux1", 10 };
      ntt::array_t<real_t*>   ux2 { "ux2", 10 };
      ntt::array_t<real_t*>   ux3 { "ux3", 10 };
      ntt::array_t<real_t*>   phi { "phi", 10 };
      ntt::array_t<real_t*>   weight { "weight", 10 };
      ntt::array_t<short*>    tag { "tag", 10 };
      const real_t            charge { 1.0 }, inv_dt { 1.0 };

      auto J_scat = Kokkos::Experimental::create_scatter_view(J);

      const real_t xi = 0.53, xf = 0.47;
      const real_t yi = 0.34, yf = 0.52;

      const real_t xr = 0.5 * (xi + xf);
      const real_t yr = 0.5 * (yi + yf);

      const real_t Wx1 = 0.5 * (xi + xr);
      const real_t Wx2 = 0.5 * (xf + xr);

      const real_t Wy1 = 0.5 * (yi + yr);
      const real_t Wy2 = 0.5 * (yf + yr);

      const real_t Fx1 = (xr - xi);
      const real_t Fx2 = (xf - xr);

      const real_t Fy1 = (yr - yi);
      const real_t Fy2 = (yf - yr);

      const real_t Jx1 = Fx1 * (1 - Wy1) + Fx2 * (1 - Wy2);
      const real_t Jx2 = Fx1 * Wy1 + Fx2 * Wy2;

      const real_t Jy1 = Fy1 * (1 - Wx1) + Fy2 * (1 - Wx2);
      const real_t Jy2 = Fy1 * Wx1 + Fy2 * Wx2;

      const int i0 = 4, j0 = 4;

      put_value<int>(i1, i0, 0);
      put_value<int>(i2, j0, 0);
      put_value<int>(i1_prev, i0, 0);
      put_value<int>(i2_prev, j0, 0);
      put_value<prtldx_t>(dx1, xf, 0);
      put_value<prtldx_t>(dx2, yf, 0);
      put_value<prtldx_t>(dx1_prev, xi, 0);
      put_value<prtldx_t>(dx2_prev, yi, 0);
      put_value<real_t>(weight, 1.0, 0);
      put_value<short>(tag, ntt::ParticleTag::alive, 0);

      Kokkos::parallel_for(
        "CurrentsDeposit",
        10,
        ntt::DepositCurrents_kernel<ntt::Dim2, SIMENGINE, ntt::Metric<ntt::Dim2>>(
          J_scat,
          i1,
          i2,
          i3,
          i1_prev,
          i2_prev,
          i3_prev,
          dx1,
          dx2,
          dx3,
          dx1_prev,
          dx2_prev,
          dx3_prev,
          ux1,
          ux2,
          ux3,
          phi,
          weight,
          tag,
          metric,
          charge,
          inv_dt));

      Kokkos::Experimental::contribute(J, J_scat);

      real_t SumDivJ { 0.0 };
      Kokkos::parallel_reduce(
        "SumDivJ",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ N_GHOSTS, N_GHOSTS },
                                               { nx1 + N_GHOSTS, nx2 + N_GHOSTS }),
        Lambda(const int i, const int j, real_t& sum) {
          sum += J(i, j, ntt::cur::jx1) - J(i - 1, j, ntt::cur::jx1) +
                 J(i, j, ntt::cur::jx2) - J(i, j - 1, ntt::cur::jx2);
        },
        SumDivJ);

      auto J_h = Kokkos::create_mirror_view(J);
      Kokkos::deep_copy(J_h, J);

      if (!ntt::AlmostEqual(ZERO, SumDivJ)) {
        throw std::logic_error("DepositCurrents_kernel::SumDivJ != 0");
      }
      if (!ntt::AlmostEqual(Jx1, J_h(i0 + N_GHOSTS, j0 + N_GHOSTS, ntt::cur::jx1))) {
        throw std::logic_error("DepositCurrents_kernel::Jx1 is incorrect");
      }
      if (!ntt::AlmostEqual(Jx2,
                            J_h(i0 + N_GHOSTS, j0 + 1 + N_GHOSTS, ntt::cur::jx1))) {
        throw std::logic_error("DepositCurrents_kernel::Jx2 is incorrect");
      }
      if (!ntt::AlmostEqual(Jy1, J_h(i0 + N_GHOSTS, j0 + N_GHOSTS, ntt::cur::jx2))) {
        throw std::logic_error("DepositCurrents_kernel::Jy1 is incorrect");
      }
      if (!ntt::AlmostEqual(Jy2,
                            J_h(i0 + 1 + N_GHOSTS, j0 + N_GHOSTS, ntt::cur::jx2))) {
        throw std::logic_error("DepositCurrents_kernel::Jy2 is incorrect");
      }
    }
    {
      /* --------------------------------- filter --------------------------------- */
      ntt::ndfield_t<ntt::Dim2, 3> J { "J", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS },
        Jbuff { "Jbuff", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };

      ntt::tuple_t<std::size_t, ntt::Dim2> size;
      size[0]                  = nx1;
      size[1]                  = nx2;
      auto J_h                 = Kokkos::create_mirror_view(J);
      J_h(5, 5, ntt::cur::jx1) = 1.0;
      J_h(4, 5, ntt::cur::jx2) = 1.0;
      J_h(5, 4, ntt::cur::jx3) = 1.0;
      Kokkos::deep_copy(J, J_h);
      const auto range = ntt::CreateRangePolicy<ntt::Dim2>(
        { N_GHOSTS, N_GHOSTS },
        { nx1 + N_GHOSTS, nx2 + N_GHOSTS + 1 });
      Kokkos::deep_copy(Jbuff, J);
      Kokkos::parallel_for("CurrentsFilter",
                           range,
                           ntt::DigitalFilter_kernel<ntt::Dim2>(J, Jbuff, size));
      real_t SumJx1 { 0.0 }, SumJx2 { 0.0 }, SumJx3 { 0.0 };
      Kokkos::parallel_reduce(
        "SumJx1",
        range,
        Lambda(const int i, const int j, real_t& sum) {
          sum += J(i, j, ntt::cur::jx1);
        },
        SumJx1);
      Kokkos::parallel_reduce(
        "SumJx2",
        range,
        Lambda(const int i, const int j, real_t& sum) {
          sum += J(i, j, ntt::cur::jx2);
        },
        SumJx2);
      Kokkos::parallel_reduce(
        "SumJx3",
        range,
        Lambda(const int i, const int j, real_t& sum) {
          sum += J(i, j, ntt::cur::jx3);
        },
        SumJx3);
      if (!ntt::AlmostEqual(ONE, SumJx1)) {
        throw std::logic_error("DigitalFilter_kernel::SumJx1 != 1");
      }
      if (!ntt::AlmostEqual(ONE, SumJx2)) {
        throw std::logic_error("DigitalFilter_kernel::SumJx2 != 1");
      }
      if (!ntt::AlmostEqual(ONE, SumJx3)) {
        throw std::logic_error("DigitalFilter_kernel::SumJx3 != 1");
      }
    }
  }

  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }

  ntt::GlobalFinalize();

  return 0;
}