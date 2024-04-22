#include "wrapper.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

#include METRIC_HEADER

#include "particle_macros.h"

#include "kernels/particle_pusher_gr.hpp"

template <typename T>
void put_value(ntt::array_t<T*>& arr, T value, int i) {
  auto arr_h = Kokkos::create_mirror_view(arr);
  arr_h(i)   = value;
  Kokkos::deep_copy(arr, arr_h);
}

template <typename T>
auto get_value(const ntt::array_t<T*>& arr, int i) -> T {
  auto arr_h = Kokkos::create_mirror_view(arr);
  Kokkos::deep_copy(arr_h, arr);
  return arr_h(i);
}

template <class M>
auto get_physical_coord(const int                      p,
                        const ntt::array_t<int*>&      i1,
                        const ntt::array_t<int*>&      i2,
                        const ntt::array_t<prtldx_t*>& dx1,
                        const ntt::array_t<prtldx_t*>& dx2,
                        const M& metric) -> std::pair<real_t, real_t> {
  std::pair<real_t, real_t> rth;
  ntt::coord_t<ntt::Dim2>   xC { ZERO };
  ntt::coord_t<ntt::Dim2>   rtheta { ZERO };
  xC[0] = i_di_to_Xi(get_value(i1, p), get_value(dx1, p));
  xC[1] = i_di_to_Xi(get_value(i2, p), get_value(dx2, p));
  metric.x_Code2Phys(xC, rtheta);
  rth.first  = rtheta[0];
  rth.second = rtheta[1];
  return rth;
}

auto dummy_metric(const unsigned int nx1, const unsigned int nx2)
  -> ntt::Metric<ntt::Dim2> {
  const auto resolution = std::vector<unsigned int>({ nx1, nx2 });
  const auto extent = std::vector<real_t>({ 1.0, 100.0, ZERO, ntt::constant::PI });
  const auto qsph_r0 = (real_t)(0.0);
  const auto qsph_h  = (real_t)(0.25);

  const auto spin = (real_t)(0.5);
  const auto rh   = ONE + std::sqrt(ONE - SQR(spin));

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
    constexpr auto nx1 = 100, nx2 = 100;
    auto           metric = dummy_metric(nx1, nx2);
    {
      /* -------------------------------------------------------------------------- */
      /*                                   pusher */
      /* -------------------------------------------------------------------------- */
      ntt::ndfield_t<ntt::Dim2, 6> DB { "DB", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
      ntt::ndfield_t<ntt::Dim2, 6> DB0 { "DB",
                                         nx1 + 2 * N_GHOSTS,
                                         nx2 + 2 * N_GHOSTS };
      ntt::array_t<int*>           i1 { "i1", 10 };
      ntt::array_t<int*>           i2 { "i2", 10 };
      ntt::array_t<int*>           i3 { "i3", 10 };
      ntt::array_t<int*>           i1_prev { "i1_prev", 10 };
      ntt::array_t<int*>           i2_prev { "i2_prev", 10 };
      ntt::array_t<int*>           i3_prev { "i3_prev", 10 };
      ntt::array_t<prtldx_t*>      dx1 { "dx1", 10 };
      ntt::array_t<prtldx_t*>      dx2 { "dx2", 10 };
      ntt::array_t<prtldx_t*>      dx3 { "dx3", 10 };
      ntt::array_t<prtldx_t*>      dx1_prev { "dx1_prev", 10 };
      ntt::array_t<prtldx_t*>      dx2_prev { "dx2_prev", 10 };
      ntt::array_t<prtldx_t*>      dx3_prev { "dx3_prev", 10 };
      ntt::array_t<real_t*>        ux1 { "ux1", 10 };
      ntt::array_t<real_t*>        ux2 { "ux2", 10 };
      ntt::array_t<real_t*>        ux3 { "ux3", 10 };
      ntt::array_t<real_t*>        phi { "phi", 10 };
      ntt::array_t<real_t*>        weight { "weight", 10 };
      ntt::array_t<short*>         tag { "tag", 10 };

      int    i1_0, i2_0;
      real_t dx1_0, dx2_0;

      const real_t            r0 = 50.5, th0 = 1.5;
      ntt::coord_t<ntt::Dim2> xC_0 { ZERO };

      const real_t          ux1_0 = 1.5, ux2_0 = 0.5, ux3_0 = -1.2;
      ntt::vec_t<ntt::Dim3> uC_0 { ZERO };

      metric.x_Phys2Code({ r0, th0 }, xC_0);
      from_Xi_to_i_di(xC_0[0], i1_0, dx1_0);
      from_Xi_to_i_di(xC_0[1], i2_0, dx2_0);

      metric.v3_Hat2Cov(xC_0, { ux1_0, ux2_0, ux3_0 }, uC_0);

      put_value<int>(i1, i1_0, 0);
      put_value<int>(i2, i2_0, 0);
      put_value<prtldx_t>(dx1, dx1_0, 0);
      put_value<prtldx_t>(dx2, dx2_0, 0);
      put_value<prtldx_t>(ux1, uC_0[0], 0);
      put_value<prtldx_t>(ux2, uC_0[1], 0);
      put_value<prtldx_t>(ux3, uC_0[2], 0);
      put_value<short>(tag, ntt::ParticleTag::alive, 0);

      std::vector<std::vector<ntt::BoundaryCondition>> boundaries;
      boundaries.push_back(
        std::vector<ntt::BoundaryCondition>(2, ntt::BoundaryCondition::PERIODIC));
      boundaries.push_back(
        std::vector<ntt::BoundaryCondition>(2, ntt::BoundaryCondition::PERIODIC));

      auto kernel = ntt::Pusher_kernel<ntt::Dim2, ntt::Metric<ntt::Dim2>>(
        DB,
        DB0,
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
        tag,
        metric,
        ONE,
        ONE,
        nx1,
        nx2,
        1,
        static_cast<real_t>(1.0e-5),
        10,
        boundaries);
      Kokkos::parallel_for("ParticlesPush",
                           Kokkos::RangePolicy<AccelExeSpace, ntt::Massive_t>(0, 1),
                           kernel);
      auto [ra, tha]   = get_physical_coord(0, i1, i2, dx1, dx2, metric);
      const real_t pha = get_value(phi, 0);

      if (metric.rg() != ZERO) {
        // for KS with M != 0
        if (!ntt::AlmostEqual(ra,
                              static_cast<real_t>(51.115658),
                              static_cast<real_t>(1e-4))) {
          throw std::runtime_error("r coordinate is not correct");
        }
        if (!ntt::AlmostEqual(tha,
                              static_cast<real_t>(1.504318),
                              static_cast<real_t>(1e-4))) {
          throw std::runtime_error("th coordinate is not correct");
        }
        if (!ntt::AlmostEqual(pha,
                              static_cast<real_t>(6.272962),
                              static_cast<real_t>(1e-4))) {
          throw std::runtime_error("phi coordinate is not correct");
        }
      } else {
        // for KS with M == 0
        if (!ntt::AlmostEqual(ra,
                              static_cast<real_t>(51.180923),
                              static_cast<real_t>(1e-4))) {
          throw std::runtime_error("r coordinate is not correct");
        }
        if (!ntt::AlmostEqual(tha,
                              static_cast<real_t>(1.504381),
                              static_cast<real_t>(1e-4))) {
          throw std::runtime_error("th coordinate is not correct");
        }
        if (!ntt::AlmostEqual(pha,
                              static_cast<real_t>(6.272648),
                              static_cast<real_t>(1e-4))) {
          throw std::runtime_error("phi coordinate is not correct");
        }
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