#include "wrapper.h"

#include METRIC_HEADER
#include "utils/qmath.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    const auto resolution = std::vector<unsigned int>({ 1256, 2048 });
    const auto extent     = std::vector<real_t>({ 1.0, 5000.0 });
    // optional for GR
    const auto spin       = (real_t)(0.95);
    const auto rh         = ONE + std::sqrt(ONE - SQR(spin));
    // optional for Qspherical
    const auto qsph_r0    = (real_t)(-10.0);
    const auto qsph_h     = (real_t)(0.1);

    auto       params     = new real_t[6];
    params[0]             = qsph_r0;
    params[1]             = qsph_h;
    params[4]             = spin;
    params[5]             = rh;
    ntt::Metric<ntt::Dim2> metric(resolution, extent, params);
    delete[] params;

    {
      /* ------------ Test conversion code <-> spherical <-> cartesian ------------ */
      std::vector<real_t> x1 { HALF, resolution[0] - HALF }, x2 { HALF, resolution[1] - HALF };

      auto                correct = true;
      const auto          tiny    = static_cast<real_t>(1e-4);

      for (auto i { 0 }; i < 2; ++i) {
        for (auto j { 0 }; j < 2; ++j) {
          ntt::coord_t<ntt::Dim2> xi { x1[i], x2[j] };
          ntt::coord_t<ntt::Dim2> xi_from_sph { ZERO }, xi_from_cart { ZERO };
          ntt::coord_t<ntt::Dim2> xsph_from_code { ZERO }, xcart_from_code { ZERO };

          metric.x_Code2Sph(xi, xsph_from_code);
          metric.x_Code2Cart(xi, xcart_from_code);
          metric.x_Sph2Code(xsph_from_code, xi_from_sph);
          metric.x_Cart2Code(xcart_from_code, xi_from_cart);

          for (auto d { 0 }; d < 2; ++d) {
            correct = correct
                      && (ntt::AlmostEqual(xi[d], xi_from_sph[d], tiny)
                          && ntt::AlmostEqual(xi[d], xi_from_cart[d], tiny));
          }
        }
      }
      !correct
        ? throw std::logic_error("Code2Cart -> Cart2Code or Code2Sph -> Sph2Code not correct")
        : (void)0;
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();

  return 0;
}