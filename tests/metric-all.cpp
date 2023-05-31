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
    const auto resolution = std::vector<unsigned int>({ 2560, 1920 });
    const auto extent     = std::vector<real_t>({ 1.0, 100.0, -10.0, 10.0 });
    // optional for GR
    const auto spin       = (real_t)(0.9);
    const auto rh         = ONE + std::sqrt(ONE - SQR(spin));
    // optional for Qspherical
    const auto qsph_r0    = (real_t)(0.0);
    const auto qsph_h     = (real_t)(0.25);

    auto       params     = new real_t[6];
    params[0]             = qsph_r0;
    params[1]             = qsph_h;
    params[4]             = spin;
    params[5]             = rh;
    ntt::Metric<ntt::Dim2> metric(resolution, extent, params);
    delete[] params;

    {
      /* ----------- Test conversion covariant <-> hat <-> contravariant ---------- */
      std::vector<real_t> x1 { HALF, resolution[0] - HALF }, x2 { HALF, resolution[1] - HALF };

      ntt::vec_t<ntt::Dim3> v_cov { 4.4, -3.3, 2.2 };

      auto                  correct = true;

      for (auto i { 0 }; i < 2; ++i) {
        for (auto j { 0 }; j < 2; ++j) {
          ntt::coord_t<ntt::Dim2> xi { x1[i], x2[j] };
          ntt::vec_t<ntt::Dim3>   v_hat_from_cov { ZERO };
          ntt::vec_t<ntt::Dim3>   v_hat_from_cntrv { ZERO };
          ntt::vec_t<ntt::Dim3>   v_cntrv_from_cov { ZERO };
          ntt::vec_t<ntt::Dim3>   v_cntrv_from_hat { ZERO };
          ntt::vec_t<ntt::Dim3>   v_cov_from_cntrv { ZERO };
          ntt::vec_t<ntt::Dim3>   v_cov_from_hat { ZERO };

          metric.v3_Cov2Hat(xi, v_cov, v_hat_from_cov);
          metric.v3_Hat2Cntrv(xi, v_hat_from_cov, v_cntrv_from_hat);
          metric.v3_Hat2Cov(xi, v_hat_from_cov, v_cov_from_hat);
          metric.v3_Cntrv2Cov(xi, v_cntrv_from_hat, v_cov_from_cntrv);
          metric.v3_Cntrv2Hat(xi, v_cntrv_from_hat, v_hat_from_cntrv);
          metric.v3_Cov2Cntrv(xi, v_cov, v_cntrv_from_cov);

          for (auto d { 0 }; d < 3; ++d) {
            correct = correct && ntt::AlmostEqual(v_cov[d], v_cov_from_cntrv[d]);
            correct = correct && ntt::AlmostEqual(v_cov[d], v_cov_from_hat[d]);
            correct = correct && ntt::AlmostEqual(v_cntrv_from_hat[d], v_cntrv_from_cov[d]);
            correct = correct && ntt::AlmostEqual(v_hat_from_cov[d], v_hat_from_cntrv[d]);
          }
        }
      }
      !correct ? throw std::logic_error("Cov2Hat -> Hat2Cntrv -> Cntrv2Cov not correct")
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