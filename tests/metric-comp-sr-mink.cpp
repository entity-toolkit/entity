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
    const auto resolution = std::vector<unsigned int>({ 200, 500, 100 });
    const auto extent     = std::vector<real_t>({ -15.0, 15.0, -75.0, 0.0, -5.0, 10.0 });

    auto       params     = new real_t[2];
    ntt::Metric<ntt::Dim3> metric(resolution, extent, params);
    delete[] params;

    {
      /* ------------------------- Test components of h_ij ------------------------ */
      const auto nx1 = (int)resolution[0];
      const auto nx2 = (int)resolution[1];
      const auto nx3 = (int)resolution[2];
      bool       correct;
      Kokkos::parallel_reduce(
        "Metric components",
        nx1 * nx2 * nx3,
        Lambda(ntt::index_t i, bool& correct_l) {
          const auto              i1_ = static_cast<real_t>(i % nx1);
          const auto              i2_ = static_cast<real_t>((i / nx1) % nx2);
          const auto              i3_ = static_cast<real_t>(i / (nx1 * nx2));
          ntt::coord_t<ntt::Dim3> xi { i1_ + HALF, i2_ + HALF, i3_ + HALF };

          const auto              h_11        = metric.h_11(xi);
          const auto              h_22        = metric.h_22(xi);
          const auto              h_33        = metric.h_33(xi);

          const auto              h_11_expect = ONE;
          const auto              h_22_expect = ONE;
          const auto              h_33_expect = ONE;

          ntt::vec_t<ntt::Dim3>   h_ij_temp { ZERO }, h_ij_predict { ZERO };
          metric.v3_Cov2PhysCov(xi, { h_11, h_22, h_33 }, h_ij_temp);
          metric.v3_Cov2PhysCov(xi, h_ij_temp, h_ij_predict);

          const auto h_11isCorrect = ntt::AlmostEqual(h_ij_predict[0], h_11_expect);
          const auto h_22isCorrect = ntt::AlmostEqual(h_ij_predict[1], h_22_expect);
          const auto h_33isCorrect = ntt::AlmostEqual(h_ij_predict[2], h_33_expect);

          const auto all_correct   = h_11isCorrect && h_22isCorrect && h_33isCorrect;

          correct_l                = correct_l && all_correct;
          if (!all_correct) {
            printf("h_11 = %f [%f]\n", h_ij_predict[0], h_11_expect);
            printf("h_22 = %f [%f]\n", h_ij_predict[1], h_22_expect);
            printf("h_33 = %f [%f]\n", h_ij_predict[2], h_33_expect);
          }
        },
        Kokkos::LAnd<bool>(correct));
      (!correct) ? throw std::logic_error("Metric is incorrect") : (void)0;
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();

  return 0;
}