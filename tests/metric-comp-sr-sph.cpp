#include "wrapper.h"

#include METRIC_HEADER
#include "utils/qmath.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    const auto resolution = std::vector<unsigned int>({ 1080, 512 });
    const auto extent     = std::vector<real_t>({ 0.25, 1000.0, 0.0, ntt::constant::PI });
    // optional for Qspherical
    const auto qsph_r0    = (real_t)(0.0);
    const auto qsph_h     = (real_t)(0.2);

    auto       params     = new real_t[2];
    params[0]             = qsph_r0;
    params[1]             = qsph_h;
    ntt::Metric<ntt::Dim2> metric(resolution, extent, params);
    delete[] params;

    {
      /* ------------------------- Test components of h_ij ------------------------ */
      const auto  nx1 = (std::size_t)resolution[0];
      const auto  nx2 = (std::size_t)resolution[1];
      std::size_t correct_cnt { 0 };
      Kokkos::parallel_reduce(
        "Metric components",
        ntt::CreateRangePolicy<ntt::Dim2>({ 0, 0 }, { nx1, nx2 }),
        Lambda(ntt::index_t i1, ntt::index_t i2, std::size_t & correct_cnt_l) {
          ntt::coord_t<ntt::Dim2> xi { (real_t)i1 + HALF, (real_t)i2 + HALF };
          ntt::coord_t<ntt::Dim2> xph { ZERO };

          const auto              h_11 = metric.h_11(xi);
          const auto              h_22 = metric.h_22(xi);
          const auto              h_33 = metric.h_33(xi);

          metric.x_Code2Sph(xi, xph);
          const auto            r           = xph[0];
          const auto            th          = xph[1];

          const auto            h_11_expect = ONE;
          const auto            h_22_expect = SQR(r);
          const auto            h_33_expect = SQR(r * math::sin(th));

          ntt::vec_t<ntt::Dim3> h_ij_temp { ZERO }, h_ij_predict { ZERO };
          metric.v3_Cov2PhysCov(xi, { h_11, h_22, h_33 }, h_ij_temp);
          metric.v3_Cov2PhysCov(xi, h_ij_temp, h_ij_predict);

          const auto h_11isCorrect = ntt::AlmostEqual(h_ij_predict[0], h_11_expect);
          const auto h_22isCorrect = ntt::AlmostEqual(h_ij_predict[1], h_22_expect);
          const auto h_33isCorrect = ntt::AlmostEqual(h_ij_predict[2], h_33_expect);

          const auto all_correct   = h_11isCorrect && h_22isCorrect && h_33isCorrect;

          correct_cnt_l += (std::size_t)all_correct;
          if (!all_correct) {
            printf("r, th = %f, %f\n", r, th);
            printf("h_11 = %f [%f]\n", h_ij_predict[0], h_11_expect);
            printf("h_22 = %f [%f]\n", h_ij_predict[1], h_22_expect);
            printf("h_33 = %f [%f]\n", h_ij_predict[2], h_33_expect);
          }
        },
        correct_cnt);
      if (correct_cnt != nx1 * nx2) {
        throw std::logic_error("Metric is incorrect: " + std::to_string(correct_cnt)
                               + " out of " + std::to_string(nx1 * nx2) + " are correct.");
      }
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}