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
    const auto extent     = std::vector<real_t>({ 1.0, 2000.0, 0.0, ntt::constant::PI });
    const auto spin       = (real_t)(0.5);
    const auto rh         = ONE + std::sqrt(ONE - SQR(spin));
    // optional for Qspherical
    const auto qsph_r0    = (real_t)(-10.0);
    const auto qsph_h     = (real_t)(0.4);

    auto       params     = new real_t[6];
    params[0]             = qsph_r0;
    params[1]             = qsph_h;
    params[4]             = spin;
    params[5]             = rh;
    ntt::Metric<ntt::Dim2> metric(resolution, extent, params);
    delete[] params;

    {
      /* -------------------- Test components of h_ij and h^ij -------------------- */
      const auto  nx1 = (std::size_t)resolution[0];
      const auto  nx2 = (std::size_t)resolution[1];
      const auto  rg  = metric.getParameter("rg");
      const auto  a   = metric.getParameter("spin") * rg;
      std::size_t correct_cnt { 0 };
      Kokkos::parallel_reduce(
        "Metric components",
        ntt::CreateRangePolicy<ntt::Dim2>({ 0, 0 }, { nx1, nx2 }),
        Lambda(ntt::index_t i1, ntt::index_t i2, std::size_t & correct_cnt_l) {
          ntt::coord_t<ntt::Dim2> xi { (real_t)i1 + HALF, (real_t)i2 + HALF };
          ntt::coord_t<ntt::Dim2> xph { ZERO };

          const auto              h11  = metric.h11(xi);
          const auto              h22  = metric.h22(xi);
          const auto              h33  = metric.h33(xi);
          const auto              h13  = metric.h13(xi);
          const auto              h_11 = metric.h_11(xi);
          const auto              h_22 = metric.h_22(xi);
          const auto              h_33 = metric.h_33(xi);
          const auto              h_13 = metric.h_13(xi);

          metric.x_Code2Sph(xi, xph);
          const auto            r     = xph[0];
          const auto            th    = xph[1];

          const auto            Sigma = SQR(r) + SQR(a * math::cos(th));
          const auto            z     = TWO * r * rg / Sigma;
          const auto            Delta = SQR(r) - TWO * rg * r + SQR(a);
          const auto            A     = SQR(SQR(r) + SQR(a)) - SQR(a * math::sin(th)) * Delta;

          const auto            h11_expect  = A / (Sigma * (Sigma + TWO * r * rg));
          const auto            h22_expect  = ONE / Sigma;
          const auto            h33_expect  = ONE / (Sigma * SQR(math::sin(th)));
          const auto            h13_expect  = a / Sigma;
          const auto            h_11_expect = ONE + z;
          const auto            h_22_expect = Sigma;
          const auto            h_33_expect = A * SQR(math::sin(th)) / Sigma;
          const auto            h_13_expect = -a * (ONE + z) * SQR(math::sin(th));

          ntt::vec_t<ntt::Dim3> hij_temp { ZERO }, hij_predict { ZERO };
          metric.v3_Cntrv2PhysCntrv(xi, { h11, h22, h33 }, hij_temp);
          metric.v3_Cntrv2PhysCntrv(xi, hij_temp, hij_predict);

          ntt::vec_t<ntt::Dim3> h13_predict_temp { ZERO };
          metric.v3_Cntrv2PhysCntrv(xi, { h13, ZERO, ZERO }, hij_temp);
          metric.v3_Cntrv2PhysCntrv(xi, { ZERO, ZERO, hij_temp[0] }, h13_predict_temp);
          const auto            h13_predict = h13_predict_temp[2];

          ntt::vec_t<ntt::Dim3> h_ij_temp { ZERO }, h_ij_predict { ZERO };
          metric.v3_Cov2PhysCov(xi, { h_11, h_22, h_33 }, h_ij_temp);
          metric.v3_Cov2PhysCov(xi, h_ij_temp, h_ij_predict);

          ntt::vec_t<ntt::Dim3> h_13_predict_temp { ZERO };
          metric.v3_Cov2PhysCov(xi, { h_13, ZERO, ZERO }, h_ij_temp);
          metric.v3_Cov2PhysCov(xi, { ZERO, ZERO, h_ij_temp[0] }, h_13_predict_temp);
          const auto h_13_predict  = h_13_predict_temp[2];

          const auto h11isCorrect  = ntt::AlmostEqual(hij_predict[0], h11_expect);
          const auto h22isCorrect  = ntt::AlmostEqual(hij_predict[1], h22_expect);
          const auto h33isCorrect  = ntt::AlmostEqual(hij_predict[2], h33_expect);
          const auto h13isCorrect  = ntt::AlmostEqual(h13_predict, h13_expect);
          const auto h_11isCorrect = ntt::AlmostEqual(h_ij_predict[0], h_11_expect);
          const auto h_22isCorrect = ntt::AlmostEqual(h_ij_predict[1], h_22_expect);
          const auto h_33isCorrect = ntt::AlmostEqual(h_ij_predict[2], h_33_expect);
          const auto h_13isCorrect = ntt::AlmostEqual(h_13_predict, h_13_expect);

          const auto all_correct = h11isCorrect && h22isCorrect && h33isCorrect && h13isCorrect
                                   && h_11isCorrect && h_22isCorrect && h_33isCorrect
                                   && h_13isCorrect;

          correct_cnt_l += (std::size_t)all_correct;
          if (!all_correct) {
            printf("r, th = %f, %f\n", r, th);
            printf("h11 = %f [%f]\n", hij_predict[0], h11_expect);
            printf("h22 = %f [%f]\n", hij_predict[1], h22_expect);
            printf("h33 = %f [%f]\n", hij_predict[2], h33_expect);
            printf("h13 = %f [%f]\n", h13_predict, h13_expect);
            printf("h_11 = %f [%f]\n", h_ij_predict[0], h_11_expect);
            printf("h_22 = %f [%f]\n", h_ij_predict[1], h_22_expect);
            printf("h_33 = %f [%f]\n", h_ij_predict[2], h_33_expect);
            printf("h_13 = %f [%f]\n", h_13_predict, h_13_expect);
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