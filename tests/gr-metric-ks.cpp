#include "wrapper.h"

#include "io/input.h"
#include "metrics/kerr_schild.h"
#include "utils/qmath.h"

#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <toml.hpp>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    using namespace toml::literals::toml_literals;
    const auto inputdata      = R"(
      [simulation]
      title   = "gr-metric-ks"

      [domain]
      a           = 0.0
      resolution  = [512, 256]
      extent      = [1.0, 20.0]
    )"_toml;

    const auto log_level_enum = plog::verbose;
    const auto sim_title = ntt::readFromInput<std::string>(inputdata, "simulation", "title");
    const auto logfile_name  = sim_title + ".log";
    const auto infofile_name = sim_title + ".info";
    std::remove(logfile_name.c_str());
    std::remove(infofile_name.c_str());
    plog::RollingFileAppender<plog::TxtFormatter>     logfileAppender(logfile_name.c_str());
    plog::RollingFileAppender<plog::Nt2InfoFormatter> infofileAppender(infofile_name.c_str());
    plog::init<ntt::LogFile>(log_level_enum, &logfileAppender);
    plog::init<ntt::InfoFile>(plog::verbose, &infofileAppender);

    const auto resolution
      = ntt::readFromInput<std::vector<unsigned int>>(inputdata, "domain", "resolution");
    const auto extent = ntt::readFromInput<std::vector<real_t>>(inputdata, "domain", "extent");
    const auto spin   = ntt::readFromInput<real_t>(inputdata, "domain", "a");
    const auto rh     = ONE + std::sqrt(ONE - SQR(spin));
    auto       params = new real_t[6];
    params[4]         = spin;
    params[5]         = rh;
    ntt::Metric<ntt::Dim2> ks_metric(resolution, extent, params);
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

          ks_metric.v3_Cov2Hat(xi, v_cov, v_hat_from_cov);
          ks_metric.v3_Hat2Cntrv(xi, v_hat_from_cov, v_cntrv_from_hat);
          ks_metric.v3_Hat2Cov(xi, v_hat_from_cov, v_cov_from_hat);
          ks_metric.v3_Cntrv2Cov(xi, v_cntrv_from_hat, v_cov_from_cntrv);
          ks_metric.v3_Cntrv2Hat(xi, v_cntrv_from_hat, v_hat_from_cntrv);
          ks_metric.v3_Cov2Cntrv(xi, v_cov, v_cntrv_from_cov);

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
    {
      /* -------------------- Test components of h_ij and h^ij -------------------- */
      auto range = ntt::CreateRangePolicy<ntt::Dim2>(
        { 0, 0 }, { (int)resolution[0], (int)resolution[1] });
      int        correct;
      const auto dr = (extent[1] - extent[0]) / static_cast<real_t>(resolution[0]);
      const auto dth
        = static_cast<real_t>(ntt::constant::PI) / static_cast<real_t>(resolution[1]);
      Kokkos::parallel_reduce(
        "test-h13",
        range,
        Lambda(ntt::index_t i1, ntt::index_t i2, int& correct_l) {
          const auto              i1_ = static_cast<real_t>(i1);
          const auto              i2_ = static_cast<real_t>(i2);
          ntt::coord_t<ntt::Dim2> xi { i1_ + HALF, i2_ + HALF };
          ntt::coord_t<ntt::Dim2> xph { ZERO };

          const auto              h11  = ks_metric.h11(xi);
          const auto              h22  = ks_metric.h22(xi);
          const auto              h33  = ks_metric.h33(xi);
          const auto              h13  = ks_metric.h13(xi);
          const auto              h_11 = ks_metric.h_11(xi);
          const auto              h_22 = ks_metric.h_22(xi);
          const auto              h_33 = ks_metric.h_33(xi);
          const auto              h_13 = ks_metric.h_13(xi);

          ks_metric.x_Code2Sph(xi, xph);
          const auto r             = xph[0];
          const auto th            = xph[1];

          const auto h11_expect    = (r / (r + TWO)) / SQR(dr);
          const auto h22_expect    = (ONE / SQR(r)) / SQR(dth);
          const auto h33_expect    = ONE / SQR(r * math::sin(th));
          const auto h13_expect    = ZERO;
          const auto h_11_expect   = SQR(dr) * (ONE + TWO / r);
          const auto h_22_expect   = SQR(dth) * SQR(r);
          const auto h_33_expect   = SQR(r * math::sin(th));
          const auto h_13_expect   = ZERO;

          const auto h11isCorrect  = ntt::AlmostEqual(h11, h11_expect);
          const auto h22isCorrect  = ntt::AlmostEqual(h22, h22_expect);
          const auto h33isCorrect  = ntt::AlmostEqual(h33, h33_expect);
          const auto h13isCorrect  = ntt::AlmostEqual(h13, h13_expect);
          const auto h_11isCorrect = ntt::AlmostEqual(h_11, h_11_expect);
          const auto h_22isCorrect = ntt::AlmostEqual(h_22, h_22_expect);
          const auto h_33isCorrect = ntt::AlmostEqual(h_33, h_33_expect);
          const auto h_13isCorrect = ntt::AlmostEqual(h_13, h_13_expect);

          const auto all_correct = h11isCorrect && h22isCorrect && h33isCorrect && h13isCorrect
                                   && h_11isCorrect && h_22isCorrect && h_33isCorrect
                                   && h_13isCorrect;

          correct_l = correct_l && all_correct;
          if (!all_correct) {
            printf("i1, i2 = %lu, %lu\n", i1, i2);
            printf("r, th = %f, %f\n", r, th);
            printf("h11 = %f [%f]\n", h11, h11_expect);
            printf("h22 = %f [%f]\n", h22, h22_expect);
            printf("h33 = %f [%f]\n", h33, h33_expect);
            printf("h13 = %f [%f]\n", h13, h13_expect);
            printf("h_11 = %f [%f]\n", h_11, h_11_expect);
            printf("h_22 = %f [%f]\n", h_22, h_22_expect);
            printf("h_33 = %f [%f]\n", h_33, h_33_expect);
            printf("h_13 = %f [%f]\n", h_13, h_13_expect);
          }
        },
        Kokkos::LAnd<int>(correct));
      (!correct) ? throw std::logic_error("h13 and/or h_13 are non-zero") : (void)0;
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();

  return 0;
}