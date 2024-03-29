#include "wrapper.h"

#include METRIC_HEADER
#include "utils/qmath.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

void printCoordinate(const std::string& label, const ntt::coord_t<ntt::Dim2>& x) {
  printf("%s: %f %f\n", label.c_str(), x[0], x[1]);
}

void printVector(const std::string& label, const ntt::vec_t<ntt::Dim3>& v) {
  printf("%s: %f %f %f\n", label.c_str(), v[0], v[1], v[2]);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    const auto resolution = std::vector<unsigned int>({ 2560, 1920 });
    const auto extent     = std::vector<real_t>({ 1.0, 100.0, 1.0, 100.0 });
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

    // acceptable error for cartesian/spherical transformations
    const real_t        tinyCart = 1e-3;

    std::vector<real_t> x1 { HALF, resolution[0] - HALF }, x2 { HALF, resolution[1] - HALF };

    {
      /* ----------- Test conversion covariant <-> hat <-> contravariant ---------- */
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

          const auto correct_1 = ntt::AlmostEqual<ntt::Dim3>(v_cov, v_cov_from_cntrv);
          const auto correct_2 = ntt::AlmostEqual<ntt::Dim3>(v_cov, v_cov_from_hat);
          const auto correct_3
            = ntt::AlmostEqual<ntt::Dim3>(v_cntrv_from_hat, v_cntrv_from_cov);
          const auto correct_4 = ntt::AlmostEqual<ntt::Dim3>(v_hat_from_cov, v_hat_from_cntrv);
          const auto all_correct = correct_1 && correct_2 && correct_3 && correct_4;

          if (!all_correct) {
            printCoordinate("xi", xi);
            printVector("v_cov", v_cov);
            printVector("v_hat_from_cov", v_hat_from_cov);
            printVector("v_cntrv_from_hat", v_cntrv_from_hat);
            printVector("v_cov_from_hat", v_cov_from_hat);
            printVector("v_hat_from_cntrv", v_hat_from_cntrv);
            printVector("v_cntrv_from_cov", v_cntrv_from_cov);
            printVector("v_cov_from_cntrv", v_cov_from_cntrv);
          }

          correct = correct && all_correct;
        }
      }
      !correct ? throw std::logic_error("Cov2Hat <-> Hat2Cntrv <-> Cntrv2Cov not correct")
               : (void)0;
    }

    {
      /* ------------ Test conversion code <-> spherical <-> cartesian ------------ */
      auto correct = true;
      for (auto i { 0 }; i < 2; ++i) {
        for (auto j { 0 }; j < 2; ++j) {
          ntt::coord_t<ntt::Dim2> xi { x1[i], x2[j] };
          ntt::coord_t<ntt::Dim2> xi_from_sph { ZERO }, xi_from_cart { ZERO };
          ntt::coord_t<ntt::Dim2> xsph_from_code { ZERO }, xcart_from_code { ZERO };

          metric.x_Code2Sph(xi, xsph_from_code);
          metric.x_Code2Cart(xi, xcart_from_code);
          metric.x_Sph2Code(xsph_from_code, xi_from_sph);
          metric.x_Cart2Code(xcart_from_code, xi_from_cart);

          const auto correct1    = ntt::AlmostEqual<ntt::Dim2>(xi, xi_from_sph, tinyCart);
          const auto correct2    = ntt::AlmostEqual<ntt::Dim2>(xi, xi_from_cart, tinyCart);
          const auto all_correct = correct1 && correct2;

          if (!all_correct) {
            printCoordinate("xi", xi);
            printCoordinate("xi_from_sph", xi_from_sph);
            printCoordinate("xi_from_cart", xi_from_cart);
            printCoordinate("xsph_from_code", xsph_from_code);
            printCoordinate("xcart_from_code", xcart_from_code);
          }
          correct = correct && all_correct;
        }
      }
      !correct ? throw std::logic_error(
        "Code2Cart <-> Cart2Code or Code2Sph <-> Sph2Code not correct")
               : (void)0;
    }

    {
      /* --------- Test conversion codeCntrv/codeCov <-> physCntrv/physCov -------- */
      ntt::vec_t<ntt::Dim3> v_cntrv { -5.4, -25.3, 12.5 };
      ntt::vec_t<ntt::Dim3> v_cov { 15.2, 12.3, -12.3 };
      const auto            v_norm
        = v_cntrv[0] * v_cov[0] + v_cntrv[1] * v_cov[1] + v_cntrv[2] * v_cov[2];
      auto correct = true;
      for (auto i { 0 }; i < 2; ++i) {
        for (auto j { 0 }; j < 2; ++j) {
          ntt::coord_t<ntt::Dim2> xi { x1[i], x2[j] };
          ntt::vec_t<ntt::Dim3>   v_physCntrv { ZERO };
          ntt::vec_t<ntt::Dim3>   v_physCov { ZERO };
          ntt::vec_t<ntt::Dim3>   v_cntrv_fromPhys { ZERO };
          ntt::vec_t<ntt::Dim3>   v_cov_fromPhys { ZERO };

          metric.v3_Cntrv2PhysCntrv(xi, v_cntrv, v_physCntrv);
          metric.v3_Cov2PhysCov(xi, v_cov, v_physCov);
          metric.v3_PhysCntrv2Cntrv(xi, v_physCntrv, v_cntrv_fromPhys);
          metric.v3_PhysCov2Cov(xi, v_physCov, v_cov_fromPhys);

          const auto v_normPhys = v_physCntrv[0] * v_physCov[0] + v_physCntrv[1] * v_physCov[1]
                                  + v_physCntrv[2] * v_physCov[2];

          const auto correct1    = ntt::AlmostEqual<ntt::Dim3>(v_cntrv, v_cntrv_fromPhys);
          const auto correct2    = ntt::AlmostEqual<ntt::Dim3>(v_cov, v_cov_fromPhys);
          const auto correct3    = ntt::AlmostEqual(v_norm, v_normPhys);
          const auto all_correct = correct1 && correct2 && correct3;

          if (!all_correct) {
            printCoordinate("xi", xi);
            printVector("v_cntrv", v_cntrv);
            printVector("v_physCntrv", v_physCntrv);
            printVector("v_cntrv_fromPhys", v_cntrv_fromPhys);
            printVector("v_cov", v_cov);
            printVector("v_physCov", v_physCov);
            printVector("v_cov_fromPhys", v_cov_fromPhys);
            printf("v_norm = %f, v_normPhys = %f\n", v_norm, v_normPhys);
          }
          correct = correct && all_correct;
        }
      }
      !correct ? throw std::logic_error("codeCntrv/codeCov <-> physCntrv/physCov not correct")
               : (void)0;
    }

    {
      /* ----------------- Test conversion cart <-> cntrv <-> cov ----------------- */
      std::vector<real_t>   x3 = { 0.0, 1.25, 4.5 };
      ntt::vec_t<ntt::Dim3> v_cov { -1.0, 2.0, -3.0 };
      auto                  correct = true;
      for (auto i { 0 }; i < 2; ++i) {
        for (auto j { 0 }; j < 2; ++j) {
          for (auto k { 0 }; k < 3; ++k) {
            ntt::coord_t<ntt::Dim2> xi { x1[i], x2[j] };
#ifdef MINKOWSKI_METRIC
            ntt::coord_t<ntt::Dim2> xi3D { x1[i], x2[j] };
#else
            ntt::coord_t<ntt::Dim3> xi3D { x1[i], x2[j], x3[k] };
#endif
            ntt::vec_t<ntt::Dim3> v_cntrv { ZERO };
            ntt::vec_t<ntt::Dim3> v_cntrv_fromCart { ZERO };
            ntt::vec_t<ntt::Dim3> v_cov_fromCart { ZERO };
            ntt::vec_t<ntt::Dim3> v_cart_fromCov { ZERO };
            ntt::vec_t<ntt::Dim3> v_cart_fromCntrv { ZERO };

            metric.v3_Cov2Cntrv(xi, v_cov, v_cntrv);
            metric.v3_Cov2Cart(xi3D, v_cov, v_cart_fromCov);
            metric.v3_Cntrv2Cart(xi3D, v_cntrv, v_cart_fromCntrv);

            metric.v3_Cart2Cntrv(xi3D, v_cart_fromCov, v_cntrv_fromCart);
            metric.v3_Cart2Cov(xi3D, v_cart_fromCntrv, v_cov_fromCart);

            const auto correct1
              = ntt::AlmostEqual<ntt::Dim3>(v_cntrv, v_cntrv_fromCart, tinyCart);
            const auto correct2 = ntt::AlmostEqual<ntt::Dim3>(v_cov, v_cov_fromCart, tinyCart);
            const auto correct3
              = ntt::AlmostEqual<ntt::Dim3>(v_cart_fromCntrv, v_cart_fromCov, tinyCart);
            const auto all_correct = correct1 && correct2 && correct3;

            if (!all_correct) {
              printCoordinate("xi", xi);
              printVector("v_cov", v_cov);
              printVector("v_cntrv", v_cntrv);
              printVector("v_cntrv_fromCart", v_cntrv_fromCart);
              printVector("v_cov_fromCart", v_cov_fromCart);
              printVector("v_cart_fromCov", v_cart_fromCov);
              printVector("v_cart_fromCntrv", v_cart_fromCntrv);
            }
            correct = correct && all_correct;
          }
        }
      }
      !correct ? throw std::logic_error("cart <-> cntrv <-> cov not correct") : (void)0;
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();

  return 0;
}