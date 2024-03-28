#include "wrapper.h"

#include METRIC_HEADER
#include "utilities/qmath.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

template <ntt::Dimension D>
void printCoordinate(const std::string& label, const ntt::coord_t<D>& x);

template <>
void printCoordinate<ntt::Dim2>(const std::string&             label,
                                const ntt::coord_t<ntt::Dim2>& x) {
  printf("%s: %f %f\n", label.c_str(), x[0], x[1]);
}

template <>
void printCoordinate<ntt::Dim3>(const std::string&             label,
                                const ntt::coord_t<ntt::Dim3>& x) {
  printf("%s: %f %f %f\n", label.c_str(), x[0], x[1], x[2]);
}

void printVector(const std::string& label, const ntt::vec_t<ntt::Dim3>& v) {
  printf("%s: %f %f %f\n", label.c_str(), v[0], v[1], v[2]);
}

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    const auto resolution = std::vector<unsigned int>({ 2560, 1920 });
#ifdef MINKOWSKI_METRIC
    const auto extent = std::vector<real_t>({ 1.0, 100.0, -30.0, 44.25 });
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

    // acceptable error for cartesian/spherical transformations
    const real_t tinyCart = 1e-3;

    std::vector<real_t> x1 { HALF, resolution[0] - HALF },
      x2 { HALF, resolution[1] - HALF };

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

          const auto correct1 = ntt::AlmostEqual<ntt::Dim3>(v_cov,
                                                            v_cov_from_cntrv);
          const auto correct2 = ntt::AlmostEqual<ntt::Dim3>(v_cov, v_cov_from_hat);
          const auto correct3    = ntt::AlmostEqual<ntt::Dim3>(v_cntrv_from_hat,
                                                            v_cntrv_from_cov);
          const auto correct4    = ntt::AlmostEqual<ntt::Dim3>(v_hat_from_cov,
                                                            v_hat_from_cntrv);
          const auto all_correct = correct1 && correct2 && correct3 && correct4;

          if (!all_correct) {
            printCoordinate<ntt::Dim2>("xi", xi);
            if (!correct1) {
              printVector("v_cov", v_cov);
              printVector("v_cov_from_cntrv", v_cov_from_cntrv);
            }
            if (!correct2) {
              printVector("v_cov", v_cov);
              printVector("v_cov_from_hat", v_cov_from_hat);
            }
            if (!correct3) {
              printVector("v_cntrv_from_hat", v_cntrv_from_hat);
              printVector("v_cntrv_from_cov", v_cntrv_from_cov);
            }
            if (!correct4) {
              printVector("v_hat_from_cov", v_hat_from_cov);
              printVector("v_hat_from_cntrv", v_hat_from_cntrv);
            }
          }

          correct = correct && all_correct;
        }
      }
      !correct ? throw std::logic_error(
                   "Cov2Hat <-> Hat2Cntrv <-> Cntrv2Cov not correct")
               : (void)0;
    }

    {
      /* ------------ Test conversion code <-> spherical <-> cartesian ------------ */
      auto correct = true;
      for (auto i { 0 }; i < 2; ++i) {
        for (auto j { 0 }; j < 2; ++j) {
          ntt::coord_t<ntt::Dim2> xi { x1[i], x2[j] };
#ifdef MINKOWSKI_METRIC
          ntt::coord_t<ntt::Dim2> xi3D { x1[i], x2[j] };
          ntt::coord_t<ntt::Dim2> xi_from_cart { ZERO }, xcart_from_code { ZERO };
#else
          ntt::coord_t<ntt::Dim3> xi3D { x1[i], x2[j], ZERO };
          ntt::coord_t<ntt::Dim3> xi_from_cart { ZERO }, xcart_from_code { ZERO };
#endif
          ntt::coord_t<ntt::Dim2> xi_from_sph { ZERO };
          ntt::coord_t<ntt::Dim2> xsph_from_code { ZERO };
          // xcart_from_code { ZERO };

          metric.x_Code2Sph(xi, xsph_from_code);
          metric.x_Code2Cart(xi3D, xcart_from_code);
          metric.x_Sph2Code(xsph_from_code, xi_from_sph);
          metric.x_Cart2Code(xcart_from_code, xi_from_cart);

          const auto correct1 = ntt::AlmostEqual<ntt::Dim2>(xi, xi_from_sph, tinyCart);

#ifdef MINKOWSKI_METRIC
          const auto correct2 = ntt::AlmostEqual<ntt::Dim2>(xi3D,
                                                            xi_from_cart,
                                                            tinyCart);
#else
          const auto correct2 = ntt::AlmostEqual<ntt::Dim3>(xi3D,
                                                            xi_from_cart,
                                                            tinyCart);
#endif
          const auto all_correct = correct1 && correct2;

          if (!all_correct) {
            printCoordinate<ntt::Dim2>("xi", xi);
            if (!correct1) {
              printCoordinate<ntt::Dim2>("xi_from_sph", xi_from_sph);
            }
            if (!correct2) {
#ifdef MINKOWSKI_METRIC
              printCoordinate<ntt::Dim2>("xi_from_cart", xi_from_cart);
#else
              printCoordinate<ntt::Dim3>("xi_from_cart", xi_from_cart);
#endif
            }
            printCoordinate<ntt::Dim2>("xsph_from_code", xsph_from_code);

#ifdef MINKOWSKI_METRIC
            printCoordinate<ntt::Dim2>("xcart_from_code", xcart_from_code);
#else
            printCoordinate<ntt::Dim3>("xcart_from_code", xcart_from_code);
#endif
          }
          correct = correct && all_correct;
        }
      }
      !correct
        ? throw std::logic_error(
            "Code2Cart <-> Cart2Code or Code2Sph <-> Sph2Code not correct")
        : (void)0;
    }

    {
      /* --------- Test conversion codeCntrv/codeCov <-> physCntrv/physCov -------- */
      ntt::vec_t<ntt::Dim3> v_cntrv { -5.4, -25.3, 12.5 };
      ntt::vec_t<ntt::Dim3> v_cov { 15.2, 12.3, -12.3 };
      const auto v_norm = v_cntrv[0] * v_cov[0] + v_cntrv[1] * v_cov[1] +
                          v_cntrv[2] * v_cov[2];
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

          const auto v_normPhys = v_physCntrv[0] * v_physCov[0] +
                                  v_physCntrv[1] * v_physCov[1] +
                                  v_physCntrv[2] * v_physCov[2];

          const auto correct1 = ntt::AlmostEqual<ntt::Dim3>(v_cntrv,
                                                            v_cntrv_fromPhys);
          const auto correct2 = ntt::AlmostEqual<ntt::Dim3>(v_cov, v_cov_fromPhys);
          const auto correct3    = ntt::AlmostEqual(v_norm, v_normPhys);
          const auto all_correct = correct1 && correct2 && correct3;

          if (!all_correct) {
            printCoordinate<ntt::Dim2>("xi", xi);
            if (!correct1) {
              printVector("v_cntrv", v_cntrv);
              printVector("v_cntrv_fromPhys", v_cntrv_fromPhys);
            }
            if (!correct2) {
              printVector("v_cov", v_cov);
              printVector("v_cov_fromPhys", v_cov_fromPhys);
            }
            if (!correct3) {
              printVector("v_physCntrv", v_physCntrv);
              printVector("v_physCov", v_physCov);
              printf("v_norm: %f, v_normPhys: %f\n", v_norm, v_normPhys);
            }
          }
          correct = correct && all_correct;
        }
      }
      !correct ? throw std::logic_error(
                   "codeCntrv/codeCov <-> physCntrv/physCov not correct")
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

            const auto correct1 = ntt::AlmostEqual<ntt::Dim3>(v_cntrv,
                                                              v_cntrv_fromCart,
                                                              tinyCart);
            const auto correct2 = ntt::AlmostEqual<ntt::Dim3>(v_cov,
                                                              v_cov_fromCart,
                                                              tinyCart);
            const auto correct3 = ntt::AlmostEqual<ntt::Dim3>(v_cart_fromCntrv,
                                                              v_cart_fromCov,
                                                              tinyCart);
            const auto all_correct = correct1 && correct2 && correct3;

            if (!all_correct) {
              printCoordinate<ntt::Dim2>("xi", xi);
              if (!correct1) {
                printVector("v_cntrv", v_cntrv);
                printVector("v_cntrv_fromCart", v_cntrv_fromCart);
              }
              if (!correct2) {
                printVector("v_cov", v_cov);
                printVector("v_cov_fromCart", v_cov_fromCart);
              }
              if (!correct3) {
                printVector("v_cart_fromCov", v_cart_fromCov);
                printVector("v_cart_fromCntrv", v_cart_fromCntrv);
              }
            }
            correct = correct && all_correct;
          }
        }
      }
      !correct ? throw std::logic_error("cart <-> cntrv <-> cov not correct")
               : (void)0;
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}