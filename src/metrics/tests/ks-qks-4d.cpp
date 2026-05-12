#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/qkerr_schild.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

template <Dimension D>
Inline auto equal(const vec_t<D>& a,
                  const vec_t<D>& b,
                  const char*     msg,
                  real_t          acc = ONE) -> bool {
  const auto eps = epsilon * acc;
  for (auto d { 0u }; d < D; ++d) {
    if (not cmp::AlmostEqual(a[d], b[d], eps)) {
      Kokkos::printf("%s [%d]: %.12e != %.12e\n", msg, d, a[d], b[d]);
      return false;
    }
  }
  return true;
}

Inline auto almostZero(real_t val, real_t scale, real_t acc) -> bool {
  return math::abs(val) <= scale * epsilon * acc;
}

template <Dimension D>
Inline void unravel(std::size_t                    idx,
                    tuple_t<std::size_t, D>&       ijk,
                    const tuple_t<std::size_t, D>& res) {
  for (auto d { 0u }; d < D; ++d) {
    ijk[d]  = idx % res[d];
    idx    /= res[d];
  }
}

/**
 * Test g^{mu alpha} * g_{alpha nu} = delta^mu_nu
 */
template <class M>
void testGInverse(const std::vector<std::size_t>&      res,
                  const boundaries_t<real_t>&          ext,
                  const real_t                         acc,
                  const std::map<std::string, real_t>& params) {
  static_assert(M::Dim == 2, "Dim != 2");

  M                            metric(res, ext, params);
  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }

  Kokkos::parallel_for(
    "g_inverse",
    npts,
    Lambda(index_t n) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x { ZERO };
      for (auto d { 0u }; d < M::Dim; ++d) {
        x[d] = (real_t)(idx[d]) + HALF;
      }

      const auto g00 = metric.template g<0, 0>(x);
      const auto g01 = metric.template g<0, 1>(x);
      const auto g11 = metric.template g<1, 1>(x);
      const auto g22 = metric.template g<2, 2>(x);
      const auto g33 = metric.template g<3, 3>(x);
      const auto g13 = metric.template g<1, 3>(x);

      const auto gl00 = metric.template g_<0, 0>(x);
      const auto gl01 = metric.template g_<0, 1>(x);
      const auto gl03 = metric.template g_<0, 3>(x);
      const auto gl11 = metric.template g_<1, 1>(x);
      const auto gl22 = metric.template g_<2, 2>(x);
      const auto gl33 = metric.template g_<3, 3>(x);
      const auto gl13 = metric.template g_<1, 3>(x);

      // g^{0 alpha} g_{alpha 0} = 1
      const auto d00 = g00 * gl00 + g01 * gl01;
      // g^{1 alpha} g_{alpha 1} = 1
      const auto d11 = g01 * gl01 + g11 * gl11 + g13 * gl13;
      // g^{2 alpha} g_{alpha 2} = 1
      const auto d22 = g22 * gl22;
      // g^{3 alpha} g_{alpha 3} = 1
      const auto d33 = g13 * gl13 + g33 * gl33;

      vec_t<Dim::_4D> diag { d00, d11, d22, d33 };
      vec_t<Dim::_4D> diag_expect { ONE, ONE, ONE, ONE };
      if (not equal<Dim::_4D>(diag, diag_expect, "g_inv_diag", acc)) {
        Kokkos::abort("g inverse: diagonal != identity");
      }

      // off-diagonal should be zero
      // g^{0 alpha} g_{alpha 1} = 0
      const auto od01 = g00 * gl01 + g01 * gl11;
      // g^{0 alpha} g_{alpha 3} = 0
      const auto od03 = g00 * gl03 + g01 * gl13;
      // g^{1 alpha} g_{alpha 0} = 0
      const auto od10 = g01 * gl00 + g11 * gl01 + g13 * gl03;
      // g^{1 alpha} g_{alpha 3} = 0
      const auto od13 = g01 * gl03 + g11 * gl13 + g13 * gl33;
      // g^{3 alpha} g_{alpha 0} = 0
      const auto od30 = g13 * gl01 + g33 * gl03;
      // g^{3 alpha} g_{alpha 1} = 0
      const auto od31 = g13 * gl11 + g33 * gl13;

      const auto scale = math::max(math::abs(gl11), math::abs(gl33));
      if (not almostZero(od01, scale, acc) ||
          not almostZero(od03, scale, acc) ||
          not almostZero(od10, scale, acc) ||
          not almostZero(od13, scale, acc) ||
          not almostZero(od30, scale, acc) ||
          not almostZero(od31, scale, acc)) {
        Kokkos::printf("off-diag: %e %e %e %e %e %e\n",
                       od01, od03, od10, od13, od30, od31);
        Kokkos::abort("g inverse: off-diagonal != 0");
      }
    });
}

/**
 * Test spatial components: g_{ij} == h_{ij}
 * Valid for all KS-family metrics (KS0, KS, QKS).
 */
template <class M>
void testGSpatial(const std::vector<std::size_t>&      res,
                  const boundaries_t<real_t>&          ext,
                  const real_t                         acc,
                  const std::map<std::string, real_t>& params) {
  static_assert(M::Dim == 2, "Dim != 2");

  M                            metric(res, ext, params);
  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }

  Kokkos::parallel_for(
    "g_spatial",
    npts,
    Lambda(index_t n) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x_Code { ZERO };
      for (auto d { 0u }; d < M::Dim; ++d) {
        x_Code[d] = (real_t)(idx[d]) + HALF;
      }

      // g_{ij} == h_{ij} for spatial components
      vec_t<Dim::_4D> g_ij {
        metric.template g_<1, 1>(x_Code),
        metric.template g_<2, 2>(x_Code),
        metric.template g_<3, 3>(x_Code),
        metric.template g_<1, 3>(x_Code)
      };
      vec_t<Dim::_4D> h_ij_expect {
        metric.template h_<1, 1>(x_Code),
        metric.template h_<2, 2>(x_Code),
        metric.template h_<3, 3>(x_Code),
        metric.template h_<1, 3>(x_Code)
      };
      if (not equal<Dim::_4D>(g_ij, h_ij_expect, "g_ij==h_ij", acc)) {
        Kokkos::abort("g_spatial: g_{ij} != h_{ij}");
      }
    });
}

/**
 * Test g_00 and g^00 against analytical Kerr formulas: z = 2r/Sigma.
 * Only valid for KerrSchild and QKerrSchild (NOT KerrSchild0).
 */
template <class M>
void testG00Analytical(const std::vector<std::size_t>&      res,
                       const boundaries_t<real_t>&          ext,
                       const real_t                         acc,
                       const std::map<std::string, real_t>& params) {
  static_assert(M::Dim == 2, "Dim != 2");

  M                            metric(res, ext, params);
  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }

  const auto a_spin = metric.spin();

  Kokkos::parallel_for(
    "g00_analytical",
    npts,
    Lambda(index_t n) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x_Code { ZERO };
      coord_t<M::Dim> x_Phys { ZERO };
      for (auto d { 0u }; d < M::Dim; ++d) {
        x_Code[d] = (real_t)(idx[d]) + HALF;
      }

      metric.template convert<Crd::Cd, Crd::Ph>(x_Code, x_Phys);
      const auto r     = x_Phys[0];
      const auto theta = x_Phys[1];
      const auto Sigma = SQR(r) + SQR(a_spin * math::cos(theta));
      const auto z_val = TWO * r / Sigma;

      vec_t<Dim::_1D> g00_code { metric.template g_<0, 0>(x_Code) };
      vec_t<Dim::_1D> g00_expect { -(ONE - z_val) };
      if (not equal<Dim::_1D>(g00_code, g00_expect, "g_00", acc)) {
        Kokkos::abort("g00_analytical: g_00");
      }

      vec_t<Dim::_1D> ginv00_code { metric.template g<0, 0>(x_Code) };
      vec_t<Dim::_1D> ginv00_expect { -(ONE + z_val) };
      if (not equal<Dim::_1D>(ginv00_code, ginv00_expect, "g^00", acc)) {
        Kokkos::abort("g00_analytical: g^00");
      }
    });
}

/**
 * Test u_0: verify g^{mu nu} u_mu u_nu + norm = 0
 * The u_0 function solves: g^{mu nu} u_mu u_nu = -norm
 * For massive particles (g^{mu nu} u_mu u_nu = -1): norm = 1
 * For null (g^{mu nu} u_mu u_nu = 0): norm = 0
 */
template <class M>
void testU0(const std::vector<std::size_t>&      res,
            const boundaries_t<real_t>&          ext,
            const real_t                         acc,
            const std::map<std::string, real_t>& params) {
  static_assert(M::Dim == 2, "Dim != 2");

  M                            metric(res, ext, params);
  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }

  Kokkos::parallel_for(
    "u0_normalization",
    npts,
    Lambda(index_t n) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x { ZERO };
      for (auto d { 0u }; d < M::Dim; ++d) {
        x[d] = (real_t)(idx[d]) + HALF;
      }

      const real_t norm = ONE;

      // Test with zero spatial velocity (covariant)
      {
        vec_t<Dim::_3D> u_i { ZERO, ZERO, ZERO };
        const auto u0 = metric.u_0(x, u_i, norm);

        const auto g00 = metric.template g<0, 0>(x);
        const auto residual = g00 * SQR(u0) + norm;
        if (not almostZero(residual, norm, acc)) {
          Kokkos::printf("u0_zero: g^00*u0^2+norm = %.12e (u0=%.12e)\n",
                         residual, u0);
          Kokkos::abort("u_0: zero velocity normalization failed");
        }
      }

      // Test with non-zero spatial velocity
      {
        vec_t<Dim::_3D> u_i { (real_t)0.1, (real_t)0.05, (real_t)0.02 };
        const auto u0 = metric.u_0(x, u_i, norm);

        const auto g00 = metric.template g<0, 0>(x);
        const auto g01 = metric.template g<0, 1>(x);
        const auto g11 = metric.template g<1, 1>(x);
        const auto g22 = metric.template g<2, 2>(x);
        const auto g33 = metric.template g<3, 3>(x);
        const auto g13 = metric.template g<1, 3>(x);

        const auto contraction = g00 * SQR(u0) +
                                 TWO * g01 * u0 * u_i[0] +
                                 g11 * SQR(u_i[0]) +
                                 g22 * SQR(u_i[1]) +
                                 g33 * SQR(u_i[2]) +
                                 TWO * g13 * u_i[0] * u_i[2];
        const auto residual = contraction + norm;

        if (not almostZero(residual, norm, acc)) {
          Kokkos::printf("u0_nonzero: g^uv u_u u_v + norm = %.12e\n", residual);
          Kokkos::abort("u_0: nonzero velocity normalization failed");
        }
      }

      // Test with norm = 0 (null vector)
      {
        vec_t<Dim::_3D> u_i { (real_t)0.3, ZERO, ZERO };
        const auto u0 = metric.u_0(x, u_i, ZERO);

        const auto g00 = metric.template g<0, 0>(x);
        const auto g01 = metric.template g<0, 1>(x);
        const auto g11 = metric.template g<1, 1>(x);

        const auto contraction = g00 * SQR(u0) +
                                 TWO * g01 * u0 * u_i[0] +
                                 g11 * SQR(u_i[0]);

        if (not almostZero(contraction, math::abs(g00), acc)) {
          Kokkos::printf("u0_null: g^uv u_u u_v = %.12e, expect 0\n",
                         contraction);
          Kokkos::abort("u_0: null normalization failed");
        }
      }
    });
}

/**
 * Test transform_4d: D->U->D and U->D->U roundtrip identity
 */
template <class M>
void testTransform4d(const std::vector<std::size_t>&      res,
                     const boundaries_t<real_t>&          ext,
                     const real_t                         acc,
                     const std::map<std::string, real_t>& params) {
  static_assert(M::Dim == 2, "Dim != 2");

  M                            metric(res, ext, params);
  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }

  Kokkos::parallel_for(
    "transform_4d",
    npts,
    Lambda(index_t n) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x { ZERO };
      for (auto d { 0u }; d < M::Dim; ++d) {
        x[d] = (real_t)(idx[d]) + HALF;
      }

      // D -> U -> D roundtrip
      {
        vec_t<Dim::_4D> v_cov { (real_t)-1.5, (real_t)0.3, (real_t)0.1, (real_t)0.2 };
        vec_t<Dim::_4D> v_cntrv { ZERO };
        vec_t<Dim::_4D> v_cov_back { ZERO };

        metric.template transform_4d<Idx::D, Idx::U>(x, v_cov, v_cntrv);
        metric.template transform_4d<Idx::U, Idx::D>(x, v_cntrv, v_cov_back);

        if (not equal<Dim::_4D>(v_cov_back, v_cov, "D->U->D", acc)) {
          Kokkos::abort("transform_4d: D->U->D roundtrip failed");
        }
      }

      // U -> D -> U roundtrip
      {
        vec_t<Dim::_4D> v_cntrv { (real_t)1.0, (real_t)0.5, (real_t)-0.2, (real_t)0.1 };
        vec_t<Dim::_4D> v_cov { ZERO };
        vec_t<Dim::_4D> v_cntrv_back { ZERO };

        metric.template transform_4d<Idx::U, Idx::D>(x, v_cntrv, v_cov);
        metric.template transform_4d<Idx::D, Idx::U>(x, v_cov, v_cntrv_back);

        if (not equal<Dim::_4D>(v_cntrv_back, v_cntrv, "U->D->U", acc)) {
          Kokkos::abort("transform_4d: U->D->U roundtrip failed");
        }
      }

      // Verify transform_4d<D,U> is consistent with direct g^{mu nu} multiplication
      {
        vec_t<Dim::_4D> v_cov { (real_t)-2.0, (real_t)0.7, (real_t)0.3, (real_t)0.4 };
        vec_t<Dim::_4D> v_cntrv_tr { ZERO };
        metric.template transform_4d<Idx::D, Idx::U>(x, v_cov, v_cntrv_tr);

        const auto g00 = metric.template g<0, 0>(x);
        const auto g01 = metric.template g<0, 1>(x);
        const auto g11 = metric.template g<1, 1>(x);
        const auto g22 = metric.template g<2, 2>(x);
        const auto g33 = metric.template g<3, 3>(x);
        const auto g13 = metric.template g<1, 3>(x);

        vec_t<Dim::_4D> v_cntrv_direct {
          g00 * v_cov[0] + g01 * v_cov[1],
          g01 * v_cov[0] + g11 * v_cov[1] + g13 * v_cov[3],
          g22 * v_cov[2],
          g13 * v_cov[1] + g33 * v_cov[3]
        };

        if (not equal<Dim::_4D>(v_cntrv_tr, v_cntrv_direct, "D->U consistency", acc)) {
          Kokkos::abort("transform_4d: D->U inconsistent with g^{mu nu}");
        }
      }

      // Verify transform_4d<U,D> is consistent with direct g_{mu nu} multiplication
      {
        vec_t<Dim::_4D> v_cntrv { (real_t)1.2, (real_t)-0.4, (real_t)0.6, (real_t)0.15 };
        vec_t<Dim::_4D> v_cov_tr { ZERO };
        metric.template transform_4d<Idx::U, Idx::D>(x, v_cntrv, v_cov_tr);

        const auto gl00 = metric.template g_<0, 0>(x);
        const auto gl01 = metric.template g_<0, 1>(x);
        const auto gl03 = metric.template g_<0, 3>(x);
        const auto gl11 = metric.template g_<1, 1>(x);
        const auto gl22 = metric.template g_<2, 2>(x);
        const auto gl33 = metric.template g_<3, 3>(x);
        const auto gl13 = metric.template g_<1, 3>(x);

        vec_t<Dim::_4D> v_cov_direct {
          gl00 * v_cntrv[0] + gl01 * v_cntrv[1] + gl03 * v_cntrv[3],
          gl01 * v_cntrv[0] + gl11 * v_cntrv[1] + gl13 * v_cntrv[3],
          gl22 * v_cntrv[2],
          gl03 * v_cntrv[0] + gl13 * v_cntrv[1] + gl33 * v_cntrv[3]
        };

        if (not equal<Dim::_4D>(v_cov_tr, v_cov_direct, "U->D consistency", acc)) {
          Kokkos::abort("transform_4d: U->D inconsistent with g_{mu nu}");
        }
      }
    });
}

template <class M>
void runAllTests(const char*                          label,
                 const std::vector<std::size_t>&      res,
                 const boundaries_t<real_t>&          ext,
                 const real_t                         acc,
                 const std::map<std::string, real_t>& params,
                 bool has_kerr_g00 = false) {
  std::cout << "[" << label << "] testGInverse..." << std::endl;
  testGInverse<M>(res, ext, acc, params);
  std::cout << "  passed." << std::endl;

  std::cout << "[" << label << "] testGSpatial..." << std::endl;
  testGSpatial<M>(res, ext, acc, params);
  std::cout << "  passed." << std::endl;

  if (has_kerr_g00) {
    std::cout << "[" << label << "] testG00Analytical..." << std::endl;
    testG00Analytical<M>(res, ext, acc, params);
    std::cout << "  passed." << std::endl;
  }

  std::cout << "[" << label << "] testU0..." << std::endl;
  testU0<M>(res, ext, acc, params);
  std::cout << "  passed." << std::endl;

  std::cout << "[" << label << "] testTransform4d..." << std::endl;
  testTransform4d<M>(res, ext, acc, params);
  std::cout << "  passed." << std::endl;
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    using namespace metric;

    const auto res = std::vector<std::size_t> { 32, 42 };
    const auto ext = boundaries_t<real_t> {
      { 0.8, 10.0 },
      { 0.0, constant::PI }
    };
    const real_t acc = static_cast<real_t>(1e4);

    // ---- KerrSchild0 (a=0, flat spacetime in spherical coords) ----
    std::cout << "=== KerrSchild0 ===" << std::endl;
    runAllTests<KerrSchild0<Dim::_2D>>("KS0", res, ext, acc, {});

    // ---- KerrSchild (full Kerr, spherical KS coords) ----
    std::cout << "=== KerrSchild (a=0.8) ===" << std::endl;
    const auto ks_params_08 = std::map<std::string, real_t> {
      { "a", (real_t)0.8 }
    };
    runAllTests<KerrSchild<Dim::_2D>>("KS a=0.8", res, ext, acc,
                                       ks_params_08, true);

    std::cout << "=== KerrSchild (a=0.95) ===" << std::endl;
    const auto ks_params_095 = std::map<std::string, real_t> {
      { "a", (real_t)0.95 }
    };
    runAllTests<KerrSchild<Dim::_2D>>(
      "KS a=0.95", { 64, 54 }, { { 0.8, 50.0 }, { 0.0, constant::PI } },
      acc, ks_params_095, true);

    // ---- QKerrSchild (h=0, quasi-spherical reduces to spherical) ----
    std::cout << "=== QKerrSchild (a=0.8, h=0) ===" << std::endl;
    const auto qks_h0 = std::map<std::string, real_t> {
      { "r0", -TWO }, { "h", ZERO }, { "a", (real_t)0.8 }
    };
    runAllTests<QKerrSchild<Dim::_2D>>("QKS h=0", res, ext, acc,
                                        qks_h0, true);

    // ---- QKerrSchild (h=0.25, with angular stretching) ----
    std::cout << "=== QKerrSchild (a=0.8, h=0.25) ===" << std::endl;
    const auto qks_h025 = std::map<std::string, real_t> {
      { "r0", -TWO }, { "h", (real_t)0.25 }, { "a", (real_t)0.8 }
    };
    runAllTests<QKerrSchild<Dim::_2D>>("QKS h=0.25", res, ext, acc,
                                        qks_h025, true);

    // ---- QKerrSchild (low spin) ----
    std::cout << "=== QKerrSchild (a=0.3, h=0.1) ===" << std::endl;
    const auto qks_low = std::map<std::string, real_t> {
      { "r0", -TWO }, { "h", (real_t)0.1 }, { "a", (real_t)0.3 }
    };
    runAllTests<QKerrSchild<Dim::_2D>>("QKS a=0.3", res, ext, acc,
                                        qks_low, true);

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
