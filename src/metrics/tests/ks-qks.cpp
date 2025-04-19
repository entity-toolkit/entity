#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/kerr_schild.h"
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
      printf("%s: %.12e : %.12e\n", msg, a[d], b[d]);
      return false;
    }
  }
  return true;
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

template <class M>
void testMetric(const std::vector<std::size_t>&      res,
                const boundaries_t<real_t>&          ext,
                const real_t                         acc    = ONE,
                const std::map<std::string, real_t>& params = {}) {
  static_assert(M::Dim == 2, "Dim != 2");
  errorIf(res.size() != (std::size_t)(M::Dim), "res.size() != M.dim");
  errorIf(ext.size() != (std::size_t)(M::Dim), "ext.size() != M.dim");
  for (const auto& e : ext) {
    errorIf(e.first >= e.second, "e.first >= e.second");
  }

  M                            metric(res, ext, params);
  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }

  const auto rg = metric.rg();
  const auto a  = metric.spin();
  Kokkos::parallel_for(
    "h_ij/hij",
    npts,
    Lambda(index_t n) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x_Code { ZERO };
      coord_t<M::Dim> x_Phys { ZERO };

      for (auto d { 0u }; d < M::Dim; ++d) {
        x_Code[d] = (real_t)(idx[d]) + HALF;
      }

      const auto h11  = metric.template h<1, 1>(x_Code);
      const auto h22  = metric.template h<2, 2>(x_Code);
      const auto h33  = metric.template h<3, 3>(x_Code);
      const auto h13  = metric.template h<1, 3>(x_Code);
      const auto h_11 = metric.template h_<1, 1>(x_Code);
      const auto h_22 = metric.template h_<2, 2>(x_Code);
      const auto h_33 = metric.template h_<3, 3>(x_Code);
      const auto h_13 = metric.template h_<1, 3>(x_Code);

      metric.template convert<Crd::Cd, Crd::Ph>(x_Code, x_Phys);
      const auto r  = x_Phys[0];
      const auto th = x_Phys[1];

      const auto Sigma = SQR(r) + SQR(a * math::cos(th));
      const auto z     = TWO * r * rg / Sigma;
      const auto Delta = SQR(r) - TWO * rg * r + SQR(a);
      const auto A     = SQR(SQR(r) + SQR(a)) - SQR(a * math::sin(th)) * Delta;

      const auto h11_expect  = A / (Sigma * (Sigma + TWO * r * rg));
      const auto h22_expect  = ONE / Sigma;
      const auto h33_expect  = ONE / (Sigma * SQR(math::sin(th)));
      const auto h13_expect  = a / Sigma;
      const auto h_11_expect = ONE + z;
      const auto h_22_expect = Sigma;
      const auto h_33_expect = A * SQR(math::sin(th)) / Sigma;
      const auto h_13_expect = -a * (ONE + z) * SQR(math::sin(th));

      vec_t<Dim::_3D> hij_temp { ZERO }, hij_predict { ZERO };
      metric.template transform<Idx::U, Idx::PU>(x_Code, { h11, h22, h33 }, hij_temp);
      metric.template transform<Idx::U, Idx::PU>(x_Code, hij_temp, hij_predict);

      vec_t<Dim::_3D> h13_predict_temp { ZERO };
      metric.template transform<Idx::U, Idx::PU>(x_Code,
                                                 { h13, ZERO, ZERO },
                                                 hij_temp);
      metric.template transform<Idx::U, Idx::PU>(x_Code,
                                                 { ZERO, ZERO, hij_temp[0] },
                                                 h13_predict_temp);
      const vec_t<Dim::_1D> h13_predict { h13_predict_temp[2] };

      vec_t<Dim::_3D> h_ij_temp { ZERO }, h_ij_predict { ZERO };
      metric.template transform<Idx::D, Idx::PD>(x_Code,
                                                 { h_11, h_22, h_33 },
                                                 h_ij_temp);
      metric.template transform<Idx::D, Idx::PD>(x_Code, h_ij_temp, h_ij_predict);

      vec_t<Dim::_3D> h_13_predict_temp { ZERO };
      metric.template transform<Idx::D, Idx::PD>(x_Code,
                                                 { h_13, ZERO, ZERO },
                                                 h_ij_temp);
      metric.template transform<Idx::D, Idx::PD>(x_Code,
                                                 { ZERO, ZERO, h_ij_temp[0] },
                                                 h_13_predict_temp);
      const vec_t<Dim::_1D> h_13_predict { h_13_predict_temp[2] };

      vec_t<Dim::_3D> hij_expect { h11_expect, h22_expect, h33_expect };
      vec_t<Dim::_3D> h_ij_expect { h_11_expect, h_22_expect, h_33_expect };

      if (not equal<Dim::_3D>(h_ij_predict, h_ij_expect, "h_ij", acc)) {
        printf("h_ij: %.12e %.12e %.12e : %.12e %.12e %.12e\n",
               h_ij_predict[0],
               h_ij_predict[1],
               h_ij_predict[2],
               h_ij_expect[0],
               h_ij_expect[1],
               h_ij_expect[2]);
        Kokkos::abort("h_ij");
      }
      if (not equal<Dim::_1D>(h_13_predict, { h_13_expect }, "h_13", acc)) {
        printf("h_13: %.12e : %.12e\n", h_13_predict[0], h_13_expect);
        Kokkos::abort("h_13");
      }
      if (not equal<Dim::_3D>(hij_predict, hij_expect, "hij", acc)) {
        printf("hij: %.12e %.12e %.12e : %.12e %.12e %.12e\n",
               hij_predict[0],
               hij_predict[1],
               hij_predict[2],
               hij_expect[0],
               hij_expect[1],
               hij_expect[2]);
        Kokkos::abort("hij");
      }
      if (not equal<Dim::_1D>(h13_predict, { h13_expect }, "h13", acc)) {
        printf("h13: %.12e : %.12e\n", h13_predict[0], h13_expect);
        Kokkos::abort("h13");
      }
    });
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    using namespace metric;
    testMetric<KerrSchild<Dim::_2D>>(
      {
        64,
        54
    },
      { { 0.8, 50.0 }, { 0.0, constant::PI } },
      10,
      { { "a", (real_t)0.95 } });

    testMetric<QKerrSchild<Dim::_2D>>(
      {
        32,
        42
    },
      { { 0.8, 10.0 }, { 0.0, constant::PI } },
      10,
      { { "r0", -TWO }, { "h", ZERO }, { "a", (real_t)0.8 } });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
