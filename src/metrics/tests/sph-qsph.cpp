#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/qspherical.h"
#include "metrics/spherical.h"

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
                  const real_t    acc = ONE) -> bool {
  const auto eps = epsilon * acc;
  for (unsigned short d = 0; d < D; ++d) {
    if (not cmp::AlmostEqual(a[d], b[d], eps)) {
      printf("%d : %.12e != %.12e %s\n", d, a[d], b[d], msg);
      return false;
    }
  }
  return true;
}

template <Dimension D>
Inline void unravel(std::size_t                    idx,
                    tuple_t<std::size_t, D>&       ijk,
                    const tuple_t<std::size_t, D>& res) {
  for (unsigned short d = 0; d < D; ++d) {
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

  unsigned long all_wrongs = 0;
  Kokkos::parallel_reduce(
    "h_ij",
    npts,
    Lambda(index_t n, unsigned long& wrongs) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x_Code { ZERO };
      coord_t<M::Dim> x_Phys { ZERO };

      for (unsigned short d = 0; d < M::Dim; ++d) {
        x_Code[d] = (real_t)(idx[d]) + HALF;
      }

      const auto h_11 = metric.template h_<1, 1>(x_Code);
      const auto h_22 = metric.template h_<2, 2>(x_Code);
      const auto h_33 = metric.template h_<3, 3>(x_Code);

      metric.template convert<Crd::Cd, Crd::Ph>(x_Code, x_Phys);
      const auto r  = x_Phys[0];
      const auto th = x_Phys[1];

      const auto      h_11_expect = ONE;
      const auto      h_22_expect = SQR(r);
      const auto      h_33_expect = SQR(r * math::sin(th));
      vec_t<Dim::_3D> h_ij_expect { h_11_expect, h_22_expect, h_33_expect };

      vec_t<Dim::_3D> h_ij_temp { ZERO }, h_ij_predict { ZERO };
      metric.template transform<Idx::D, Idx::PD>(x_Code,
                                                 { h_11, h_22, h_33 },
                                                 h_ij_temp);
      metric.template transform<Idx::D, Idx::PD>(x_Code, h_ij_temp, h_ij_predict);

      wrongs += not equal<Dim::_3D>(h_ij_predict, h_ij_expect, "h_ij", acc);
    },
    all_wrongs);

  errorIf(all_wrongs != 0,
          "wrong h_ij for " + std::to_string(M::Dim) + "D " +
            std::string(metric.Label) + " with " + std::to_string(all_wrongs) +
            " errors");
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    using namespace metric;
    const auto res = std::vector<std::size_t> { 64, 32 };
    const auto ext = boundaries_t<real_t> {
      {1.0,         10.0},
      {0.0, constant::PI}
    };
    const auto params = std::map<std::string, real_t> {
      {"r0",         -ONE},
      { "h", (real_t)0.25}
    };

    testMetric<Spherical<Dim::_2D>>(res, ext, 10);
    testMetric<QSpherical<Dim::_2D>>(res, ext, 10, params);

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}