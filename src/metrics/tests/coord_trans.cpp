#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

template <Dimension D>
Inline auto equal(const coord_t<D>& a,
                  const coord_t<D>& b,
                  const char*       msg,
                  real_t            acc = ONE) -> bool {
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
  errorIf(res.size() != (std::size_t)(M::Dim), "res.size() != M.dim");
  errorIf(ext.size() != (std::size_t)(M::Dim), "ext.size() != M.dim");
  for (const auto& e : ext) {
    errorIf(e.first >= e.second, "e.first >= e.second");
  }

  M metric(res, ext, params);

  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }

  // !ACC: had to reduce accuracy on some of the tests
  unsigned long all_wrongs = 0;
  Kokkos::parallel_reduce(
    "code-sph-phys",
    npts,
    Lambda(index_t n, unsigned long& wrongs) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x_Code_1 { ZERO };
      coord_t<M::Dim> x_Code_2 { ZERO };
      coord_t<M::Dim> x_Phys_1 { ZERO };
      coord_t<M::Dim> x_Sph_1 { ZERO };
      for (unsigned short d = 0; d < M::Dim; ++d) {
        x_Code_1[d] = (real_t)(idx[d]) + HALF;
      }
      metric.template convert<Crd::Cd, Crd::Ph>(x_Code_1, x_Phys_1);
      metric.template convert<Crd::Ph, Crd::Cd>(x_Phys_1, x_Code_2);
      wrongs += not equal<M::Dim>(x_Code_1, x_Code_2, "code->phys not invertible", acc);
      if constexpr (M::Dim != Dim::_1D) {
        metric.template convert<Crd::Cd, Crd::Sph>(x_Code_1, x_Sph_1);
        metric.template convert<Crd::Sph, Crd::Cd>(x_Sph_1, x_Code_2);
        wrongs += not equal<M::Dim>(x_Code_1,
                                    x_Code_2,
                                    "code->sph not invertible",
                                    acc);
      }
      // 1D/2D/3D
      wrongs += (x_Phys_1[0] >= metric.x1_max);
      wrongs += (x_Phys_1[0] < metric.x1_min);
      // 2D/3D
      if constexpr (M::Dim == Dim::_2D || M::Dim == Dim::_3D) {
        wrongs += (x_Phys_1[1] >= metric.x2_max);
        wrongs += (x_Phys_1[1] < metric.x2_min);
      }
      // 3D
      if constexpr (M::Dim == Dim::_3D) {
        wrongs += (x_Phys_1[2] >= metric.x3_max);
        wrongs += (x_Phys_1[2] < metric.x3_min);
      }
    },
    all_wrongs);

  errorIf(all_wrongs != 0,
          "code-sph-phys for " + std::to_string(M::Dim) + "D " +
            std::string(metric.Label) + " failed with " +
            std::to_string(all_wrongs) + " errors");
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    using namespace metric;
    const auto res2d     = std::vector<std::size_t> { 64, 32 };
    const auto res3d     = std::vector<std::size_t> { 64, 32, 16 };
    const auto ext1dcart = boundaries_t<real_t> {
      {10.0, 20.0}
    };
    const auto ext2dcart = boundaries_t<real_t> {
      {0.0, 20.0},
      {0.0, 10.0}
    };
    const auto ext3dcart = boundaries_t<real_t> {
      {-2.0, 2.0},
      {-1.0, 1.0},
      {-0.5, 0.5}
    };
    const auto extsph = boundaries_t<real_t> {
      {1.0,         10.0},
      {0.0, constant::PI}
    };
    const auto params = std::map<std::string, real_t> {
      {"r0",         -ONE},
      { "h", (real_t)0.25}
    };

    testMetric<Minkowski<Dim::_1D>>({ 128 }, ext1dcart);
    testMetric<Minkowski<Dim::_2D>>(res2d, ext2dcart, 200);
    testMetric<Minkowski<Dim::_3D>>(res3d, ext3dcart, 500);
    testMetric<Spherical<Dim::_2D>>(res2d, extsph, 10);
    testMetric<QSpherical<Dim::_2D>>(res2d, extsph, 100, params);

    const auto resks  = std::vector<std::size_t> { 64, 54 };
    const auto extsks = boundaries_t<real_t> {
      {0.8,         50.0},
      {0.0, constant::PI}
    };
    const auto paramsks = std::map<std::string, real_t> {
      {"a", (real_t)0.95}
    };
    testMetric<KerrSchild<Dim::_2D>>(resks, extsks, 150, paramsks);

    const auto resqks = std::vector<std::size_t> { 64, 42 };
    const auto extqks = boundaries_t<real_t> {
      {0.8,         10.0},
      {0.0, constant::PI}
    };
    const auto paramsqks = std::map<std::string, real_t> {
      {"r0",        -TWO},
      { "h",        ZERO},
      { "a", (real_t)0.8}
    };
    testMetric<QKerrSchild<Dim::_2D>>(resqks, extqks, 500, paramsqks);

    const auto resks0 = std::vector<std::size_t> { 64, 54 };
    const auto extks0 = boundaries_t<real_t> {
      {0.5,         20.0},
      {0.0, constant::PI}
    };
    testMetric<KerrSchild0<Dim::_2D>>(resks0, extks0, 150);

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
