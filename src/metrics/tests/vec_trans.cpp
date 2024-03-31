#include "global.h"
// metrics >
#include "kerr_schild.h"
#include "minkowski.h"
#include "qkerr_schild.h"
#include "qspherical.h"
#include "spherical.h"

#include "kerr_schild_0.h"
// < metrics

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

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

template <Dimension D>
Inline auto equal(const coord_t<D>& a,
                  const coord_t<D>& b,
                  const char*       msg,
                  const real_t      acc = ONE) -> bool {
  const auto eps = std::numeric_limits<real_t>::epsilon() * acc;
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
void testMetric(const std::vector<unsigned int>&              res,
                const std::vector<std::pair<real_t, real_t>>& ext,
                const real_t                                  acc    = ONE,
                const std::map<std::string, real_t>&          params = {}) {
  errorIf(res.size() != (std::size_t)(M::Dim), "res.size() != M.dim");
  errorIf(ext.size() != (std::size_t)(M::Dim), "ext.size() != M.dim");
  for (const auto& e : ext) {
    errorIf(e.first >= e.second, "e.first >= e.second");
  }
  using namespace std;

  M metric(res, ext, params);

  tuple_t<std::size_t, M::Dim> res_tup;
  std::size_t                  npts = 1;
  for (auto d = 0; d < M::Dim; ++d) {
    res_tup[d]  = res[d];
    npts       *= res[d];
  }

  // !TODO: had to reduce accuracy on some of the tests
  unsigned long all_wrongs = 0;
  Kokkos::parallel_reduce(
    "hat-cntrv-cov",
    npts,
    Lambda(index_t n, unsigned long& wrongs) {
      tuple_t<std::size_t, M::Dim> idx;
      unravel<M::Dim>(n, idx, res_tup);
      coord_t<M::Dim> x_Code { ZERO };
      for (unsigned short d = 0; d < M::Dim; ++d) {
        x_Code[d] = (real_t)(idx[d]) + HALF;
      }
      vec_t<Dim::_3D> v_Hat_1 { ZERO };
      vec_t<Dim::_3D> v_Hat_2 { ZERO };
      vec_t<Dim::_3D> v_Cntrv_1 { ZERO };
      vec_t<Dim::_3D> v_Cntrv_2 { ZERO };
      vec_t<Dim::_3D> v_Cov_1 { ZERO };
      vec_t<Dim::_3D> v_Cov_2 { ZERO };
      vec_t<Dim::_3D> v_PhysCntrv_1 { ZERO };
      vec_t<Dim::_3D> v_PhysCntrv_2 { ZERO };
      vec_t<Dim::_3D> v_PhysCov_1 { ZERO };
      vec_t<Dim::_3D> v_PhysCov_2 { ZERO };

      // init
      for (unsigned short d = 0; d < Dim::_3D; ++d) {
        v_Hat_1[d]       += ONE;
        v_PhysCntrv_1[d] += ONE;
        v_PhysCov_1[d]   += ONE;
      }

      // hat <-> cntrv
      metric.v3_Hat2Cntrv(x_Code, v_Hat_1, v_Cntrv_1);
      for (unsigned short d = 0; d < Dim::_3D; ++d) {
        vec_t<Dim::_3D> e_d { ZERO };
        vec_t<Dim::_3D> v_Cntrv_temp { ZERO };
        e_d[d] = ONE;
        metric.v3_Hat2Cntrv(x_Code, e_d, v_Cntrv_temp);
        for (unsigned short d = 0; d < Dim::_3D; ++d) {
          v_Cntrv_2[d] += v_Cntrv_temp[d];
        }
      }
      wrongs += not equal<Dim::_3D>(v_Cntrv_1, v_Cntrv_2, "hat->cntrv is linear", acc);

      metric.v3_Cntrv2Hat(x_Code, v_Cntrv_1, v_Hat_2);
      wrongs += not equal<Dim::_3D>(v_Hat_1, v_Hat_2, "hat->cntrv is invertible", acc);

      // cntrv <-> cov & hat <-> cov
      metric.v3_Cntrv2Cov(x_Code, v_Cntrv_1, v_Cov_1);
      metric.v3_Hat2Cov(x_Code, v_Hat_1, v_Cov_2);
      wrongs += not equal<Dim::_3D>(v_Cov_1,
                                    v_Cov_2,
                                    "cntrv->cov is equal to hat->cov",
                                    acc);
      for (unsigned short d = 0; d < Dim::_3D; ++d) {
        v_Cov_2[d] = ZERO;
      }
      for (unsigned short d = 0; d < Dim::_3D; ++d) {
        vec_t<Dim::_3D> e_d { ZERO };
        vec_t<Dim::_3D> v_Cov_temp { ZERO };
        e_d[d] = ONE;
        metric.v3_Hat2Cov(x_Code, e_d, v_Cov_temp);
        for (unsigned short d = 0; d < Dim::_3D; ++d) {
          v_Cov_2[d] += v_Cov_temp[d];
        }
      }
      wrongs += not equal<Dim::_3D>(v_Cov_1, v_Cov_2, "hat->cov is linear", acc);

      metric.v3_Cov2Cntrv(x_Code, v_Cov_1, v_Cntrv_2);
      wrongs += not equal<Dim::_3D>(v_Cntrv_1,
                                    v_Cntrv_2,
                                    "cntrv->cov is invertible",
                                    acc);

      metric.v3_Cov2Hat(x_Code, v_Cov_1, v_Hat_2);
      wrongs += not equal<Dim::_3D>(v_Hat_1, v_Hat_2, "hat->cov is invertible", acc);

      // phys <-> cntrv & phys <-> cov
      metric.v3_PhysCntrv2Cntrv(x_Code, v_PhysCntrv_1, v_Cntrv_1);
      metric.v3_Cntrv2PhysCntrv(x_Code, v_Cntrv_1, v_PhysCntrv_2);
      wrongs += not equal<Dim::_3D>(v_PhysCntrv_1,
                                    v_PhysCntrv_2,
                                    "phys->cntrv is invertible",
                                    acc);

      metric.v3_PhysCov2Cov(x_Code, v_PhysCov_1, v_Cov_1);
      metric.v3_Cov2PhysCov(x_Code, v_Cov_1, v_PhysCov_2);
      wrongs += not equal<Dim::_3D>(v_PhysCov_1,
                                    v_PhysCov_2,
                                    "phys->cov is invertible",
                                    acc);
    },
    all_wrongs);

  errorIf(all_wrongs != 0,
          "hat-cntrv-cov for " + metric.label + " failed with " +
            std::to_string(all_wrongs) + " errors");
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testMetric<Minkowski<Dim::_1D>>(
      {
        128,
    },
      { { 10.0, 20.0 } });

    testMetric<Minkowski<Dim::_2D>>(
      {
        64,
        32
    },
      { { 0.0, 20.0 }, { 0.0, 10.0 } });

    testMetric<Minkowski<Dim::_3D>>(
      {
        64,
        32,
        16
    },
      { { -2.0, 2.0 }, { -1.0, 1.0 }, { -0.5, 0.5 } });

    testMetric<Spherical<Dim::_2D>>(
      {
        64,
        32
    },
      { { 1.0, 10.0 }, { 0.0, constant::PI } },
      10);

    testMetric<QSpherical<Dim::_2D>>(
      {
        64,
        32
    },
      { { 1.0, 10.0 }, { 0.0, constant::PI } },
      10,
      { { "r0", -ONE }, { "h", (real_t)0.25 } });

    testMetric<KerrSchild<Dim::_2D>>(
      {
        64,
        54
    },
      { { 0.8, 50.0 }, { 0.0, constant::PI } },
      150,
      { { "a", (real_t)0.95 } });

    testMetric<QKerrSchild<Dim::_2D>>(
      {
        64,
        42
    },
      { { 0.8, 10.0 }, { 0.0, constant::PI } },
      300,
      { { "r0", -TWO }, { "h", ZERO }, { "a", (real_t)0.8 } });

    testMetric<KerrSchild0<Dim::_2D>>(
      {
        64,
        54
    },
      { { 0.5, 20.0 }, { 0.0, constant::PI } },
      10);

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
