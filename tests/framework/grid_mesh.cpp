#include "global.h"

#include "utils/comparators.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "framework/domain/mesh.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    using namespace ntt;
    using namespace metric;
    const auto res = std::vector<std::size_t> { 10, 10, 10 };
    const auto ext = boundaries_t<real_t> {
      { -1.0, 1.0 },
      { -1.0, 1.0 },
      { -1.0, 1.0 }
    };
    auto mesh = Mesh<Minkowski<Dim::_3D>>(res, ext, {});
    for (const auto& d : { in::x1, in::x2, in::x3 }) {
      raise::ErrorIf(mesh.i_min(d) != N_GHOSTS, "i_min != N_GHOSTS", HERE);
      raise::ErrorIf(mesh.i_max(d) != res[(dim_t)d] + N_GHOSTS,
                     "i_max != res+N_GHOSTS",
                     HERE);
      raise::ErrorIf(mesh.n_active(d) != res[(dim_t)d], "n_active != res", HERE);
      raise::ErrorIf(mesh.n_all(d) != res[(dim_t)d] + 2 * N_GHOSTS,
                     "n_all != res+2*N_GHOSTS",
                     HERE);
      raise::ErrorIf(mesh.extent(d) != ext[(dim_t)d], "extent != ext", HERE);
    }
    raise::ErrorIf(
      not cmp::AlmostEqual(mesh.metric.dxMin(), (real_t)(0.2 / std::sqrt(3.0))),
      "dxMin wrong",
      HERE);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();
  return 0;
}