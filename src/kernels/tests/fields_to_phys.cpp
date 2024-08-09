#include "kernels/fields_to_phys.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/numeric.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

using namespace ntt;
using namespace metric;

template <typename M>
void testFlds2Phys(const std::vector<std::size_t>&      res,
                   const boundaries_t<real_t>&          ext,
                   const std::map<std::string, real_t>& params = {}) {
  static_assert(M::Dim == 2);
  errorIf(res.size() != M::Dim, "res.size() != M::Dim");

  boundaries_t<real_t> extent;
  if constexpr (M::CoordType == Coord::Cart) {
    extent = ext;
  } else {
    extent = {
      ext[0],
      {ZERO, constant::PI}
    };
  }

  M metric { res, extent, params };

  const auto nx1 = res[0];
  const auto nx2 = res[1];

  ndfield_t<M::Dim, 3> from { "from", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS, 3 };
  ndfield_t<M::Dim, 6> to { "to", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS, 6 };

  auto from_h     = Kokkos::create_mirror_view(from);
  from_h(5, 5, 0) = 1.0;
  from_h(5, 5, 1) = 2.0;
  from_h(5, 5, 2) = 3.0;
  Kokkos::deep_copy(from, from_h);

  list_t<unsigned short, 3> comp_from = { 0, 1, 2 };
  list_t<unsigned short, 3> comp_to1  = { 3, 4, 5 };
  list_t<unsigned short, 3> comp_to2  = { 0, 1, 2 };

  Kokkos::parallel_for(
    "InterpFields",
    CreateRangePolicy<M::Dim>({ N_GHOSTS, N_GHOSTS },
                              { nx1 + N_GHOSTS, nx2 + N_GHOSTS }),
    kernel::FieldsToPhys_kernel<M, 3, 6>(from,
                                         to,
                                         comp_from,
                                         comp_to1,
                                         PrepareOutput::InterpToCellCenterFromEdges,
                                         metric));
  auto to_h = Kokkos::create_mirror_view(to);
  Kokkos::deep_copy(to_h, to);

  errorIf(to_h(3 + N_GHOSTS, 2 + N_GHOSTS, 3) != HALF,
          "wrong interpolation from edges");
  errorIf(to_h(3 + N_GHOSTS, 3 + N_GHOSTS, 3) != HALF,
          "wrong interpolation from edges");
  errorIf(to_h(2 + N_GHOSTS, 3 + N_GHOSTS, 4) != ONE,
          "wrong interpolation from edges");
  errorIf(to_h(3 + N_GHOSTS, 3 + N_GHOSTS, 4) != ONE,
          "wrong interpolation from edges");
  errorIf(to_h(2 + N_GHOSTS, 2 + N_GHOSTS, 5) != (real_t)(0.75),
          "wrong interpolation from edges");
  errorIf(to_h(2 + N_GHOSTS, 3 + N_GHOSTS, 5) != (real_t)(0.75),
          "wrong interpolation from edges");
  errorIf(to_h(3 + N_GHOSTS, 2 + N_GHOSTS, 5) != (real_t)(0.75),
          "wrong interpolation from edges");
  errorIf(to_h(3 + N_GHOSTS, 3 + N_GHOSTS, 5) != (real_t)(0.75),
          "wrong interpolation from edges");

  Kokkos::parallel_for(
    "InterpFields",
    CreateRangePolicy<M::Dim>({ N_GHOSTS, N_GHOSTS },
                              { nx1 + N_GHOSTS, nx2 + N_GHOSTS }),
    kernel::FieldsToPhys_kernel<M, 3, 6>(from,
                                         to,
                                         comp_from,
                                         comp_to2,
                                         PrepareOutput::InterpToCellCenterFromFaces,
                                         metric));
  Kokkos::deep_copy(to_h, to);
  errorIf(to_h(2 + N_GHOSTS, 3 + N_GHOSTS, 0) != HALF,
          "wrong interpolation from faces");
  errorIf(to_h(3 + N_GHOSTS, 3 + N_GHOSTS, 0) != HALF,
          "wrong interpolation from faces");
  errorIf(to_h(3 + N_GHOSTS, 2 + N_GHOSTS, 1) != ONE,
          "wrong interpolation from faces");
  errorIf(to_h(3 + N_GHOSTS, 3 + N_GHOSTS, 1) != ONE,
          "wrong interpolation from faces");
  errorIf(to_h(3 + N_GHOSTS, 3 + N_GHOSTS, 2) != THREE,
          "wrong interpolation from faces");
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testFlds2Phys<Minkowski<Dim::_2D>>(
      {
        10,
        10
    },
      { { 0.0, 10.0 }, { 0.0, 10.0 } },
      {});

    testFlds2Phys<Spherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 2.0 } },
      {});

    testFlds2Phys<QSpherical<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 10.0 } },
      { { "r0", 0.0 }, { "h", 0.25 } });

    testFlds2Phys<KerrSchild<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "a", 0.9 } });

    testFlds2Phys<QKerrSchild<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "r0", 0.0 }, { "h", 0.25 }, { "a", 0.9 } });

    testFlds2Phys<KerrSchild0<Dim::_2D>>(
      {
        10,
        10
    },
      { { 1.0, 100.0 } },
      { { "a", 0.9 } });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}