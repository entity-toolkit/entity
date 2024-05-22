#include "framework/domain/comm_mpi.hpp"

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"

#include <iostream>
#include <stdexcept>

using namespace ntt;

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    const std::size_t nx1 = 15, nx2 = 15;
    ndfield_t<Dim::_2D, 3> fld_b1 { "fld", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
    ndfield_t<Dim::_2D, 3> fld_b2 { "fld", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };

    Kokkos::parallel_for(
      "Fill",
      CreateRangePolicy<Dim::_2D>({ 0, 0 },
                                  { nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS }),
      Lambda(index_t i1, index_t i2) {
        if ((i1 >= 2 * N_GHOSTS) and (i1 < nx1) and (i2 >= 2 * N_GHOSTS) and
            (i2 < nx2)) {
          fld_b1(i1, i2, 0) = 4.0;
          fld_b1(i1, i2, 1) = 12.0;
          fld_b1(i1, i2, 2) = 20.0;
          fld_b2(i1, i2, 0) = 4.0;
          fld_b2(i1, i2, 1) = 12.0;
          fld_b2(i1, i2, 2) = 20.0;
        } else if (
          ((i1 < 2 * N_GHOSTS or i1 >= nx1) and (i2 >= 2 * N_GHOSTS and i2 < nx2)) or
          ((i2 < 2 * N_GHOSTS or i2 >= nx2) and (i1 >= 2 * N_GHOSTS and i1 < nx1))) {
          fld_b1(i1, i2, 0) = 2.0;
          fld_b1(i1, i2, 1) = 6.0;
          fld_b1(i1, i2, 2) = 10.0;
          fld_b2(i1, i2, 0) = 2.0;
          fld_b2(i1, i2, 1) = 6.0;
          fld_b2(i1, i2, 2) = 10.0;
        } else {
          fld_b1(i1, i2, 0) = 1.0;
          fld_b1(i1, i2, 1) = 3.0;
          fld_b1(i1, i2, 2) = 5.0;
          fld_b2(i1, i2, 0) = 1.0;
          fld_b2(i1, i2, 1) = 3.0;
          fld_b2(i1, i2, 2) = 5.0;
        }
      });
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }

  Kokkos::finalize();
  return 0;
}