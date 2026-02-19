#include "framework/domain/comm_nompi.hpp"

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include <iostream>
#include <stdexcept>

using namespace ntt;

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    const std::size_t nx1 = 15, nx2 = 15;
    ndfield_t<Dim::_2D, 3> fld { "fld", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };
    ndfield_t<Dim::_2D, 3> buff { "buff", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };

    Kokkos::parallel_for(
      "Fill",
      CreateRangePolicy<Dim::_2D>({ 0, 0 },
                                  { nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS }),
      Lambda(index_t i1, index_t i2) {
        if ((i1 >= 2 * N_GHOSTS) and (i1 < nx1) and (i2 >= 2 * N_GHOSTS) and
            (i2 < nx2)) {
          fld(i1, i2, 0) = 4.0;
          fld(i1, i2, 1) = 12.0;
          fld(i1, i2, 2) = 20.0;
        } else if (
          ((i1 < 2 * N_GHOSTS or i1 >= nx1) and (i2 >= 2 * N_GHOSTS and i2 < nx2)) or
          ((i2 < 2 * N_GHOSTS or i2 >= nx2) and (i1 >= 2 * N_GHOSTS and i1 < nx1))) {
          fld(i1, i2, 0) = 2.0;
          fld(i1, i2, 1) = 6.0;
          fld(i1, i2, 2) = 10.0;
        } else {
          fld(i1, i2, 0) = 1.0;
          fld(i1, i2, 1) = 3.0;
          fld(i1, i2, 2) = 5.0;
        }
      });
    Kokkos::deep_copy(buff, ZERO);

    const auto send_slice = std::vector<range_tuple_t> {
      { nx1 + N_GHOSTS, nx1 + 2 * N_GHOSTS },
      { nx2 + N_GHOSTS, nx2 + 2 * N_GHOSTS }
    };
    const auto recv_slice = std::vector<range_tuple_t> {
      { N_GHOSTS, 2 * N_GHOSTS },
      { N_GHOSTS, 2 * N_GHOSTS }
    };
    const auto comp_slice = range_tuple_t(cur::jx1, cur::jx3 + 1);

    const auto i_min = [](in c) {
      switch (c) {
        case in::x1:
          return (std::size_t)N_GHOSTS;
        case in::x2:
          return (std::size_t)N_GHOSTS;
        case in::x3:
          return (std::size_t)N_GHOSTS;
      }
      return (std::size_t)0;
    };
    const auto i_max = [](in c) {
      switch (c) {
        case in::x1:
          return nx1 + N_GHOSTS;
        case in::x2:
          return nx2 + N_GHOSTS;
        case in::x3:
          return (std::size_t)0;
      }
      return (std::size_t)0;
    };
    const in components[] = { in::x1, in::x2, in::x3 };
    for (auto& direction : dir::Directions<Dim::_2D>::all) {
      auto send_slice = std::vector<range_tuple_t> {};
      auto recv_slice = std::vector<range_tuple_t> {};
      for (std::size_t d { 0 }; d < direction.size(); ++d) {
        const auto c   = components[d];
        const auto dir = direction[d];
        if (dir == 0) {
          send_slice.emplace_back(i_min(c) - N_GHOSTS, i_max(c) + N_GHOSTS);
        } else if (dir == 1) {
          send_slice.emplace_back(i_max(c) - N_GHOSTS, i_max(c) + N_GHOSTS);
        } else {
          send_slice.emplace_back(i_min(c) - N_GHOSTS, i_min(c) + N_GHOSTS);
        }
        if (-dir == 0) {
          recv_slice.emplace_back(i_min(c) - N_GHOSTS, i_max(c) + N_GHOSTS);
        } else if (-dir == 1) {
          recv_slice.emplace_back(i_max(c) - N_GHOSTS, i_max(c) + N_GHOSTS);
        } else {
          recv_slice.emplace_back(i_min(c) - N_GHOSTS, i_min(c) + N_GHOSTS);
        }
      }
      comm::CommunicateField<Dim::_2D, 3>((unsigned int)0,
                                          fld,
                                          buff,
                                          0,
                                          0,
                                          0,
                                          0,
                                          send_slice,
                                          recv_slice,
                                          comp_slice,
                                          true);
    }
    // add buffers
    Kokkos::parallel_for(
      "Fill",
      CreateRangePolicy<Dim::_2D>({ 0, 0 },
                                  { nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS }),
      Lambda(index_t i1, index_t i2) {
        for (auto k { 0 }; k < 3; ++k) {
          fld(i1, i2, k) += buff(i1, i2, k);
        }
      });

    // check
    Kokkos::parallel_for(
      "Fill",
      CreateRangePolicy<Dim::_2D>({ 0, 0 },
                                  { nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS }),
      Lambda(index_t i1, index_t i2) {
        if (fld(i1, i2, 0) != 4.0) {
          raise::KernelError(HERE, "fld0 wrong after comm");
        }
        if (fld(i1, i2, 1) != 12.0) {
          raise::KernelError(HERE, "fld1 wrong after comm");
        }
        if (fld(i1, i2, 2) != 20.0) {
          raise::KernelError(HERE, "fld2 wrong after comm");
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
