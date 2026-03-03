#include "framework/domain/comm_mpi.hpp"

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <iostream>
#include <stdexcept>

using namespace ntt;

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  try {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const ncells_t nx1 = 11, nx2 = 15;
    ndfield_t<Dim::_2D, 3> fld { "fld", nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS };

    Kokkos::parallel_for(
      "Fill",
      CreateRangePolicy<Dim::_2D>({ 0, 0 },
                                  { nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS }),
      Lambda(index_t i1, index_t i2) {
        if ((i1 >= N_GHOSTS) and (i1 < N_GHOSTS + nx1) and (i2 >= N_GHOSTS) and
            (i2 < N_GHOSTS + nx2)) {
          fld(i1, i2, 0) = static_cast<real_t>(rank + 1) + 4.0;
          fld(i1, i2, 1) = static_cast<real_t>(rank + 1) + 12.0;
          fld(i1, i2, 2) = static_cast<real_t>(rank + 1) + 20.0;
        }
      });

    {
      // send right, recv left
      const int          send_idx  = (rank + 1) % size;
      const int          recv_idx  = (rank - 1 + size) % size;
      const unsigned int send_rank = (unsigned int)send_idx;
      const unsigned int recv_rank = (unsigned int)recv_idx;

      const std::vector<range_tuple_t> send_slice {
        {      nx1, nx1 + N_GHOSTS },
        { N_GHOSTS, nx2 + N_GHOSTS }
      };
      const std::vector<range_tuple_t> recv_slice {
        {        0,       N_GHOSTS },
        { N_GHOSTS, nx2 + N_GHOSTS }
      };
      const range_tuple_t comp_slice { 0, 3 };
      comm::CommunicateField<Dim::_2D, 3>((unsigned int)(rank),
                                          fld,
                                          fld,
                                          send_idx,
                                          recv_idx,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          comp_slice,
                                          false);
    }
    {
      // recv right, send left
      const int          send_idx  = (rank - 1 + size) % size;
      const int          recv_idx  = (rank + 1) % size;
      const unsigned int send_rank = (unsigned int)send_idx;
      const unsigned int recv_rank = (unsigned int)recv_idx;

      const std::vector<range_tuple_t> send_slice {
        { N_GHOSTS,   N_GHOSTS + 2 },
        { N_GHOSTS, nx2 + N_GHOSTS }
      };
      const std::vector<range_tuple_t> recv_slice {
        { nx1 + N_GHOSTS, nx1 + 2 * N_GHOSTS },
        {       N_GHOSTS,     nx2 + N_GHOSTS }
      };
      const range_tuple_t comp_slice { 0, 3 };
      comm::CommunicateField<Dim::_2D, 3>((unsigned int)(rank),
                                          fld,
                                          fld,
                                          send_idx,
                                          recv_idx,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          comp_slice,
                                          false);
    }

    {
      const auto left_expect = static_cast<real_t>((rank - 1 + size) % size + 1);
      const auto right_expect = static_cast<real_t>((rank + 1) % size + 1);

      Kokkos::parallel_for(
        "Check",
        CreateRangePolicy<Dim::_1D>({ N_GHOSTS }, { nx2 + N_GHOSTS }),
        Lambda(index_t i2) {
          for (auto i1 { 0u }; i1 < N_GHOSTS; ++i1) {
            if (fld(i1, i2, 0) != left_expect + 4.0) {
              raise::KernelError(HERE, "Left boundary not correct for #0");
            }
            if (fld(i1, i2, 1) != left_expect + 12.0) {
              raise::KernelError(HERE, "Left boundary not correct for #1");
            }
            if (fld(i1, i2, 2) != left_expect + 20.0) {
              raise::KernelError(HERE, "Left boundary not correct for #2");
            }
          }
          for (auto i1 { nx1 + N_GHOSTS }; i1 < nx1 + 2 * N_GHOSTS; ++i1) {
            if (fld(i1, i2, 0) != right_expect + 4.0) {
              raise::KernelError(HERE, "Right boundary not correct for #0");
            }
            if (fld(i1, i2, 1) != right_expect + 12.0) {
              raise::KernelError(HERE, "Right boundary not correct for #1");
            }
            if (fld(i1, i2, 2) != right_expect + 20.0) {
              raise::KernelError(HERE, "Right boundary not correct for #2");
            }
          }
        });
    }

    Kokkos::parallel_for(
      "Carve",
      CreateRangePolicy<Dim::_2D>({ 0, 0 },
                                  { nx1 + 2 * N_GHOSTS, nx2 + 2 * N_GHOSTS }),
      Lambda(index_t i1, index_t i2) {
        if (((i1 >= N_GHOSTS) and (i1 < 2 * N_GHOSTS)) or
            ((i1 >= nx1) and (i1 < nx1 + N_GHOSTS))) {
          fld(i1, i2, 0) = ZERO;
          fld(i1, i2, 1) = ZERO;
          fld(i1, i2, 2) = ZERO;
        }
      });

    {
      // send right, recv left
      const int          send_idx  = (rank + 1) % size;
      const int          recv_idx  = (rank - 1 + size) % size;
      const unsigned int send_rank = (unsigned int)send_idx;
      const unsigned int recv_rank = (unsigned int)recv_idx;

      const std::vector<range_tuple_t> send_slice {
        { nx1 + N_GHOSTS, nx1 + 2 * N_GHOSTS },
        {       N_GHOSTS,     nx2 + N_GHOSTS }
      };
      const std::vector<range_tuple_t> recv_slice {
        { N_GHOSTS,   2 * N_GHOSTS },
        { N_GHOSTS, nx2 + N_GHOSTS }
      };
      const range_tuple_t comp_slice { 0, 3 };
      comm::CommunicateField<Dim::_2D, 3>((unsigned int)(rank),
                                          fld,
                                          fld,
                                          send_idx,
                                          recv_idx,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          comp_slice,
                                          true);
    }
    {
      // recv right, send left
      const int          send_idx  = (rank - 1 + size) % size;
      const int          recv_idx  = (rank + 1) % size;
      const unsigned int send_rank = (unsigned int)send_idx;
      const unsigned int recv_rank = (unsigned int)recv_idx;

      const std::vector<range_tuple_t> send_slice {
        {        0,       N_GHOSTS },
        { N_GHOSTS, nx2 + N_GHOSTS }
      };
      const std::vector<range_tuple_t> recv_slice {
        {      nx1, nx1 + N_GHOSTS },
        { N_GHOSTS, nx2 + N_GHOSTS }
      };
      const range_tuple_t comp_slice { 0, 3 };
      comm::CommunicateField<Dim::_2D, 3>((unsigned int)(rank),
                                          fld,
                                          fld,
                                          send_idx,
                                          recv_idx,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          comp_slice,
                                          true);
    }

    {
      const auto expect = static_cast<real_t>(rank + 1);
      Kokkos::parallel_for(
        "Check",
        CreateRangePolicy<Dim::_1D>({ N_GHOSTS }, { nx2 + N_GHOSTS }),
        Lambda(index_t i2) {
          for (auto i1 { N_GHOSTS }; i1 < 2 * N_GHOSTS; ++i1) {
            if (fld(i1, i2, 0) != expect + 4.0) {
              raise::KernelError(HERE, "Left boundary not correct for #0");
            }
            if (fld(i1, i2, 1) != expect + 12.0) {
              raise::KernelError(HERE, "Left boundary not correct for #1");
            }
            if (fld(i1, i2, 2) != expect + 20.0) {
              raise::KernelError(HERE, "Left boundary not correct for #2");
            }
          }
          for (auto i1 { nx1 }; i1 < nx1 + N_GHOSTS; ++i1) {
            if (fld(i1, i2, 0) != expect + 4.0) {
              raise::KernelError(HERE, "Right boundary not correct for #0");
            }
            if (fld(i1, i2, 1) != expect + 12.0) {
              raise::KernelError(HERE, "Right boundary not correct for #1");
            }
            if (fld(i1, i2, 2) != expect + 20.0) {
              raise::KernelError(HERE, "Right boundary not correct for #2");
            }
          }
        });
    }
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    MPI_Finalize();
    Kokkos::finalize();
    return 1;
  }

  MPI_Finalize();
  Kokkos::finalize();
  return 0;
}
