#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#define MPI_ROOT_RANK 0
#define N_GHOSTS      2

template <typename Func, typename... Args>
void CallOnce(Func, Args&&...);

template <typename T, unsigned short D>
using R = std::conditional_t<
  D == 1,
  T*,
  std::conditional_t<D == 2, T**, std::conditional_t<D == 3, T***, void>>>;

template <typename T, unsigned short D, unsigned short N>
void send_recv(int                             send_to,
               int                             recv_from,
               bool                            sendxmin,
               const Kokkos::View<R<T, D>[N]>& view,
               std::size_t                     smallsize) {
  const auto  mpi_type = std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE;
  std::size_t nsend    = 0;
  Kokkos::View<R<T, D>[N]> send_buffer;
  if (send_to == MPI_PROC_NULL) {
    nsend = 0;
  } else {
    std::pair<std::size_t, std::size_t> range = { 0, N_GHOSTS };
    if (not sendxmin) {
      range = { view.extent(0) - N_GHOSTS, view.extent(0) };
    }
    if constexpr (D == 1) {
      nsend       = N_GHOSTS * N;
      send_buffer = Kokkos::View<R<T, D>[N]> {
        "comm_1d_send_buffer", N_GHOSTS
      };
      Kokkos::deep_copy(send_buffer, Kokkos::subview(view, range, Kokkos::ALL));
    } else if constexpr (D == 2) {
      nsend       = N_GHOSTS * smallsize * N;
      send_buffer = Kokkos::View<R<T, D>[N]> {
        "comm_2d_send_buffer", N_GHOSTS, smallsize
      };
      Kokkos::deep_copy(send_buffer,
                        Kokkos::subview(view, range, Kokkos::ALL, Kokkos::ALL));
    } else if constexpr (D == 3) {
      nsend       = N_GHOSTS * smallsize * smallsize * N;
      send_buffer = Kokkos::View<R<T, D>[N]> {
        "comm_3d_send_buffer", N_GHOSTS, smallsize, smallsize
      };
      Kokkos::deep_copy(
        send_buffer,
        Kokkos::subview(view, range, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    }
  }

  std::size_t              nrecv = 0;
  Kokkos::View<R<T, D>[N]> recv_buffer;
  if (recv_from == MPI_PROC_NULL) {
    nrecv = 0;
  } else {
    if constexpr (D == 1) {
      nrecv       = N_GHOSTS * N;
      recv_buffer = Kokkos::View<R<T, D>[N]> {
        "comm_1d_recv_buffer", N_GHOSTS
      };
    } else if constexpr (D == 2) {
      nrecv       = N_GHOSTS * smallsize * N;
      recv_buffer = Kokkos::View<R<T, D>[N]> {
        "comm_2d_recv_buffer", N_GHOSTS, smallsize
      };
    } else if constexpr (D == 3) {
      nrecv       = N_GHOSTS * smallsize * smallsize * N;
      recv_buffer = Kokkos::View<R<T, D>[N]> {
        "comm_3d_recv_buffer", N_GHOSTS, smallsize, smallsize
      };
    }
  }

  if (nrecv == 0 and nsend == 0) {
    throw std::invalid_argument(
      "Both nsend and nrecv are zero, no communication to perform.");
  } else if (nrecv > 0 and nsend > 0) {
#if defined(GPU_AWARE_MPI) || !defined(DEVICE_ENABLED)
    MPI_Sendrecv(send_buffer.data(),
                 nsend,
                 mpi_type,
                 send_to,
                 0,
                 recv_buffer.data(),
                 nrecv,
                 mpi_type,
                 recv_from,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
#else
    auto send_buffer_h = Kokkos::create_mirror_view(send_buffer);
    auto recv_buffer_h = Kokkos::create_mirror_view(recv_buffer);
    Kokkos::deep_copy(send_buffer_h, send_buffer);
    MPI_Sendrecv(send_buffer_h.data(),
                 nsend,
                 mpi_type,
                 send_to,
                 0,
                 recv_buffer_h.data(),
                 nrecv,
                 mpi_type,
                 recv_from,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    Kokkos::deep_copy(recv_buffer, recv_buffer_h);
#endif
  } else if (nrecv > 0) {
#if defined(GPU_AWARE_MPI) || !defined(DEVICE_ENABLED)
    MPI_Recv(recv_buffer.data(),
             nrecv,
             mpi_type,
             recv_from,
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
#else
    auto recv_buffer_h = Kokkos::create_mirror_view(recv_buffer);
    MPI_Recv(recv_buffer_h.data(),
             nrecv,
             mpi_type,
             recv_from,
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    Kokkos::deep_copy(recv_buffer, recv_buffer_h);
#endif
  } else if (nsend > 0) {
#if defined(GPU_AWARE_MPI) || !defined(DEVICE_ENABLED)
    MPI_Send(send_buffer.data(), nsend, mpi_type, send_to, 0, MPI_COMM_WORLD);
#else
    auto send_buffer_h = Kokkos::create_mirror_view(send_buffer);
    Kokkos::deep_copy(send_buffer_h, send_buffer);
    MPI_Send(send_buffer_h.data(), nsend, mpi_type, send_to, 0, MPI_COMM_WORLD);
#endif
  }

  if (nrecv > 0) {
    std::pair<std::size_t, std::size_t> range = { view.extent(0) - N_GHOSTS,
                                                  view.extent(0) };
    if (not sendxmin) {
      range = { 0, N_GHOSTS };
    }
    if constexpr (D == 1) {
      Kokkos::deep_copy(Kokkos::subview(view, range, Kokkos::ALL), recv_buffer);
    } else if constexpr (D == 2) {
      Kokkos::deep_copy(Kokkos::subview(view, range, Kokkos::ALL, Kokkos::ALL),
                        recv_buffer);
    } else if constexpr (D == 3) {
      Kokkos::deep_copy(
        Kokkos::subview(view, range, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
        recv_buffer);
    }
  }
}

template <typename T, unsigned short D, unsigned short N>
void comm(int rank, int size, std::size_t bigsize, std::size_t smallsize) {
  static_assert(D <= 3 and D != 0, "Only dimensions 1, 2, and 3 are supported.");
  static_assert(N == 3 or N == 6, "Only 3 or 6 last indices are supported.");
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Only float and double types are supported.");

  // smallsize must be the same for all ranks
  if (bigsize < 2 * N_GHOSTS) {
    throw std::invalid_argument(
      "bigsize must be at least 2 * N_GHOSTS for communication to work.");
  }

  Kokkos::View<R<T, D>[N]> view;

  // define and fill the view
  if constexpr (D == 1) {
    view = Kokkos::View<R<T, D>[N]> {
      "comm_1d_view", bigsize
    };
    Kokkos::parallel_for(
      "fill_comm_1d_view",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 },
                                             { view.extent(0), view.extent(1) }),
      KOKKOS_LAMBDA(std::size_t i, std::size_t c) {
        view(i, c) = static_cast<T>(i * c + rank);
      });
  } else if constexpr (D == 2) {
    view = Kokkos::View<R<T, D>[N]> {
      "comm_2d_view", bigsize, smallsize
    };
    Kokkos::parallel_for(
      "fill_comm_2d_view",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        { 0, 0, 0 },
        { view.extent(0), view.extent(1), view.extent(2) }),
      KOKKOS_LAMBDA(std::size_t i, std::size_t j, std::size_t c) {
        view(i, j, c) = static_cast<T>(i * j * c + rank);
      });
  } else if constexpr (D == 3) {
    view = Kokkos::View<R<T, D>[N]> {
      "comm_3d_view", bigsize, smallsize, smallsize
    };
    Kokkos::parallel_for(
      "fill_comm_3d_view",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
        { 0, 0, 0, 0 },
        { view.extent(0), view.extent(1), view.extent(2), view.extent(3) }),
      KOKKOS_LAMBDA(std::size_t i, std::size_t j, std::size_t k, std::size_t c) {
        view(i, j, k, c) = static_cast<T>(i * j * k * c + rank);
      });
  }

  // communicate
  const int r_neighbor = (rank != size - 1) ? rank + 1 : MPI_PROC_NULL;
  const int l_neighbor = (rank != 0) ? rank - 1 : MPI_PROC_NULL;

  send_recv<T, D, N>(r_neighbor, l_neighbor, false, view, smallsize);
  send_recv<T, D, N>(l_neighbor, r_neighbor, true, view, smallsize);

  MPI_Barrier(MPI_COMM_WORLD);
  CallOnce([]() {
    std::cout << "Finished " << D << "D ";
    if constexpr (std::is_same_v<T, float>) {
      std::cout << "float";
    } else {
      std::cout << "double";
    }
    std::cout << " communication test" << std::endl;
  });
}

auto main(int argc, char** argv) -> int {
  try {
    Kokkos::initialize(argc, argv);
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::size_t bigsize   = (std::sin((rank + 1) * 0.25) + 2) * 1e3;
    const std::size_t smallsize = 123;

    CallOnce(
      [](auto&& size, auto&& bigsize, auto&& smallsize) {
        std::cout << "Running the MPI communication test" << std::endl;
        std::cout << "- Number of MPI ranks: " << size << std::endl;
        std::cout << "- Big size: " << bigsize << std::endl;
        std::cout << "- Small size: " << smallsize << std::endl;
#if defined(GPU_AWARE_MPI) && defined(DEVICE_ENABLED)
        std::cout << "- GPU-aware MPI is enabled" << std::endl;
#else
        std::cout << "- GPU-aware MPI is disabled" << std::endl;
#endif
      },
      size,
      bigsize,
      smallsize);

    comm<float, 1, 3>(rank, size, bigsize, smallsize);
    comm<float, 2, 3>(rank, size, bigsize, smallsize);
    comm<float, 3, 3>(rank, size, bigsize, smallsize);

    comm<float, 1, 6>(rank, size, bigsize, smallsize);
    comm<float, 2, 6>(rank, size, bigsize, smallsize);
    comm<float, 3, 6>(rank, size, bigsize, smallsize);

    comm<double, 1, 3>(rank, size, bigsize, smallsize);
    comm<double, 2, 3>(rank, size, bigsize, smallsize);
    comm<double, 3, 3>(rank, size, bigsize, smallsize);

    comm<double, 1, 6>(rank, size, bigsize, smallsize);
    comm<double, 2, 6>(rank, size, bigsize, smallsize);
    comm<double, 3, 6>(rank, size, bigsize, smallsize);
  } catch (const std::exception& e) {
    if (MPI_COMM_WORLD != MPI_COMM_NULL) {
      MPI_Finalize();
    }
    if (Kokkos::is_initialized()) {
      Kokkos::finalize();
    }
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  MPI_Finalize();
  Kokkos::finalize();
  return 0;
}

template <typename Func, typename... Args>
void CallOnce(Func func, Args&&... args) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == MPI_ROOT_RANK) {
    func(std::forward<Args>(args)...);
  }
}
