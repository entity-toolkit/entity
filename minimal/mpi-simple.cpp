#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <utility>

auto main(int argc, char** argv) -> int {
  try {
    Kokkos::initialize(argc, argv);
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto nelems = 500u;
    const auto nsend  = 10u;
    const auto nrecv  = 10u;

    if (rank == 0) {
      std::cout << "Running the simple MPI communication test" << std::endl;
      std::cout << "- Number of MPI ranks: " << size << std::endl;
      std::cout << "- Number elements to send/recv (2D): " << nelems << "x"
                << nsend << std::endl;
#if defined(GPU_AWARE_MPI) && defined(DEVICE_ENABLED)
      std::cout << "- GPU-aware MPI is enabled" << std::endl;
#else
      std::cout << "- GPU-aware MPI is disabled" << std::endl;
#endif
    }

    Kokkos::View<float**> view("view", nelems, nelems);
    Kokkos::View<float**> send("send", nsend, nelems);
    Kokkos::View<float**> recv("recv", nrecv, nelems);
    Kokkos::deep_copy(
      send,
      Kokkos::subview(view, std::make_pair(0u, nsend), Kokkos::ALL));

#if defined(GPU_AWARE_MPI) || !defined(DEVICE_ENABLED)
    MPI_Sendrecv(send.data(),
                 nsend * nelems,
                 MPI_FLOAT,
                 (rank + 1) % size,
                 0,
                 recv.data(),
                 nrecv * nelems,
                 MPI_FLOAT,
                 (rank - 1 + size) % size,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
#else
    auto send_h = Kokkos::create_mirror_view(send);
    auto recv_h = Kokkos::create_mirror_view(recv);
    Kokkos::deep_copy(send_h, send);
    MPI_Sendrecv(send_h.data(),
                 nsend * nelems,
                 MPI_FLOAT,
                 (rank + 1) % size,
                 0,
                 recv_h.data(),
                 nrecv * nelems,
                 MPI_FLOAT,
                 (rank - 1 + size) % size,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    Kokkos::deep_copy(recv, recv_h);
#endif
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
