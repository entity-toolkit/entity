#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <mpi.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

auto main(int argc, char** argv) -> int {
  try {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Kokkos::initialize(argc, argv);

    Kokkos::DefaultExecutionSpace {}.print_configuration(std::cout);

    std::uint64_t                    RandomSeed = 0x123456789abcdef0;
    Kokkos::Random_XorShift64_Pool<> random_pool(
      RandomSeed + static_cast<std::uint64_t>(rank));

    const auto sz = (std::size_t)(1e8);

    Kokkos::View<float*> view { "rand_view", sz };

    for (auto i = 0; i < 100; ++i) {
      Kokkos::parallel_for(
        "fill_random",
        Kokkos::RangePolicy<>(0, sz),
        KOKKOS_LAMBDA(std::size_t i) {
          auto gen = random_pool.get_state();
          view(i)  = gen.frand();
          random_pool.free_state(gen);
        });
      Kokkos::fence();
    }

    std::cout << "Rank " << rank << " finished\n" << std::flush;
    Kokkos::fence();

  } catch (const std::exception& e) {
    if (Kokkos::is_initialized()) {
      Kokkos::finalize();
      MPI_Finalize();
    }
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
