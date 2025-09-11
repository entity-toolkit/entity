#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <mpi.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

template <typename T>
void create_histogram(Kokkos::View<T*>& histogram,
                      T                 min,
                      T                 max,
                      Kokkos::View<T*>  values) {
  deep_copy(histogram, static_cast<T>(0));
  Kokkos::View<T*, Kokkos::MemoryTraits<Kokkos::Atomic>> histogram_atomic = histogram;
  Kokkos::parallel_for(
    "histogram",
    values.extent(0),
    KOKKOS_LAMBDA(std::size_t i) {
      if (values(i) < min || values(i) >= max) {
        return;
      }
      const std::size_t index = (1.0 * (values(i) - min) / (max - min)) *
                                histogram.extent(0);
      if (index >= histogram.extent(0)) {
        return;
      }
      histogram_atomic(index)++;
    });
}

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
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      const int            histogram_bins = 10;
      float                min            = 0.0f;
      float                max            = 0.1f;
      Kokkos::View<float*> histogram("histogram", histogram_bins);
      create_histogram(histogram, min, max, view);
      auto h_histogram = Kokkos::create_mirror_view(histogram);
      Kokkos::deep_copy(h_histogram, histogram);
      std::cout << "hist: (" << min << ", " << max << ")\n";
      for (int i = 0; i < histogram_bins; ++i) {
        std::cout << h_histogram(i) << " ";
      }
      std::cout << std::endl;
    }

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
