#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

auto main(int argc, char** argv) -> int {
  try {
    Kokkos::initialize(argc, argv);
    Kokkos::DefaultExecutionSpace {}.print_configuration(std::cout);

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);

    const auto sz = (std::size_t)(1e8);

    Kokkos::View<float*> view { "rand_view", sz };
    Kokkos::parallel_for(
      "fill_random",
      Kokkos::RangePolicy<>(0, sz),
      KOKKOS_LAMBDA(std::size_t i) {
        auto gen = random_pool.get_state();
        view(i)  = gen.frand();
        random_pool.free_state(gen);
      });
    Kokkos::fence();

  } catch (const std::exception& e) {
    if (Kokkos::is_initialized()) {
      Kokkos::finalize();
    }
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  Kokkos::finalize();
  return 0;
}
