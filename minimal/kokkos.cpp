#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

auto main(int argc, char** argv) -> int {
  try {
    Kokkos::initialize(argc, argv);
    Kokkos::DefaultExecutionSpace {}.print_configuration(std::cout);

    std::cout << "1D views" << std::endl;
    for (const auto& sz : { 100u, 10000u, 1000000u }) {
      Kokkos::View<float*> view { "test_view", sz };
      Kokkos::parallel_for(
        "fill_1d",
        Kokkos::RangePolicy<>(0, sz),
        KOKKOS_LAMBDA(std::size_t i) { view(i) = static_cast<float>(i); });
      Kokkos::fence();
      std::cout << "- allocated " << view.size() << std::endl;
    }

    std::cout << "2D views" << std::endl;
    for (const auto& sz : { 10u, 100u, 1000u }) {
      Kokkos::View<float**> view { "test_view", sz, 2 * sz };
      Kokkos::parallel_for(
        "fill_2d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { sz, 2 * sz }),
        KOKKOS_LAMBDA(std::size_t i, std::size_t j) {
          view(i, j) = static_cast<float>(i * 2 * sz + j);
        });
      Kokkos::fence();
      std::cout << "- allocated " << view.size() << std::endl;
    }

    std::cout << "3D views" << std::endl;
    for (const auto& sz : { 10u, 100u }) {
      Kokkos::View<float***> view { "test_view", sz, 2 * sz, 3 * sz };
      Kokkos::parallel_for(
        "fill_3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { sz, 2 * sz, 3 * sz }),
        KOKKOS_LAMBDA(std::size_t i, std::size_t j, std::size_t k) {
          view(i, j, k) = static_cast<float>(i * 2 * sz * 3 * sz + j * 3 * sz + k);
        });
      Kokkos::fence();
      std::cout << "- allocated " << view.size() << std::endl;
    }

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
