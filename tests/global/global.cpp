#include "global.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

void errorIf(bool condition, const char* message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

auto main(int argc, char* argv[]) -> int {
  using namespace ntt;

  static_assert(N_GHOSTS == 2, "N_GHOSTS must be 2");
  static_assert(COORD(2.0) == 0.0, "COORD(2) must be 0");
  static_assert(Dim::_1D == 1, "Dim::_1D must be 1");
  static_assert(Dim::_2D == 2, "Dim::_2D must be 2");
  static_assert(Dim::_3D == 3, "Dim::_3D must be 3");

  static_assert(sizeof(tuple_t<int, Dim::_1D>) == 1 * sizeof(int));
  static_assert(sizeof(tuple_t<int, Dim::_2D>) == 2 * sizeof(int));
  static_assert(sizeof(tuple_t<int, Dim::_3D>) == 3 * sizeof(int));

  tuple_t<int, Dim::_1D> t1d { 123 };
  errorIf(t1d[0] != 123, "t1d[0] must be 123");
  tuple_t<int, Dim::_2D> t2d { 123, 456 };
  errorIf(t2d[0] != 123, "t2d[0] must be 123");
  errorIf(t2d[1] != 456, "t2d[1] must be 456");
  tuple_t<int, Dim::_3D> t3d { 123, 456, 789 };
  errorIf(t3d[0] != 123, "t3d[0] must be 123");
  errorIf(t3d[1] != 456, "t3d[1] must be 456");
  errorIf(t3d[2] != 789, "t3d[2] must be 789");

  static_assert(std::is_same_v<list_t<int, 1>, tuple_t<int, Dim::_1D>>);
  static_assert(std::is_same_v<list_t<int, 2>, tuple_t<int, Dim::_2D>>);
  static_assert(std::is_same_v<list_t<int, 3>, tuple_t<int, Dim::_3D>>);

  static_assert(std::is_same_v<coord_t<Dim::_1D>, tuple_t<real_t, Dim::_1D>>);
  static_assert(std::is_same_v<coord_t<Dim::_2D>, tuple_t<real_t, Dim::_2D>>);
  static_assert(std::is_same_v<coord_t<Dim::_3D>, tuple_t<real_t, Dim::_3D>>);

  static_assert(std::is_same_v<vec_t<Dim::_1D>, tuple_t<real_t, Dim::_1D>>);
  static_assert(std::is_same_v<vec_t<Dim::_2D>, tuple_t<real_t, Dim::_2D>>);
  static_assert(std::is_same_v<vec_t<Dim::_3D>, tuple_t<real_t, Dim::_3D>>);

  static_assert(std::is_same_v<range_tuple_t, std::pair<std::size_t, std::size_t>>);
  static_assert(std::is_same_v<index_t, const std::size_t>);

  try {
    GlobalInitialize(argc, argv);

    errorIf(not Kokkos::is_initialized(), "Kokkos was not initialized");

  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    GlobalFinalize();
    return -1;
  }

  GlobalFinalize();

  errorIf(Kokkos::is_initialized(), "Kokkos was not finalized");
  errorIf(not Kokkos::is_finalized(), "Kokkos was not finalized");

  return 0;
}