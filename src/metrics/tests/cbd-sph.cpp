#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/cubed_sphere.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

template <Dimension D>
Inline auto equal(const coord_t<D>& a, const coord_t<D>& b, real_t acc = ONE) -> bool {
  for (auto d { 0u }; d < D; ++d) {
    if (not cmp::AlmostEqual(a[d], b[d], epsilon * acc)) {
      Kokkos::printf("%d : %.12f != %.12f\n", d, a[d], b[d]);
      return false;
    }
  }
  return true;
}

auto main(int argc, char* argv[]) -> int {
    Kokkos::initialize(argc, argv);
  
    try {
      using namespace ntt;
      using namespace metric;
      const auto res = std::vector<std::size_t> { 64, 32 };
      const auto ext = boundaries_t<real_t> {
        { 1.0,         10.0 },
        { 0.0, constant::PI }
      };
      const auto params = std::map<std::string, real_t> {
        { "r0",         -ONE },
        {  "h", (real_t)0.25 }
      };
  
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
      Kokkos::finalize();
      return 1;
    }
    Kokkos::finalize();
    return 0;
  }