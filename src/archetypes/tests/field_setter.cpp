#include "archetypes/field_setter.h"

#include "enums.h"
#include "global.h"

#include "metrics/minkowski.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

using namespace ntt;
using namespace metric;
using namespace arch;

template <Dimension D>
struct ConstantFiller {

  ConstantFiller(real_t value) : value { value } {}

  Inline auto ex1(const coord_t<D>&) const -> real_t {
    return value;
  }

  Inline auto bx3(const coord_t<D>&) const -> real_t {
    return -value;
  }

private:
  real_t value;
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    constexpr std::size_t    nx1 = 20;
    ndfield_t<Dim::_1D, 6>   EM { "EM", nx1 + 2 * N_GHOSTS };
    Minkowski<Dim::_1D>      metric { { nx1 }, { { -2.0, 2.0 } } };
    ConstantFiller<Dim::_1D> finit { 123.0 };
    SetEMFields_kernel<ConstantFiller<Dim::_1D>, SimEngine::SRPIC, Minkowski<Dim::_1D>>
      finitializer { EM, finit, metric };
    Kokkos::parallel_for(
      "InitFields",
      CreateRangePolicy<Dim::_1D>({ N_GHOSTS }, { nx1 + N_GHOSTS }),
      finitializer);
    auto EM_h = Kokkos::create_mirror_view(EM);
    Kokkos::deep_copy(EM_h, EM);
    for (std::size_t i1 = 0; i1 < nx1 + 2 * N_GHOSTS; ++i1) {
      if ((i1 < N_GHOSTS) || (i1 >= nx1 + N_GHOSTS)) {
        for (const auto& comp :
             { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
          raise::ErrorIf(EM_h(i1, comp) != 0.0, "Invalid value", HERE);
        }
      }
      if ((i1 >= N_GHOSTS) && (i1 < nx1 + N_GHOSTS)) {
        raise::ErrorIf(EM_h(i1, em::ex1) != 615.0, "Invalid value", HERE);
        raise::ErrorIf(EM_h(i1, em::bx3) != -123.0, "Invalid value", HERE);
        for (const auto& comp : { em::ex2, em::ex3, em::bx1, em::bx2 }) {
          raise::ErrorIf(EM_h(i1, comp) != 0.0, "Invalid value", HERE);
        }
      }
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}