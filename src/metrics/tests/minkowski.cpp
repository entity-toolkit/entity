#include "metrics/minkowski.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

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
Inline auto equal(const coord_t<D>& a, const coord_t<D>& b, real_t acc = ONE)
  -> bool {
  for (auto d { 0u }; d < D; ++d) {
    if (not cmp::AlmostEqual(a[d], b[d], epsilon * acc)) {
      printf("%d : %.12f != %.12f\n", d, a[d], b[d]);
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
    {
      // catch unequal dx error
      try {
        Minkowski<Dim::_3D>(
          {
            256,
            128,
            64
        },
          { { -2.0, 2.0 }, { -1.0, 1.0 }, { -1.0, 1.0 } });
      } catch (const std::exception& e) {
        errorIf(std::string(e.what()).find("dx3 must be equal to dx1 in 3D") ==
                  std::string::npos,
                "unequal dx error not caught");
      }
    }

    constexpr unsigned int nx = 128;
    constexpr unsigned int ny = 64;
    constexpr unsigned int nz = 32;
    const auto             M  = Minkowski<Dim::_3D>(
      {
        nx,
        ny,
        nz
    },
      { { -2.0, 2.0 }, { -1.0, 1.0 }, { -0.5, 0.5 } });
    const auto        dx = (real_t)(4.0 / nx);
    coord_t<Dim::_3D> dummy { ZERO, ZERO, ZERO };
    errorIf(not cmp::AlmostEqual(M.dxMin(), dx / (real_t)math::sqrt(3.0)),
            "dxMin not set correctly");
    errorIf(not cmp::AlmostEqual(M.h_<1, 1>(dummy), SQR(dx)),
            "h_11 not set correctly");
    errorIf(not cmp::AlmostEqual(M.h_<2, 2>(dummy), SQR(dx)),
            "h_22 not set correctly");
    errorIf(not cmp::AlmostEqual(M.h_<3, 3>(dummy), SQR(dx)),
            "h_33 not set correctly");
    errorIf(not cmp::AlmostEqual(M.sqrt_det_h(dummy), CUBE(dx)),
            "sqrt_det_h not set correctly");

    // coord transformations
    // code <-> phys <-> sph
    unsigned long all_wrongs = 0;
    Kokkos::parallel_reduce(
      "code-phys",
      CreateRangePolicy<Dim::_3D>({ N_GHOSTS, N_GHOSTS, N_GHOSTS },
                                  { nx + N_GHOSTS, ny + N_GHOSTS, nz + N_GHOSTS }),
      Lambda(index_t i1, index_t i2, index_t i3, unsigned long& wrongs) {
        const coord_t<Dim::_3D> x_Code { COORD(i1), COORD(i2), COORD(i3) };
        coord_t<Dim::_3D>       x_Sph { ZERO, ZERO, ZERO };
        coord_t<Dim::_3D>       x_Phys { ZERO, ZERO, ZERO };

        M.template convert<Crd::Cd, Crd::Ph>(x_Code, x_Phys);
        wrongs += not equal<Dim::_3D>(
          x_Phys,
          { -TWO + dx * x_Code[0], -ONE + dx * x_Code[1], -HALF + dx * x_Code[2] });

        M.template convert<Crd::Cd, Crd::Sph>(x_Code, x_Sph);
        wrongs += not equal<Dim::_3D>(
          x_Sph,
          { math::sqrt(SQR(x_Phys[0]) + SQR(x_Phys[1]) + SQR(x_Phys[2])),
            static_cast<real_t>(constant::HALF_PI) -
              math::atan2(x_Phys[2], math::sqrt(SQR(x_Phys[0]) + SQR(x_Phys[1]))),
            static_cast<real_t>(constant::PI) -
              math::atan2(x_Phys[1], -x_Phys[0]) });
      },
      all_wrongs);
    errorIf(all_wrongs != 0, "code-phys-sph transformations failed");
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
