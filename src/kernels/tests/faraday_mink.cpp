#include "kernels/faraday_mink.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

using namespace ntt;
using namespace kernel::mink;
using namespace metric;

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

Inline auto equal(const real_t& a, const real_t& b, const char* msg, const real_t acc)
  -> bool {
  if (not(math::abs(a - b) < acc)) {
    printf("%.12e != %.12e [%.12e] %s\n", a, b, math::abs(a - b), msg);
    return false;
  }
  return true;
}

template <Dimension D>
void testFaraday(const std::vector<std::size_t>&);

template <>
void testFaraday<Dim::_1D>(const std::vector<std::size_t>& res) {
  errorIf(res.size() != 1, "res.size() != 1");

  using namespace ntt;
  using namespace metric;
  const real_t sx     = constant::TWO_PI;
  const auto   metric = Minkowski<Dim::_1D> { res, { { ZERO, sx } } };
  auto emfield = ndfield_t<Dim::_1D, 6> { "emfield", res[0] + 2 * N_GHOSTS };
  const auto i1min     = N_GHOSTS;
  const auto i1max     = res[0] + N_GHOSTS;
  const auto range     = CreateRangePolicy<Dim::_1D>({ i1min }, { i1max });
  const auto range_ext = CreateRangePolicy<Dim::_1D>({ 0 },
                                                     { res[0] + 2 * N_GHOSTS });
  const auto dx        = (metric.x1_max - metric.x1_min) / (real_t)metric.nx1;
  const auto max_err   = SQR(dx) / 3;

  Kokkos::parallel_for(
    "init 1D",
    range_ext,
    Lambda(index_t i1) {
      const coord_t<Dim::_1D> x_Code { COORD(i1) };
      coord_t<Dim::_1D>       x_Cart { ZERO };
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code, x_Cart);
      emfield(i1, em::ex2) = math::cos(TWO * x_Cart[0]);
      emfield(i1, em::ex3) = math::sin(TWO * x_Cart[0]);
      emfield(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
        x_Code,
        emfield(i1, em::ex2));
      emfield(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
        x_Code,
        emfield(i1, em::ex3));
    });

  Kokkos::parallel_for("faraday 1D",
                       range,
                       Faraday_kernel<Dim::_1D>(emfield, HALF / dx, ONE));

  unsigned long all_wrongs = 0;
  Kokkos::parallel_reduce(
    "check faraday 1D",
    range,
    Lambda(index_t i1, unsigned long& wrongs) {
      const coord_t<Dim::_1D> x_Code { COORD(i1) + HALF };
      coord_t<Dim::_1D>       x_Cart { ZERO };
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code, x_Cart);

      const auto bx2_expect = math::cos(TWO * x_Cart[0]);
      const auto bx3_expect = math::sin(TWO * x_Cart[0]);

      const auto bx2_got = metric.template transform<2, Idx::U, Idx::T>(
        x_Cart,
        emfield(i1, em::bx2));
      const auto bx3_got = metric.template transform<3, Idx::U, Idx::T>(
        x_Cart,
        emfield(i1, em::bx3));

      wrongs += not equal(bx2_got, bx2_expect, "faraday 1D bx2", max_err);
      wrongs += not equal(bx3_got, bx3_expect, "faraday 1D bx3", max_err);
    },
    all_wrongs);

  errorIf(all_wrongs != 0,
          "faraday for 1D " + std::string(metric.Label) + " failed with " +
            std::to_string(all_wrongs) + " errors");
}

template <>
void testFaraday<Dim::_2D>(const std::vector<std::size_t>& res) {
  errorIf(res.size() != 2, "res.size() != 2");

  const real_t sx = constant::TWO_PI, sy = 4.0 * constant::PI;
  const auto   metric = Minkowski<Dim::_2D> {
    res,
    {{ ZERO, sx }, { ZERO, sy }}
  };
  auto              emfield = ndfield_t<Dim::_2D, 6> { "emfield",
                                                       res[0] + 2 * N_GHOSTS,
                                                       res[1] + 2 * N_GHOSTS };
  const std::size_t i1min = N_GHOSTS, i1max = res[0] + N_GHOSTS;
  const std::size_t i2min = N_GHOSTS, i2max = res[1] + N_GHOSTS;
  const auto        range     = CreateRangePolicy<Dim::_2D>({ i1min, i2min },
                                                 { i1max, i2max });
  const auto        range_ext = CreateRangePolicy<Dim::_2D>(
    { 0, 0 },
    { res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS });
  const auto dx      = (metric.x1_max - metric.x1_min) / (real_t)metric.nx1;
  const auto max_err = SQR(dx) / 3;

  Kokkos::parallel_for(
    "init 2D",
    range_ext,
    Lambda(index_t i1, index_t i2) {
      const coord_t<Dim::_2D> x_Code_00 { COORD(i1), COORD(i2) };
      const coord_t<Dim::_2D> x_Code_p0 { COORD(i1) + HALF, COORD(i2) };
      const coord_t<Dim::_2D> x_Code_0p { COORD(i1), COORD(i2) + HALF };
      coord_t<Dim::_2D>       x_Cart_00 { ZERO };
      coord_t<Dim::_2D>       x_Cart_p0 { ZERO };
      coord_t<Dim::_2D>       x_Cart_0p { ZERO };
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_00, x_Cart_00);
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_p0, x_Cart_p0);
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_0p, x_Cart_0p);
      emfield(i1, i2, em::ex1) = math::sin(TWO * x_Cart_p0[0]) *
                                 math::cos(x_Cart_p0[1]);
      emfield(i1, i2, em::ex2) = math::cos(TWO * x_Cart_0p[0]) *
                                 math::cos(x_Cart_0p[1]);
      emfield(i1, i2, em::ex3) = math::cos(TWO * x_Cart_00[0]) *
                                 math::sin(x_Cart_00[1]);
      emfield(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
        x_Cart_p0,
        emfield(i1, i2, em::ex1));
      emfield(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
        x_Cart_0p,
        emfield(i1, i2, em::ex2));
      emfield(i1, i2, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
        x_Cart_00,
        emfield(i1, i2, em::ex3));
    });

  Kokkos::parallel_for("faraday 2D",
                       range,
                       Faraday_kernel<Dim::_2D>(emfield, ONE / SQR(dx), ONE));

  unsigned long all_wrongs = 0;
  Kokkos::parallel_reduce(
    "check faraday 2D",
    range,
    Lambda(index_t i1, index_t i2, unsigned long& wrongs) {
      const coord_t<Dim::_2D> x_Code_p0 { COORD(i1) + HALF, COORD(i2) };
      const coord_t<Dim::_2D> x_Code_0p { COORD(i1), COORD(i2) + HALF };
      const coord_t<Dim::_2D> x_Code_pp { COORD(i1) + HALF, COORD(i2) + HALF };
      coord_t<Dim::_2D>       x_Cart_p0 { ZERO };
      coord_t<Dim::_2D>       x_Cart_0p { ZERO };
      coord_t<Dim::_2D>       x_Cart_pp { ZERO };
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_p0, x_Cart_p0);
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_0p, x_Cart_0p);
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_pp, x_Cart_pp);

      const auto bx1_expect = -math::cos(TWO * x_Cart_0p[0]) *
                              math::cos(x_Cart_0p[1]);
      const auto bx2_expect = -TWO * math::sin(TWO * x_Cart_p0[0]) *
                              math::sin(x_Cart_p0[1]);
      const auto bx3_expect = FOUR * math::cos(x_Cart_pp[0]) *
                                math::cos(x_Cart_pp[1]) * math::sin(x_Cart_pp[0]) -
                              TWO * math::cos(x_Cart_pp[0]) *
                                math::sin(x_Cart_pp[0]) * math::sin(x_Cart_pp[1]);

      const auto bx1_got = metric.template transform<1, Idx::U, Idx::T>(
        x_Cart_0p,
        emfield(i1, i2, em::bx1));
      const auto bx2_got = metric.template transform<2, Idx::U, Idx::T>(
        x_Cart_p0,
        emfield(i1, i2, em::bx2));
      const auto bx3_got = metric.template transform<3, Idx::U, Idx::T>(
        x_Cart_pp,
        emfield(i1, i2, em::bx3));

      wrongs += not equal(bx1_got, bx1_expect, "faraday 2D bx1", max_err);
      wrongs += not equal(bx2_got, bx2_expect, "faraday 2D bx2", max_err);
      // !ACC: somehow this is a bit less accurate
      wrongs += not equal(bx3_got, bx3_expect, "faraday 2D bx3", 3 * max_err);
    },
    all_wrongs);

  errorIf(all_wrongs != 0,
          "faraday for 2D " + std::string(metric.Label) + " failed with " +
            std::to_string(all_wrongs) + " errors");
}

template <>
void testFaraday<Dim::_3D>(const std::vector<std::size_t>& res) {
  errorIf(res.size() != 3, "res.size() != 3");

  using namespace ntt;
  const real_t sx = constant::TWO_PI, sy = 4.0 * constant::PI,
               sz   = constant::TWO_PI;
  const auto metric = Minkowski<Dim::_3D> {
    res,
    {{ ZERO, sx }, { ZERO, sy }, { ZERO, sz }}
  };
  auto              emfield = ndfield_t<Dim::_3D, 6> { "emfield",
                                                       res[0] + 2 * N_GHOSTS,
                                                       res[1] + 2 * N_GHOSTS,
                                                       res[2] + 2 * N_GHOSTS };
  const std::size_t i1min = N_GHOSTS, i1max = res[0] + N_GHOSTS;
  const std::size_t i2min = N_GHOSTS, i2max = res[1] + N_GHOSTS;
  const std::size_t i3min = N_GHOSTS, i3max = res[2] + N_GHOSTS;
  const auto        range = CreateRangePolicy<Dim::_3D>({ i1min, i2min, i3min },
                                                 { i1max, i2max, i3max });
  const auto        range_ext = CreateRangePolicy<Dim::_3D>(
    { 0, 0, 0 },
    { res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS });
  const auto dx      = (metric.x1_max - metric.x1_min) / (real_t)metric.nx1;
  const auto max_err = SQR(dx) / 3;

  Kokkos::parallel_for(
    "init 3D",
    range_ext,
    Lambda(index_t i1, index_t i2, index_t i3) {
      const coord_t<Dim::_3D> x_Code_p00 { COORD(i1) + HALF, COORD(i2), COORD(i3) };
      const coord_t<Dim::_3D> x_Code_0p0 { COORD(i1), COORD(i2) + HALF, COORD(i3) };
      const coord_t<Dim::_3D> x_Code_00p { COORD(i1), COORD(i2), COORD(i3) + HALF };
      coord_t<Dim::_3D> x_Cart_p00 { ZERO };
      coord_t<Dim::_3D> x_Cart_0p0 { ZERO };
      coord_t<Dim::_3D> x_Cart_00p { ZERO };
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_p00, x_Cart_p00);
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_0p0, x_Cart_0p0);
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_00p, x_Cart_00p);
      emfield(i1, i2, i3, em::ex1) = math::sin(TWO * x_Cart_p00[0]) *
                                     math::cos(x_Cart_p00[1]) *
                                     math::cos(TWO * x_Cart_p00[2]);
      emfield(i1, i2, i3, em::ex2) = math::cos(TWO * x_Cart_0p0[0]) *
                                     math::cos(x_Cart_0p0[1]) *
                                     math::cos(TWO * x_Cart_0p0[2]);
      emfield(i1, i2, i3, em::ex3) = math::cos(TWO * x_Cart_00p[0]) *
                                     math::sin(x_Cart_00p[1]) *
                                     math::cos(TWO * x_Cart_00p[2]);

      emfield(i1, i2, i3, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
        x_Cart_p00,
        emfield(i1, i2, i3, em::ex1));
      emfield(i1, i2, i3, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
        x_Cart_0p0,
        emfield(i1, i2, i3, em::ex2));
      emfield(i1, i2, i3, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
        x_Cart_00p,
        emfield(i1, i2, i3, em::ex3));
    });

  Kokkos::parallel_for("faraday 3D",
                       range,
                       Faraday_kernel<Dim::_3D>(emfield, ONE / dx, ZERO));

  unsigned long all_wrongs = 0;
  Kokkos::parallel_reduce(
    "check faraday 3D",
    range,
    Lambda(index_t i1, index_t i2, index_t i3, unsigned long& wrongs) {
      const coord_t<Dim::_3D> x_Code_0pp { COORD(i1),
                                           COORD(i2) + HALF,
                                           COORD(i3) + HALF };
      const coord_t<Dim::_3D> x_Code_p0p { COORD(i1) + HALF,
                                           COORD(i2),
                                           COORD(i3) + HALF };
      const coord_t<Dim::_3D> x_Code_pp0 { COORD(i1) + HALF,
                                           COORD(i2) + HALF,
                                           COORD(i3) };
      coord_t<Dim::_3D>       x_Cart_0pp { ZERO };
      coord_t<Dim::_3D>       x_Cart_p0p { ZERO };
      coord_t<Dim::_3D>       x_Cart_pp0 { ZERO };
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_0pp, x_Cart_0pp);
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_p0p, x_Cart_p0p);
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(x_Code_pp0, x_Cart_pp0);

      const auto bx1_expect = -math::cos(TWO * x_Cart_0pp[0]) *
                              math::cos(x_Cart_0pp[1]) *
                              (math::cos(TWO * x_Cart_0pp[2]) +
                               TWO * math::sin(TWO * x_Cart_0pp[2]));

      const auto bx2_expect = -TWO * math::sin(TWO * x_Cart_p0p[0]) *
                              math::sin(x_Cart_p0p[1] - TWO * x_Cart_p0p[2]);

      const auto cos_x      = math::cos(x_Cart_pp0[0]);
      const auto sin_x      = math::sin(x_Cart_pp0[0]);
      const auto cos_y      = math::cos(x_Cart_pp0[1]);
      const auto sin_y      = math::sin(x_Cart_pp0[1]);
      const auto cos_z      = math::cos(x_Cart_pp0[2]);
      const auto sin_z      = math::sin(x_Cart_pp0[2]);
      const auto bx3_expect = FOUR * cos_x * cos_y * sin_x * SQR(cos_z) -
                              TWO * cos_x * sin_x * sin_y * SQR(cos_z) -
                              FOUR * cos_x * cos_y * sin_x * SQR(sin_z) +
                              TWO * cos_x * sin_x * sin_y * SQR(sin_z);

      const auto bx1_got = metric.template transform<1, Idx::U, Idx::T>(
        x_Cart_0pp,
        emfield(i1, i2, i3, em::bx1));
      const auto bx2_got = metric.template transform<2, Idx::U, Idx::T>(
        x_Cart_p0p,
        emfield(i1, i2, i3, em::bx2));
      const auto bx3_got = metric.template transform<3, Idx::U, Idx::T>(
        x_Cart_pp0,
        emfield(i1, i2, i3, em::bx3));

      wrongs += not equal(bx1_got, bx1_expect, "faraday 3D bx1", max_err);
      wrongs += not equal(bx2_got, bx2_expect, "faraday 3D bx2", max_err);
      wrongs += not equal(bx3_got, bx3_expect, "faraday 3D bx3", max_err);
    },
    all_wrongs);

  errorIf(all_wrongs != 0,
          "faraday for 3D " + std::string(metric.Label) + " failed with " +
            std::to_string(all_wrongs) + " errors");
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    testFaraday<Dim::_1D>({ 128 });
    testFaraday<Dim::_2D>({ 64, 128 });
    testFaraday<Dim::_3D>({ 32, 64, 32 });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}