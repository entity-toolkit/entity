#include "kernels/reduced_stats.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace ntt;
using namespace metric;

template <Dimension D, unsigned short N>
class Fill_kernel {
  ndfield_t<D, N> arr;
  real_t          v;
  unsigned short  c;

public:
  Fill_kernel(ndfield_t<D, N>& arr_, real_t v_, unsigned short c_)
    : arr { arr_ }
    , v { v_ }
    , c { c_ } {
    raise::ErrorIf(c_ >= N, "c > N", HERE);
  }

  Inline void operator()(cellidx_t i1) const {
    arr(i1, c) = v;
  }

  Inline void operator()(cellidx_t i1, cellidx_t i2) const {
    arr(i1, i2, c) = v;
  }

  Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
    arr(i1, i2, i3, c) = v;
  }
};

template <Dimension D, unsigned short N>
void put_value(ndfield_t<D, N>& arr, real_t v, unsigned short c) {
  range_t<D> range;
  if constexpr (D == Dim::_1D) {
    range = range_t<D>({ 0u, arr.extent(0) });
  } else if constexpr (D == Dim::_2D) {
    range = {
      {            0u,            0u },
      { arr.extent(0), arr.extent(1) }
    };
  } else {
    range = {
      {            0u,            0u,            0u },
      { arr.extent(0), arr.extent(1), arr.extent(2) }
    };
  }
  Kokkos::parallel_for("Fill", range, Fill_kernel<D, N>(arr, v, c));
}

template <SimEngine::type S, class M, StatsID::type F, unsigned short I = 0>
auto compute_field_stat(const M&                    metric,
                        const ndfield_t<M::Dim, 6>& em,
                        const ndfield_t<M::Dim, 3>& j,
                        const range_t<M::Dim>&      range) -> real_t {
  real_t buff = ZERO;
  Kokkos::parallel_reduce("ReduceFields",
                          range,
                          kernel::ReducedFields_kernel<S, M, F, I>(em, j, metric),
                          buff);
  return buff / metric.totVolume();
}

inline static constexpr auto epsilon = std::numeric_limits<real_t>::epsilon();

/**
 * Relative comparison with an absolute floor of `tol`, so that expected values
 * of exactly zero are handled as well.
 */
auto almost_equal(real_t a, real_t b, real_t tol) -> bool {
  return math::fabs(a - b) <= tol * math::max(math::fabs(b), ONE);
}

void check(const std::string& name, real_t got, real_t expect, real_t tol) {
  raise::ErrorIf(not almost_equal(got, expect, tol),
                 name + " does not match expected value: got " +
                   std::to_string(got) + ", expected " + std::to_string(expect) +
                   " (tolerance " + std::to_string(tol) + ")",
                 HERE);
}

/**
 * @param acc dimensionless safety factor on top of the expected round-off bound
 */
template <SimEngine::type S, class M>
void testReducedStats(const std::vector<ncells_t>& res,
                      const boundaries_t<real_t>&  ext,
                      const real_t                 acc = ONE) {
  raise::ErrorIf(res.size() != M::Dim, "Invalid resolution size", HERE);

  M metric { res, ext, {} };

  std::vector<ncells_t> x_s, y_s, z_s;

  coord_t<M::Dim>     dummy { ZERO };
  std::vector<real_t> values;
  values.push_back(metric.template transform<1, Idx::T, Idx::U>(dummy, ONE));
  values.push_back(metric.template transform<2, Idx::T, Idx::U>(dummy, TWO));
  values.push_back(metric.template transform<3, Idx::T, Idx::U>(dummy, THREE));

  values.push_back(metric.template transform<1, Idx::T, Idx::U>(dummy, FOUR * ONE));
  values.push_back(metric.template transform<2, Idx::T, Idx::U>(dummy, FOUR * TWO));
  values.push_back(
    metric.template transform<3, Idx::T, Idx::U>(dummy, FOUR * THREE));

  values.push_back(metric.template transform<1, Idx::T, Idx::U>(dummy, -ONE));
  values.push_back(metric.template transform<2, Idx::T, Idx::U>(dummy, -TWO));
  values.push_back(metric.template transform<3, Idx::T, Idx::U>(dummy, THREE));

  values.push_back(metric.template transform<1, Idx::T, Idx::U>(dummy, FOUR));
  values.push_back(metric.template transform<2, Idx::T, Idx::U>(dummy, TWO));
  values.push_back(metric.template transform<3, Idx::T, Idx::U>(dummy, ONE));

  ndfield_t<M::Dim, 6> EM;
  ndfield_t<M::Dim, 3> J;
  range_t<M::Dim>      cell_range;

  if constexpr (M::Dim == Dim::_1D) {
    EM         = ndfield_t<M::Dim, 6> { "EM", res[0] + 2 * N_GHOSTS };
    J          = ndfield_t<M::Dim, 3> { "J", res[0] + 2 * N_GHOSTS };
    cell_range = { N_GHOSTS, res[0] + N_GHOSTS };

    put_value<M::Dim, 6>(EM, values[0], em::ex1);
    put_value<M::Dim, 6>(EM, values[1], em::ex2);
    put_value<M::Dim, 6>(EM, values[2], em::ex3);

    put_value<M::Dim, 6>(EM, values[6], em::bx1);
    put_value<M::Dim, 6>(EM, values[7], em::bx2);
    put_value<M::Dim, 6>(EM, values[8], em::bx3);

    put_value<M::Dim, 3>(J, values[9], cur::jx1);
    put_value<M::Dim, 3>(J, values[10], cur::jx2);
    put_value<M::Dim, 3>(J, values[11], cur::jx3);
  } else if constexpr (M::Dim == Dim::_2D) {
    EM = ndfield_t<M::Dim, 6> { "EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS };
    J = ndfield_t<M::Dim, 3> { "J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS };

    cell_range = {
      {          N_GHOSTS,          N_GHOSTS },
      { res[0] + N_GHOSTS, res[1] + N_GHOSTS }
    };

    put_value<M::Dim, 6>(EM, values[0], em::ex1);
    put_value<M::Dim, 6>(EM, values[1], em::ex2);
    put_value<M::Dim, 6>(EM, values[2], em::ex3);

    put_value<M::Dim, 6>(EM, values[6], em::bx1);
    put_value<M::Dim, 6>(EM, values[7], em::bx2);
    put_value<M::Dim, 6>(EM, values[8], em::bx3);

    put_value<M::Dim, 3>(J, values[9], cur::jx1);
    put_value<M::Dim, 3>(J, values[10], cur::jx2);
    put_value<M::Dim, 3>(J, values[11], cur::jx3);
  } else {
    EM = ndfield_t<M::Dim, 6> { "EM",
                                res[0] + 2 * N_GHOSTS,
                                res[1] + 2 * N_GHOSTS,
                                res[2] + 2 * N_GHOSTS };
    J  = ndfield_t<M::Dim, 3> { "J",
                                res[0] + 2 * N_GHOSTS,
                                res[1] + 2 * N_GHOSTS,
                                res[2] + 2 * N_GHOSTS };

    cell_range = {
      {          N_GHOSTS,          N_GHOSTS,          N_GHOSTS },
      { res[0] + N_GHOSTS, res[1] + N_GHOSTS, res[2] + N_GHOSTS }
    };

    put_value<M::Dim, 6>(EM, values[0], em::ex1);
    put_value<M::Dim, 6>(EM, values[1], em::ex2);
    put_value<M::Dim, 6>(EM, values[2], em::ex3);

    put_value<M::Dim, 6>(EM, values[6], em::bx1);
    put_value<M::Dim, 6>(EM, values[7], em::bx2);
    put_value<M::Dim, 6>(EM, values[8], em::bx3);

    put_value<M::Dim, 3>(J, values[9], cur::jx1);
    put_value<M::Dim, 3>(J, values[10], cur::jx2);
    put_value<M::Dim, 3>(J, values[11], cur::jx3);
  }

  // the stats kernels accumulate over all active cells with a plain (i.e.
  // non-compensated) sum in `real_t`, so the round-off of the reduction itself
  // grows linearly with the number of cells being reduced over
  ncells_t ncells = 1;
  for (const auto& r : res) {
    ncells *= r;
  }
  const auto tol = acc * epsilon * static_cast<real_t>(ncells);

  check("Ex_Sq",
        compute_field_stat<S, M, StatsID::E2, 1>(metric, EM, J, cell_range),
        (real_t)(1),
        tol);
  check("Ey_Sq",
        compute_field_stat<S, M, StatsID::E2, 2>(metric, EM, J, cell_range),
        (real_t)(4),
        tol);
  check("Ez_Sq",
        compute_field_stat<S, M, StatsID::E2, 3>(metric, EM, J, cell_range),
        (real_t)(9),
        tol);

  check("Bx_Sq",
        compute_field_stat<S, M, StatsID::B2, 1>(metric, EM, J, cell_range),
        (real_t)(1),
        tol);
  check("By_Sq",
        compute_field_stat<S, M, StatsID::B2, 2>(metric, EM, J, cell_range),
        (real_t)(4),
        tol);
  check("Bz_Sq",
        compute_field_stat<S, M, StatsID::B2, 3>(metric, EM, J, cell_range),
        (real_t)(9),
        tol);

  check("ExB_x",
        compute_field_stat<S, M, StatsID::ExB, 1>(metric, EM, J, cell_range),
        (real_t)(12),
        tol);
  check("ExB_y",
        compute_field_stat<S, M, StatsID::ExB, 2>(metric, EM, J, cell_range),
        (real_t)(-6),
        tol);
  check("ExB_z",
        compute_field_stat<S, M, StatsID::ExB, 3>(metric, EM, J, cell_range),
        (real_t)(0),
        tol);

  check("JdotE",
        compute_field_stat<S, M, StatsID::JdotE>(metric, EM, J, cell_range),
        (real_t)(11),
        tol);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    const ncells_t            nx = 100, ny = 123, nz = 52;
    std::pair<real_t, real_t> x_ext { -2.0, 2.0 };
    std::pair<real_t, real_t> y_ext { 0.0, 4.92 };
    std::pair<real_t, real_t> z_ext { 0.0, 2.08 };

    testReducedStats<SimEngine::SRPIC, Minkowski<Dim::_1D>>({ nx }, { x_ext });
    testReducedStats<SimEngine::SRPIC, Minkowski<Dim::_2D>>({ nx, ny },
                                                            { x_ext, y_ext });
    testReducedStats<SimEngine::SRPIC, Minkowski<Dim::_3D>>({ nx, ny, nz },
                                                            { x_ext, y_ext, z_ext });

  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
