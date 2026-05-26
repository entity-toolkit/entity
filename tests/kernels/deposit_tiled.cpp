/**
 * @file tests/kernels/deposit_tiled.cpp
 * @brief X-1 numerical-equivalence test for the tiled deposit kernel.
 *
 * Runs the flat (`DepositCurrents_kernel`) and tiled
 * (`DepositCurrents_kernel_tiled`) kernels on identical particle SoA inputs
 * for shape orders O = 1..11 and asserts that the resulting J array is
 * identical cell-by-cell within a small floating-point tolerance.
 *
 * Built only when `team_policy=ON` (`-D TEAM_POLICY` defined). The test
 * matches the per-particle setup used in `deposit.cpp` so that any
 * regression in the shared `kernel::deposit::deposit_one_particle` body
 * is caught by both tests.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"

#include "metrics/minkowski.h"

#include "kernels/currents_deposit.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

  using namespace ntt;

  void errorIf(bool condition, const std::string& msg) {
    if (condition) {
      throw std::runtime_error(msg);
    }
  }

  template <typename T>
  void put_value(const array_t<T*>& arr, T value, int i) {
    auto h = Kokkos::create_mirror_view(arr);
    h(i)   = value;
    Kokkos::deep_copy(arr, h);
  }

  // Builds tile_offsets for a single-particle test. Particle 0 is alive
  // and lives in tile (tx1, tx2); slots 1..n_slots-1 carry the dead
  // sentinel and are never referenced by tile_offsets — so the tiled
  // kernel never iterates over them.
  array_t<npart_t*> build_tile_offsets_single_particle(ncells_t ntx1,
                                                       ncells_t ntx2,
                                                       ncells_t tx1,
                                                       ncells_t tx2) {
    const ncells_t total_tiles = ntx1 * ntx2;
    const ncells_t hot_tile    = tx1 * ntx2 + tx2;
    array_t<npart_t*> offsets("tile_offsets", total_tiles + 1u);
    auto h = Kokkos::create_mirror_view(offsets);
    for (ncells_t t = 0; t <= total_tiles; ++t) {
      h(t) = (t <= hot_tile) ? static_cast<npart_t>(0)
                             : static_cast<npart_t>(1);
    }
    Kokkos::deep_copy(offsets, h);
    return offsets;
  }

  template <unsigned short O, unsigned short T_TILE>
  void run_one_case() {
    using metric_t = metric::Minkowski<Dim::_2D>;
    constexpr unsigned short nx1 = 50u, nx2 = 50u;
    metric_t metric { { nx1, nx2 },
                      { { 0.0, 55.0 }, { 0.0, 55.0 } },
                      {} };

    // Particle setup (mirrors deposit.cpp).
    const int      i0 = 25, j0 = 21, i0f = 24, j0f = 20;
    const real_t   uz = 2.5;
    const prtldx_t dxi = static_cast<prtldx_t>(0.65);
    const prtldx_t dxf = static_cast<prtldx_t>(0.99);
    const prtldx_t dyi = static_cast<prtldx_t>(0.65);
    const prtldx_t dyf = static_cast<prtldx_t>(0.80);

    array_t<int*>      i1 { "i1", 10 };
    array_t<int*>      i2 { "i2", 10 };
    array_t<int*>      i3 { "i3", 10 };
    array_t<int*>      i1_prev { "i1_prev", 10 };
    array_t<int*>      i2_prev { "i2_prev", 10 };
    array_t<int*>      i3_prev { "i3_prev", 10 };
    array_t<prtldx_t*> dx1 { "dx1", 10 };
    array_t<prtldx_t*> dx2 { "dx2", 10 };
    array_t<prtldx_t*> dx3 { "dx3", 10 };
    array_t<prtldx_t*> dx1_prev { "dx1_prev", 10 };
    array_t<prtldx_t*> dx2_prev { "dx2_prev", 10 };
    array_t<prtldx_t*> dx3_prev { "dx3_prev", 10 };
    array_t<real_t*>   ux1 { "ux1", 10 };
    array_t<real_t*>   ux2 { "ux2", 10 };
    array_t<real_t*>   ux3 { "ux3", 10 };
    array_t<real_t*>   phi { "phi", 10 };
    array_t<real_t*>   weight { "weight", 10 };
    array_t<short*>    tag { "tag", 10 };
    const real_t       charge = 1.0, dt = 1.0;

    put_value<int>(i1, i0f, 0);
    put_value<int>(i2, j0f, 0);
    put_value<int>(i1_prev, i0, 0);
    put_value<int>(i2_prev, j0, 0);
    put_value<prtldx_t>(dx1, dxf, 0);
    put_value<prtldx_t>(dx2, dyf, 0);
    put_value<prtldx_t>(dx1_prev, dxi, 0);
    put_value<prtldx_t>(dx2_prev, dyi, 0);
    put_value<real_t>(ux1, ZERO, 0);
    put_value<real_t>(ux2, ZERO, 0);
    put_value<real_t>(ux3, uz, 0);
    put_value<real_t>(weight, 1.0, 0);
    put_value<short>(tag, ParticleTag::alive, 0);

    // Run the flat kernel.
    ndfield_t<Dim::_2D, 3> J_flat { "J_flat",
                                    nx1 + 2u * N_GHOSTS,
                                    nx2 + 2u * N_GHOSTS };
    {
      auto J_scat = Kokkos::Experimental::create_scatter_view(J_flat);
      Kokkos::parallel_for(
        "FlatDeposit",
        10,
        kernel::DepositCurrents_kernel<SimEngine::SRPIC, metric_t, O>(
          J_scat,
          i1, i2, i3,
          i1_prev, i2_prev, i3_prev,
          dx1, dx2, dx3,
          dx1_prev, dx2_prev, dx3_prev,
          ux1, ux2, ux3,
          phi, weight, tag,
          metric, charge, dt));
      Kokkos::Experimental::contribute(J_flat, J_scat);
      Kokkos::fence("flat deposit done");
    }

    // Run the tiled kernel. Build a TileLayout with one alive particle
    // landing in its expected tile (sort key = min(i, i_prev) / T_TILE).
    ndfield_t<Dim::_2D, 3> J_tiled { "J_tiled",
                                     nx1 + 2u * N_GHOSTS,
                                     nx2 + 2u * N_GHOSTS };
    {
      const auto sort_i1 = static_cast<int>(
        (i0 < i0f) ? i0 : i0f); // min(i, i_prev) before clamp
      const auto sort_i2 = static_cast<int>((j0 < j0f) ? j0 : j0f);
      const auto ntx1    = static_cast<ncells_t>(
        std::ceil(static_cast<double>(nx1) / static_cast<double>(T_TILE)));
      const auto ntx2 = static_cast<ncells_t>(
        std::ceil(static_cast<double>(nx2) / static_cast<double>(T_TILE)));
      const auto tx1 = static_cast<ncells_t>(sort_i1) / T_TILE;
      const auto tx2 = static_cast<ncells_t>(sort_i2) / T_TILE;

      TileLayout<Dim::_2D> layout;
      layout.ntiles_per_axis[0] = ntx1;
      layout.ntiles_per_axis[1] = ntx2;
      layout.ntiles_per_axis[2] = 1u;
      layout.ntiles_total       = ntx1 * ntx2;
      layout.tile_size          = T_TILE;
      layout.tile_offsets       = build_tile_offsets_single_particle(ntx1,
                                                                      ntx2,
                                                                      tx1,
                                                                      tx2);

      using kernel_t =
        kernel::DepositCurrents_kernel_tiled<SimEngine::SRPIC, metric_t, O, T_TILE>;
      kernel_t kern { J_tiled,
                      i1, i2, i3,
                      i1_prev, i2_prev, i3_prev,
                      dx1, dx2, dx3,
                      dx1_prev, dx2_prev, dx3_prev,
                      ux1, ux2, ux3,
                      phi, weight, tag,
                      metric, charge, dt, layout };

      Kokkos::TeamPolicy<> policy(static_cast<int>(layout.ntiles_total),
                                  Kokkos::AUTO);
      policy.set_scratch_size(0,
                              Kokkos::PerTeam(kernel_t::scratch_bytes()));
      Kokkos::parallel_for("TiledDeposit", policy, kern);
      Kokkos::fence("tiled deposit done");
    }

    // Compare J_flat vs J_tiled cell-by-cell.
    auto h_flat  = Kokkos::create_mirror_view(J_flat);
    auto h_tiled = Kokkos::create_mirror_view(J_tiled);
    Kokkos::deep_copy(h_flat, J_flat);
    Kokkos::deep_copy(h_tiled, J_tiled);

    const real_t eps      = static_cast<real_t>(1.0e-5);
    real_t       max_diff = ZERO;
    int          fail_count = 0;
    for (ncells_t i = 0; i < h_flat.extent(0); ++i) {
      for (ncells_t j = 0; j < h_flat.extent(1); ++j) {
        for (int c = 0; c < 3; ++c) {
          const real_t a    = h_flat(i, j, c);
          const real_t b    = h_tiled(i, j, c);
          const real_t diff = math::fabs(a - b);
          const real_t mag  = math::max(math::fabs(a), math::fabs(b));
          if (diff > max_diff) {
            max_diff = diff;
          }
          if (diff > eps * math::max(mag, static_cast<real_t>(1.0))) {
            if (fail_count < 5) {
              std::cerr << "  J(" << i << "," << j << ",c=" << c
                        << ") flat=" << a << " tiled=" << b
                        << " diff=" << diff << '\n';
            }
            ++fail_count;
          }
        }
      }
    }
    if (fail_count > 0) {
      std::cerr << "X-1 deposit_tiled equivalence FAILED for O=" << O
                << " T_TILE=" << T_TILE
                << " : " << fail_count << " mismatches; max_diff=" << max_diff
                << '\n';
      throw std::logic_error("DepositCurrents_kernel_tiled mismatch");
    }
    std::cerr << "X-1 deposit_tiled OK  O=" << O << " T_TILE=" << T_TILE
              << "  max_diff=" << max_diff << '\n';
  }

  template <unsigned short T_TILE>
  void run_all_orders() {
    run_one_case<0u, T_TILE>();
    run_one_case<1u, T_TILE>();
    run_one_case<2u, T_TILE>();
    run_one_case<3u, T_TILE>();
    run_one_case<4u, T_TILE>();
    run_one_case<5u, T_TILE>();
    run_one_case<6u, T_TILE>();
    run_one_case<7u, T_TILE>();
    run_one_case<8u, T_TILE>();
    run_one_case<9u, T_TILE>();
    run_one_case<10u, T_TILE>();
    run_one_case<11u, T_TILE>();
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    // Run with each tile-size choice from the validated CMake list.
    run_all_orders<4u>();
    run_all_orders<6u>();
    run_all_orders<8u>();
    run_all_orders<10u>();
    run_all_orders<12u>();
    run_all_orders<14u>();
    run_all_orders<16u>();
  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
