/**
 * @file tests/kernels/moments_tiled.cpp
 * @brief Numerical-equivalence + conservation test for the tiled hybrid
 *        push + moment deposit.
 *
 * Runs the flat (`kernel::hybrid::Pusher_kernel` over a ScatterView) and
 * tiled (`kernel::TiledScatter_kernel` harness with the same kernel as
 * body) paths on identical particle SoA inputs and asserts:
 *   - the resulting aux moments (V -> comps 0..2, N -> comp 3) agree
 *     cell-by-cell within a small floating-point tolerance;
 *   - aux comps 4..5 are untouched by the tiled flush (NC = 4 < NG = 6);
 *   - exact conservation: sum_cells N and sum_cells V match the
 *     analytically summed particle weights (interior particles, so the
 *     shape function partitions unity);
 *   - the per-particle escape valve (maximally-stale tile layout) and the
 *     partitioned-prefix + flat-tail composition both reproduce the flat
 *     reference.
 *
 * Modes covered: MomentsOnly (stored positions, no push) and Predictor
 * (in-register push with zero E/B — velocities unchanged, positions
 * drift by dt*v — exercising the post-push footprint select without
 * mutating the SoA, so flat and tiled share one particle set).
 *
 * Built only when `team_policy=ON` (`-D TEAM_POLICY` defined).
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include "kernels/hybrid/pusher.hpp"
#include "kernels/tiled_scatter.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using namespace ntt;

  // Pack the per-test SoA arrays into a ParticleArrays — the struct the
  // pusher kernel takes. Payload (pld_*) members stay default; these are
  // 2D Cartesian Minkowski cases, so phi/i3/dx3 are present but unread.
  ParticleArrays pack_arrays(const array_t<int*>&      i1,
                             const array_t<int*>&      i2,
                             const array_t<int*>&      i3,
                             const array_t<int*>&      i1_prev,
                             const array_t<int*>&      i2_prev,
                             const array_t<int*>&      i3_prev,
                             const array_t<prtldx_t*>& dx1,
                             const array_t<prtldx_t*>& dx2,
                             const array_t<prtldx_t*>& dx3,
                             const array_t<prtldx_t*>& dx1_prev,
                             const array_t<prtldx_t*>& dx2_prev,
                             const array_t<prtldx_t*>& dx3_prev,
                             const array_t<real_t*>&   ux1,
                             const array_t<real_t*>&   ux2,
                             const array_t<real_t*>&   ux3,
                             const array_t<real_t*>&   phi,
                             const array_t<real_t*>&   weight,
                             const array_t<short*>&    tag) {
    ParticleArrays pa;
    pa.i1 = i1, pa.i2 = i2, pa.i3 = i3;
    pa.i1_prev = i1_prev, pa.i2_prev = i2_prev, pa.i3_prev = i3_prev;
    pa.dx1 = dx1, pa.dx2 = dx2, pa.dx3 = dx3;
    pa.dx1_prev = dx1_prev, pa.dx2_prev = dx2_prev, pa.dx3_prev = dx3_prev;
    pa.ux1 = ux1, pa.ux2 = ux2, pa.ux3 = ux3;
    pa.phi = phi, pa.weight = weight, pa.tag = tag;
    return pa;
  }

  // Host mirror of one particle's state, used to build the SoA and the
  // analytic conservation sums.
  struct TestPrtl {
    int    i1, i2;
    real_t dx1, dx2;
    real_t ux1, ux2, ux3;
    real_t weight;
  };

  // Cell-by-cell comparison of the deposited moments (comps 0..3); throws
  // on mismatch. Also asserts comps 4..5 of `aux_tiled` are exactly zero
  // (the tiled flush must not touch components >= NC).
  void compare_aux(const ndfield_t<Dim::_2D, 6>& aux_flat,
                   const ndfield_t<Dim::_2D, 6>& aux_tiled,
                   unsigned short                T_TILE,
                   const char*                   label) {
    auto h_flat  = Kokkos::create_mirror_view(aux_flat);
    auto h_tiled = Kokkos::create_mirror_view(aux_tiled);
    Kokkos::deep_copy(h_flat, aux_flat);
    Kokkos::deep_copy(h_tiled, aux_tiled);

    const real_t eps        = static_cast<real_t>(1.0e-5);
    real_t       max_diff   = ZERO;
    int          fail_count = 0;
    for (ncells_t i = 0; i < h_flat.extent(0); ++i) {
      for (ncells_t j = 0; j < h_flat.extent(1); ++j) {
        for (int c = 0; c < 4; ++c) {
          const real_t a    = h_flat(i, j, c);
          const real_t b    = h_tiled(i, j, c);
          const real_t diff = math::fabs(a - b);
          const real_t mag  = math::max(math::fabs(a), math::fabs(b));
          if (diff > max_diff) {
            max_diff = diff;
          }
          if (diff > eps * math::max(mag, static_cast<real_t>(1.0))) {
            if (fail_count < 5) {
              std::cerr << "  [" << label << "] aux(" << i << "," << j
                        << ",c=" << c << ") flat=" << a << " tiled=" << b
                        << " diff=" << diff << '\n';
            }
            ++fail_count;
          }
        }
        for (int c = 4; c < 6; ++c) {
          if (h_tiled(i, j, c) != ZERO) {
            if (fail_count < 5) {
              std::cerr << "  [" << label << "] aux(" << i << "," << j
                        << ",c=" << c << ") = " << h_tiled(i, j, c)
                        << " != 0 (comp >= NC must stay untouched)\n";
            }
            ++fail_count;
          }
        }
      }
    }
    if (fail_count > 0) {
      std::cerr << "moments_tiled[" << label << "] FAILED for T_TILE=" << T_TILE
                << " : " << fail_count << " mismatches; max_diff=" << max_diff
                << '\n';
      throw std::logic_error("tiled hybrid moment deposit mismatch");
    }
    std::cerr << "moments_tiled[" << label << "] OK  T_TILE=" << T_TILE
              << "  max_diff=" << max_diff << '\n';
  }

  // Conservation: the shape function partitions unity for particles whose
  // full stencil lies inside the storage array, so
  //   sum_cells N   = sum_p inv_n0 / sqrt_det_h * w_p
  //   sum_cells V_c = sum_p inv_n0 / sqrt_det_h * w_p * mass * v_c(p)
  // (sqrt_det_h is uniform for Minkowski). Accumulated in double.
  void check_conservation(const ndfield_t<Dim::_2D, 6>& aux,
                          const std::vector<TestPrtl>&  prtls,
                          double                        w_norm, // inv_n0/sqrt_det_h
                          double                        mass,
                          unsigned short                T_TILE,
                          const char*                   label) {
    auto h = Kokkos::create_mirror_view(aux);
    Kokkos::deep_copy(h, aux);

    double got[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (ncells_t i = 0; i < h.extent(0); ++i) {
      for (ncells_t j = 0; j < h.extent(1); ++j) {
        for (int c = 0; c < 4; ++c) {
          got[c] += static_cast<double>(h(i, j, c));
        }
      }
    }
    double expect[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (const auto& p : prtls) {
      const double w = w_norm * static_cast<double>(p.weight);
      expect[0] += w * mass * static_cast<double>(p.ux1);
      expect[1] += w * mass * static_cast<double>(p.ux2);
      expect[2] += w * mass * static_cast<double>(p.ux3);
      expect[3] += w;
    }
    bool failed = false;
    for (int c = 0; c < 4; ++c) {
      const double scale = std::max({ std::fabs(expect[c]), std::fabs(got[c]), 1.0 });
      if (std::fabs(got[c] - expect[c]) > 1.0e-5 * scale) {
        failed = true;
        std::cerr << "moments_tiled[" << label << "] NON-CONSERVED comp " << c
                  << " for T_TILE=" << T_TILE << " : sum=" << got[c]
                  << " expected=" << expect[c]
                  << " ratio=" << (got[c] / expect[c]) << '\n';
      }
    }
    if (failed) {
      throw std::logic_error("tiled hybrid moment deposit non-conservation");
    }
    std::cerr << "moments_tiled[" << label << "] conserved  T_TILE=" << T_TILE
              << '\n';
  }

  /**
   * One test case.
   *
   * @tparam T_TILE tile edge length
   * @tparam Mode   MomentsOnly or Predictor
   * @param stale_layout  bucket ALL partitioned particles into tile 0 (the
   *                      maximally-stale layout): every particle off tile 0
   *                      must take the per-particle escape valve.
   * @param n_partitioned how many of the alive particles the tile layout
   *                      partitions; the remainder [n_partitioned, n_alive)
   *                      is deposited by the flat tail pass, emulating the
   *                      launcher's appended-particles handling.
   */
  template <unsigned short T_TILE, kernel::hybrid::PushMode Mode>
  void run_case(bool stale_layout, int n_partitioned, const char* label) {
    using metric_t = metric::Minkowski<Dim::_2D>;
    using kernel_t = kernel::hybrid::Pusher_kernel<metric_t, Mode>;

    constexpr ncells_t nx1 = 50u, nx2 = 50u;
    metric_t metric { { nx1, nx2 }, { { 0.0, 55.0 }, { 0.0, 55.0 } }, {} };
    // Minkowski: sqrt_det_h = dx^2, dx = 55/50
    const double dxc         = 55.0 / 50.0;
    const double sqrt_det_h  = dxc * dxc;

    // interior particle lattice (margins >= 4 cells so any build's
    // window + the tiny Predictor drift stays inside the storage array —
    // required for the exact conservation check)
    constexpr int n_base          = 5;
    const int     bases[n_base]   = { 4, 13, 25, 37, 44 };
    constexpr int n_alive         = n_base * n_base; // 25
    constexpr int n_slots         = 32;              // dead tail beyond n_alive

    std::vector<TestPrtl> prtls;
    prtls.reserve(n_alive);
    int p = 0;
    for (int a = 0; a < n_base; ++a) {
      for (int b = 0; b < n_base; ++b, ++p) {
        TestPrtl tp;
        tp.i1     = bases[a];
        tp.i2     = bases[b];
        tp.dx1    = static_cast<real_t>(0.15 + 0.028 * p);
        tp.dx2    = static_cast<real_t>(0.85 - 0.026 * p);
        tp.ux1    = static_cast<real_t>(0.9 - 0.07 * p);
        tp.ux2    = static_cast<real_t>(-0.8 + 0.06 * p);
        tp.ux3    = static_cast<real_t>(2.5 - 0.1 * a);
        tp.weight = static_cast<real_t>(0.5 + 0.05 * p);
        prtls.push_back(tp);
      }
    }
    // NOTE: iterating a-major, b-minor makes (i1/T, i2/T) lexicographically
    // nondecreasing, i.e. the natural order is already tile-sorted for the
    // row-major tile index tx1 * ntx2 + tx2 — as SortSpatially guarantees.

    array_t<int*>      i1 { "i1", n_slots }, i2 { "i2", n_slots },
      i3 { "i3", n_slots };
    array_t<int*>      i1_prev { "i1_prev", n_slots },
      i2_prev { "i2_prev", n_slots }, i3_prev { "i3_prev", n_slots };
    array_t<prtldx_t*> dx1 { "dx1", n_slots }, dx2 { "dx2", n_slots },
      dx3 { "dx3", n_slots };
    array_t<prtldx_t*> dx1_prev { "dx1_prev", n_slots },
      dx2_prev { "dx2_prev", n_slots }, dx3_prev { "dx3_prev", n_slots };
    array_t<real_t*>   ux1 { "ux1", n_slots }, ux2 { "ux2", n_slots },
      ux3 { "ux3", n_slots };
    array_t<real_t*>   phi { "phi", n_slots }, weight { "weight", n_slots };
    array_t<short*>    tag { "tag", n_slots };

    {
      auto h_i1  = Kokkos::create_mirror_view(i1);
      auto h_i2  = Kokkos::create_mirror_view(i2);
      auto h_dx1 = Kokkos::create_mirror_view(dx1);
      auto h_dx2 = Kokkos::create_mirror_view(dx2);
      auto h_ux1 = Kokkos::create_mirror_view(ux1);
      auto h_ux2 = Kokkos::create_mirror_view(ux2);
      auto h_ux3 = Kokkos::create_mirror_view(ux3);
      auto h_w   = Kokkos::create_mirror_view(weight);
      auto h_tag = Kokkos::create_mirror_view(tag);
      for (int q = 0; q < n_alive; ++q) {
        h_i1(q)  = prtls[q].i1;
        h_i2(q)  = prtls[q].i2;
        h_dx1(q) = static_cast<prtldx_t>(prtls[q].dx1);
        h_dx2(q) = static_cast<prtldx_t>(prtls[q].dx2);
        h_ux1(q) = prtls[q].ux1;
        h_ux2(q) = prtls[q].ux2;
        h_ux3(q) = prtls[q].ux3;
        h_w(q)   = prtls[q].weight;
        h_tag(q) = ParticleTag::alive;
      }
      // slots >= n_alive stay zero == dead
      Kokkos::deep_copy(i1, h_i1);
      Kokkos::deep_copy(i2, h_i2);
      Kokkos::deep_copy(dx1, h_dx1);
      Kokkos::deep_copy(dx2, h_dx2);
      Kokkos::deep_copy(ux1, h_ux1);
      Kokkos::deep_copy(ux2, h_ux2);
      Kokkos::deep_copy(ux3, h_ux3);
      Kokkos::deep_copy(weight, h_w);
      Kokkos::deep_copy(tag, h_tag);
    }
    auto arrays = pack_arrays(i1, i2, i3,
                              i1_prev, i2_prev, i3_prev,
                              dx1, dx2, dx3,
                              dx1_prev, dx2_prev, dx3_prev,
                              ux1, ux2, ux3,
                              phi, weight, tag);

    // context: non-trivial mass / inv_n0 / weights; dt small enough that a
    // Predictor push drifts every particle by << 1 cell (stays interior)
    const float  mass = 1.5f, charge = 1.0f;
    const real_t dt = 0.01, omegaB0 = 0.8, inv_n0 = 0.7;
    const kernel::hybrid::PusherContext ctx { mass,
                                              charge,
                                              dt,
                                              omegaB0,
                                              inv_n0,
                                              /* use_weights */ true,
                                              static_cast<int>(nx1),
                                              static_cast<int>(nx2),
                                              1 };
    const boundaries_t<PrtlBC> prtl_bc {
      { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
      { PrtlBC::PERIODIC, PrtlBC::PERIODIC }
    };
    const kernel::sr::PusherBoundaries<Dim::_2D> bc { prtl_bc };

    // zero E/B: the Predictor push leaves v unchanged and drifts x by dt*v
    ndfield_t<Dim::_2D, 6> EB { "EB", nx1 + 2u * N_GHOSTS, nx2 + 2u * N_GHOSTS };

    // ---------------- flat reference (all alive particles) ----------------
    ndfield_t<Dim::_2D, 6> aux_flat { "aux_flat",
                                      nx1 + 2u * N_GHOSTS,
                                      nx2 + 2u * N_GHOSTS };
    {
      auto scatter_aux = Kokkos::Experimental::create_scatter_view(aux_flat);
      Kokkos::parallel_for(
        "FlatPushDeposit",
        n_slots,
        kernel_t { ctx, bc, arrays, EB, scatter_aux, metric });
      Kokkos::Experimental::contribute(aux_flat, scatter_aux);
      Kokkos::fence("flat done");
    }

    // ---------------- tiled (+ flat tail over [n_partitioned, n_alive)) ----
    ndfield_t<Dim::_2D, 6> aux_tiled { "aux_tiled",
                                       nx1 + 2u * N_GHOSTS,
                                       nx2 + 2u * N_GHOSTS };
    {
      const auto ntx1 = static_cast<ncells_t>(
        std::ceil(static_cast<double>(nx1) / static_cast<double>(T_TILE)));
      const auto ntx2 = static_cast<ncells_t>(
        std::ceil(static_cast<double>(nx2) / static_cast<double>(T_TILE)));
      const auto total_tiles = ntx1 * ntx2;

      TileLayout<Dim::_2D> layout;
      layout.ntiles_per_axis[0] = ntx1;
      layout.ntiles_per_axis[1] = ntx2;
      layout.ntiles_per_axis[2] = 1u;
      layout.ntiles_total       = total_tiles;
      layout.tile_size          = T_TILE;
      layout.npart_partitioned  = static_cast<npart_t>(n_partitioned);
      layout.tile_offsets = array_t<npart_t*> { "tile_offsets",
                                                total_tiles + 1u };
      {
        auto h = Kokkos::create_mirror_view(layout.tile_offsets);
        if (stale_layout) {
          // all partitioned particles in tile 0 — maximally stale
          h(0) = 0u;
          for (ncells_t t = 1; t <= total_tiles; ++t) {
            h(t) = static_cast<npart_t>(n_partitioned);
          }
        } else {
          // proper binning by (i1/T, i2/T) of the (tile-sorted) prefix
          for (ncells_t t = 0; t <= total_tiles; ++t) {
            h(t) = 0u;
          }
          for (int q = 0; q < n_partitioned; ++q) {
            const auto tid = static_cast<ncells_t>(prtls[q].i1 / T_TILE) * ntx2 +
                             static_cast<ncells_t>(prtls[q].i2 / T_TILE);
            h(tid + 1u) += 1u;
          }
          for (ncells_t t = 1; t <= total_tiles; ++t) {
            h(t) += h(t - 1u);
          }
        }
        Kokkos::deep_copy(layout.tile_offsets, h);
      }

      using tiled_t = kernel::TiledScatter_kernel<Dim::_2D,
                                                  4,
                                                  6,
                                                  kernel_t::window,
                                                  T_TILE,
                                                  kernel_t>;
      const kernel_t body { ctx, bc, arrays, EB, metric };
      const tiled_t  kern { aux_tiled,
                            body,
                            layout,
                            static_cast<npart_t>(n_alive) };
      const auto     policy = kernel::MakeTiledPolicy(kern, total_tiles, 0);
      Kokkos::parallel_for("TiledPushDeposit", policy, kern);
      Kokkos::fence("tiled done");

      if (n_partitioned < n_alive) {
        // flat tail pass, exactly as the launcher composes it
        auto scatter_aux = Kokkos::Experimental::create_scatter_view(aux_tiled);
        Kokkos::parallel_for(
          "TiledPushDepositTail",
          CreateParticleRangePolicy<Dim::_1D>(
            { static_cast<npart_t>(n_partitioned) },
            { static_cast<npart_t>(n_alive) }),
          kernel_t { ctx, bc, arrays, EB, scatter_aux, metric });
        Kokkos::Experimental::contribute(aux_tiled, scatter_aux);
        Kokkos::fence("tail done");
      }
    }

    compare_aux(aux_flat, aux_tiled, T_TILE, label);
    check_conservation(aux_tiled,
                       prtls,
                       static_cast<double>(inv_n0) / sqrt_det_h,
                       static_cast<double>(mass),
                       T_TILE,
                       label);
  }

  template <unsigned short T_TILE>
  void run_all() {
    constexpr int n_alive = 25;
    using kernel::hybrid::PushMode;
    // in-tile fast path
    run_case<T_TILE, PushMode::MomentsOnly>(false, n_alive, "moments");
    run_case<T_TILE, PushMode::Predictor>(false, n_alive, "predictor");
    // per-particle escape valve (maximally-stale layout)
    run_case<T_TILE, PushMode::MomentsOnly>(true, n_alive, "moments/stale");
    run_case<T_TILE, PushMode::Predictor>(true, n_alive, "predictor/stale");
    // appended-particles tail: tiled prefix + flat tail == flat all
    run_case<T_TILE, PushMode::MomentsOnly>(false, 15, "moments/tail");
    run_case<T_TILE, PushMode::Predictor>(false, 15, "predictor/tail");
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    run_all<4u>();
    run_all<8u>();
    run_all<16u>();
  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
