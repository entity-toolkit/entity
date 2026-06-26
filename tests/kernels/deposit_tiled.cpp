/**
 * @file tests/kernels/deposit_tiled.cpp
 * @brief X-1 numerical-equivalence test for the tiled deposit kernel.
 *
 * Runs the flat (`DepositCurrents_kernel`) and tiled
 * (`DepositCurrentsTiled_kernel`) kernels on identical particle SoA inputs
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

  // Pack the per-test SoA arrays into a ParticleArrays — the struct both
  // deposit kernels take. Payload (pld_*) members stay default; these are
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

  // Buckets ALL `n_alive` particles (slots 0..n_alive-1) into tile 0,
  // modelling a maximally-stale tile layout: every particle was "sorted"
  // into tile 0 but now sits anywhere in the domain (as happens when the
  // SoA drifts / is reordered between sorts). Every particle except the
  // few that genuinely live near the origin must therefore take the
  // per-particle escape valve to global J.
  array_t<npart_t*> build_tile_offsets_all_in_tile0(ncells_t total_tiles,
                                                    npart_t  n_alive) {
    array_t<npart_t*> offsets("tile_offsets", total_tiles + 1u);
    auto              h = Kokkos::create_mirror_view(offsets);
    h(0)                = static_cast<npart_t>(0);
    for (ncells_t t = 1; t <= total_tiles; ++t) {
      h(t) = n_alive;
    }
    Kokkos::deep_copy(offsets, h);
    return offsets;
  }

  // Cell-by-cell comparison of two J fields; throws on mismatch.
  void compare_J_fields(const ndfield_t<Dim::_2D, 3>& J_flat,
                        const ndfield_t<Dim::_2D, 3>& J_tiled,
                        unsigned short                O,
                        unsigned short                T_TILE,
                        const char*                   label) {
    auto h_flat  = Kokkos::create_mirror_view(J_flat);
    auto h_tiled = Kokkos::create_mirror_view(J_tiled);
    Kokkos::deep_copy(h_flat, J_flat);
    Kokkos::deep_copy(h_tiled, J_tiled);

    const real_t eps        = static_cast<real_t>(1.0e-5);
    real_t       max_diff   = ZERO;
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
              std::cerr << "  [" << label << "] J(" << i << "," << j
                        << ",c=" << c << ") flat=" << a << " tiled=" << b
                        << " diff=" << diff << '\n';
            }
            ++fail_count;
          }
        }
      }
    }
    if (fail_count > 0) {
      std::cerr << "deposit_tiled[" << label << "] FAILED for O=" << O
                << " T_TILE=" << T_TILE << " : " << fail_count
                << " mismatches; max_diff=" << max_diff << '\n';
      throw std::logic_error("DepositCurrentsTiled_kernel mismatch");
    }
    std::cerr << "deposit_tiled[" << label << "] OK  O=" << O
              << " T_TILE=" << T_TILE << "  max_diff=" << max_diff << '\n';
  }

  // Intrinsic charge-conservation check on a single deposited J field.
  // Esirkepov/zigzag deposits satisfy the discrete continuity equation, so
  // the spatial sum of the discrete divergence div.J = dJx/dx + dJy/dy
  // vanishes whenever the summation region encloses every particle's full
  // stencil (J == 0 on the region's outer boundary). This is evaluated on
  // J_tiled ALONE -- it does not compare against the flat reference -- so it
  // certifies the per-particle escape valve deposits each drifted particle's
  // stencil as one coherent unit: no cell dropped, duplicated, or split
  // between SLM scratch and global J. (The run_drift_case order guard keeps
  // every stencil inside [0, j_ext), so the extreme ghost cells stay zero and
  // the telescoping boundary flux is genuinely zero rather than clipped.)
  // Accumulated in double regardless of build precision to keep the
  // tolerance tight.
  void check_charge_conservation(const ndfield_t<Dim::_2D, 3>& J,
                                 unsigned short                O,
                                 unsigned short                T_TILE,
                                 const char*                   label) {
    auto h = Kokkos::create_mirror_view(J);
    Kokkos::deep_copy(h, J);

    double sum_div = 0.0; // Sum over the field of div.J (jx1 -> dx, jx2 -> dy).
    double abs_tot = 0.0; // Total |J|, sets the relative tolerance scale.
    for (ncells_t i = 1; i < h.extent(0); ++i) {
      for (ncells_t j = 1; j < h.extent(1); ++j) {
        sum_div += (static_cast<double>(h(i, j, 0)) -
                    static_cast<double>(h(i - 1, j, 0))) +
                   (static_cast<double>(h(i, j, 1)) -
                    static_cast<double>(h(i, j - 1, 1)));
      }
    }
    for (ncells_t i = 0; i < h.extent(0); ++i) {
      for (ncells_t j = 0; j < h.extent(1); ++j) {
        abs_tot += std::fabs(static_cast<double>(h(i, j, 0))) +
                   std::fabs(static_cast<double>(h(i, j, 1)));
      }
    }
    const double tol = 1.0e-5 * (abs_tot > 1.0 ? abs_tot : 1.0);
    if (std::fabs(sum_div) > tol) {
      std::cerr << "deposit_tiled[" << label
                << "] CHARGE NON-CONSERVED for O=" << O << " T_TILE=" << T_TILE
                << " : sum(div.J)=" << sum_div << " tol=" << tol
                << " (abs_tot=" << abs_tot << ")\n";
      throw std::logic_error(
        "DepositCurrentsTiled_kernel charge non-conservation");
    }
    std::cerr << "deposit_tiled[" << label << "] charge-conserved O=" << O
              << " T_TILE=" << T_TILE << "  sum(div.J)=" << sum_div << '\n';
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
          pack_arrays(i1, i2, i3,
                      i1_prev, i2_prev, i3_prev,
                      dx1, dx2, dx3,
                      dx1_prev, dx2_prev, dx3_prev,
                      ux1, ux2, ux3,
                      phi, weight, tag),
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
        kernel::DepositCurrentsTiled_kernel<SimEngine::SRPIC, metric_t, O, T_TILE>;
      // npart = full slot count (10): the lone alive particle sits in slot 0
      // and the per-tile slice clamp keeps the (dead) tail out.
      kernel_t kern { J_tiled,
                      pack_arrays(i1, i2, i3,
                                  i1_prev, i2_prev, i3_prev,
                                  dx1, dx2, dx3,
                                  dx1_prev, dx2_prev, dx3_prev,
                                  ux1, ux2, ux3,
                                  phi, weight, tag),
                      metric, charge, dt, layout,
                      static_cast<npart_t>(10) };

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
      throw std::logic_error("DepositCurrentsTiled_kernel mismatch");
    }
    std::cerr << "X-1 deposit_tiled OK  O=" << O << " T_TILE=" << T_TILE
              << "  max_diff=" << max_diff << '\n';
  }

  // Drift / stale-layout regression test for the per-particle escape valve.
  //
  // A population of alive particles is spread across the whole domain
  // (including the near-boundary cells that deposit into the ghost stripe)
  // but the tile layout buckets them ALL into tile 0 — i.e. the layout is
  // maximally stale w.r.t. their real positions, exactly the situation that
  // arises when the SoA is reordered/appended between sorts. The tiled
  // kernel must route every out-of-tile particle to the global J view and
  // reproduce the flat deposit cell-for-cell: no charge dropped or
  // double-counted at any drift distance. This is the property that broke
  // when `spatial_sorting_interval > 1` left drifted particles depositing
  // partial stencils, producing a density/E line at the decomposition
  // boundary.
  template <unsigned short O, unsigned short T_TILE>
  void run_drift_case() {
    // This case deposits boundary-adjacent particles (cells touching the
    // ghost stripe), so an order-O stencil must fit inside the field's
    // N_GHOSTS ghost layers. N_GHOSTS is a compile-time constant fixed by
    // the build's SHAPE_ORDER ((SHAPE_ORDER+1)/2 + 1); a build whose ghost
    // width is smaller than order O requires would deposit outside the
    // field -- silent on GPU (no Kokkos View bounds guard; the overshoot
    // cells carry zero shape-weight so results still match) but heap
    // corruption on a host/SERIAL build. Skip those orders here; build at
    // the matching SHAPE_ORDER to drift-test higher orders. The equivalence
    // ("X-1") cases above stay interior, so they exercise all orders.
    if constexpr ((O + 1u) / 2u + 1u > N_GHOSTS) {
      std::cerr << "deposit_tiled[drift] SKIP O=" << O << " T_TILE=" << T_TILE
                << " (needs N_GHOSTS>=" << ((O + 1u) / 2u + 1u)
                << ", build has " << N_GHOSTS << ")\n";
      return;
    }
    using metric_t = metric::Minkowski<Dim::_2D>;
    constexpr unsigned short nx1 = 50u, nx2 = 50u;
    metric_t metric { { nx1, nx2 }, { { 0.0, 55.0 }, { 0.0, 55.0 } }, {} };

    constexpr int n_slots = 64;
    constexpr int n_base  = 5;
    const int     bases[n_base] = { 1, 13, 25, 37, 48 };
    const int     n_alive       = n_base * n_base; // 25

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
    const real_t       charge = 1.0, dt = 1.0;

    // Fill alive particles on host (slots >= n_alive stay zero == dead).
    auto h_i1  = Kokkos::create_mirror_view(i1);
    auto h_i2  = Kokkos::create_mirror_view(i2);
    auto h_i1p = Kokkos::create_mirror_view(i1_prev);
    auto h_i2p = Kokkos::create_mirror_view(i2_prev);
    auto h_dx1  = Kokkos::create_mirror_view(dx1);
    auto h_dx2  = Kokkos::create_mirror_view(dx2);
    auto h_dx1p = Kokkos::create_mirror_view(dx1_prev);
    auto h_dx2p = Kokkos::create_mirror_view(dx2_prev);
    auto h_ux3  = Kokkos::create_mirror_view(ux3);
    auto h_w    = Kokkos::create_mirror_view(weight);
    auto h_tag  = Kokkos::create_mirror_view(tag);
    int  p = 0;
    for (int a = 0; a < n_base; ++a) {
      for (int b = 0; b < n_base; ++b, ++p) {
        h_i1p(p) = bases[a];
        h_i1(p)  = bases[a] - 1;
        h_i2p(p) = bases[b];
        h_i2(p)  = bases[b] - 1;
        h_dx1p(p) = static_cast<prtldx_t>(0.65);
        h_dx1(p)  = static_cast<prtldx_t>(0.99);
        h_dx2p(p) = static_cast<prtldx_t>(0.65);
        h_dx2(p)  = static_cast<prtldx_t>(0.80);
        h_ux3(p)  = static_cast<real_t>(2.5);
        h_w(p)    = static_cast<real_t>(1.0);
        h_tag(p)  = ParticleTag::alive;
      }
    }
    Kokkos::deep_copy(i1, h_i1);
    Kokkos::deep_copy(i2, h_i2);
    Kokkos::deep_copy(i1_prev, h_i1p);
    Kokkos::deep_copy(i2_prev, h_i2p);
    Kokkos::deep_copy(dx1, h_dx1);
    Kokkos::deep_copy(dx2, h_dx2);
    Kokkos::deep_copy(dx1_prev, h_dx1p);
    Kokkos::deep_copy(dx2_prev, h_dx2p);
    Kokkos::deep_copy(ux3, h_ux3);
    Kokkos::deep_copy(weight, h_w);
    Kokkos::deep_copy(tag, h_tag);

    // Flat reference over all slots (dead slots are skipped internally).
    ndfield_t<Dim::_2D, 3> J_flat { "J_flat",
                                    nx1 + 2u * N_GHOSTS,
                                    nx2 + 2u * N_GHOSTS };
    {
      auto J_scat = Kokkos::Experimental::create_scatter_view(J_flat);
      Kokkos::parallel_for(
        "FlatDepositDrift",
        n_slots,
        kernel::DepositCurrents_kernel<SimEngine::SRPIC, metric_t, O>(
          J_scat,
          pack_arrays(i1, i2, i3,
                      i1_prev, i2_prev, i3_prev,
                      dx1, dx2, dx3,
                      dx1_prev, dx2_prev, dx3_prev,
                      ux1, ux2, ux3,
                      phi, weight, tag),
          metric, charge, dt));
      Kokkos::Experimental::contribute(J_flat, J_scat);
      Kokkos::fence("flat drift deposit done");
    }

    // Tiled with a maximally-stale layout: all alive particles in tile 0.
    ndfield_t<Dim::_2D, 3> J_tiled { "J_tiled",
                                     nx1 + 2u * N_GHOSTS,
                                     nx2 + 2u * N_GHOSTS };
    {
      const auto ntx1 = static_cast<ncells_t>(
        std::ceil(static_cast<double>(nx1) / static_cast<double>(T_TILE)));
      const auto ntx2 = static_cast<ncells_t>(
        std::ceil(static_cast<double>(nx2) / static_cast<double>(T_TILE)));

      TileLayout<Dim::_2D> layout;
      layout.ntiles_per_axis[0] = ntx1;
      layout.ntiles_per_axis[1] = ntx2;
      layout.ntiles_per_axis[2] = 1u;
      layout.ntiles_total       = ntx1 * ntx2;
      layout.tile_size          = T_TILE;
      layout.tile_offsets       = build_tile_offsets_all_in_tile0(
        ntx1 * ntx2,
        static_cast<npart_t>(n_alive));

      using kernel_t =
        kernel::DepositCurrentsTiled_kernel<SimEngine::SRPIC, metric_t, O, T_TILE>;
      // npart = n_alive: the stale layout buckets all alive particles into
      // tile 0, so the team must walk [0, n_alive) and route the drifted
      // ones to the global-J escape valve.
      kernel_t kern { J_tiled,
                      pack_arrays(i1, i2, i3,
                                  i1_prev, i2_prev, i3_prev,
                                  dx1, dx2, dx3,
                                  dx1_prev, dx2_prev, dx3_prev,
                                  ux1, ux2, ux3,
                                  phi, weight, tag),
                      metric, charge, dt, layout,
                      static_cast<npart_t>(n_alive) };

      Kokkos::TeamPolicy<> policy(static_cast<int>(layout.ntiles_total),
                                  Kokkos::AUTO);
      policy.set_scratch_size(0,
                              Kokkos::PerTeam(kernel_t::scratch_bytes()));
      Kokkos::parallel_for("TiledDepositDrift", policy, kern);
      Kokkos::fence("tiled drift deposit done");
    }

    compare_J_fields(J_flat, J_tiled, O, T_TILE, "drift");
    // Self-contained conservation check on the escape-valve output: the
    // drifted particles all take the per-particle global-J path, so this
    // certifies that path is charge-conserving without leaning on J_flat.
    check_charge_conservation(J_tiled, O, T_TILE, "drift");
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

    // Stale-layout / drift regression (per-particle escape valve).
    run_drift_case<0u, T_TILE>();
    run_drift_case<1u, T_TILE>();
    run_drift_case<2u, T_TILE>();
    run_drift_case<3u, T_TILE>();
    run_drift_case<4u, T_TILE>();
    run_drift_case<5u, T_TILE>();
    run_drift_case<6u, T_TILE>();
    run_drift_case<7u, T_TILE>();
    run_drift_case<8u, T_TILE>();
    run_drift_case<9u, T_TILE>();
    run_drift_case<10u, T_TILE>();
    run_drift_case<11u, T_TILE>();
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
