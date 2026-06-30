/**
 * @file output/render/fieldlines.h
 * @brief Host-side magnetic-field-line tracer + bucketed tube-geometry builder
 * @implements
 *   - out::CoarseField
 *   - out::traceFieldLines
 *   - out::buildTubeSet
 *   - out::emptyTubeSet
 * @namespaces:
 *   - out::
 * @macros:
 *   - OUTPUT_ENABLED
 * @note
 * Field lines are intrinsically non-local (a streamline wanders across MPI
 * domains), which would normally demand parallel particle advection. We sidestep
 * that entirely: the (physical-basis) field is volume-averaged onto a COARSE
 * grid and MPI-replicated to every rank (see Metadomain::buildFieldLineTubes),
 * so every rank traces the SAME global polylines locally and renders only the
 * segments overlapping its own domain. The existing ordered cross-domain
 * composite then stitches the pieces. Tracing/geometry here is metric-agnostic:
 * the only supported 3D render mode is Cartesian (Minkowski), so the coarse grid
 * is a plain uniform lattice in world coordinates.
 *
 * Performance: a ray sample must not test every segment. Segments are bucketed
 * into the coarse grid (CSR), and since the tube radius is << one coarse cell,
 * a sample only needs the segments registered in its own cell. The kernel
 * (raymarch.hpp) walks that short bucket.
 */

#ifndef OUTPUT_RENDER_FIELDLINES_H
#define OUTPUT_RENDER_FIELDLINES_H

#include "global.h"

#include "arch/kokkos_aliases.h"

#include "output/render/renderer.h"
#include "output/render/transfer_fn.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace out {

  /**
   * @brief A coarse, MPI-replicated copy of the physical-basis vector field.
   * @note B is laid out component-fastest: index (c0,c1,c2,comp) lives at
   * ((c2*n1 + c1)*n0 + c0)*3 + comp, with c0 the fastest spatial axis.
   */
  struct CoarseField {
    std::vector<real_t> B;                 // n0*n1*n2*3
    int                 n[3] { 0, 0, 0 };
    real_t              origin[3] { ZERO, ZERO, ZERO };
    real_t              dx[3] { ONE, ONE, ONE };
  };

  /** @brief One traced field line: world-space vertices + per-vertex |field|. */
  struct Polyline {
    std::vector<std::array<real_t, 3>> pts;
    std::vector<real_t>                scal;
  };

  namespace fl_hidden {

    inline auto cellLinear(const CoarseField& cf, int c0, int c1, int c2)
      -> std::size_t {
      return (static_cast<std::size_t>(c2) * cf.n[1] + c1) * cf.n[0] + c0;
    }

    // trilinear sample of the coarse field at world point p -> B[3], |B|.
    // Clamps to the grid (so a sample just outside a face still returns the edge
    // value); membership in the global box is the caller's stop test.
    inline auto sampleCoarse(const CoarseField& cf, const real_t p[3], real_t B[3])
      -> real_t {
      int    i0[3], i1[3];
      real_t fr[3];
      for (int d = 0; d < 3; ++d) {
        if (cf.n[d] <= 1) {
          i0[d] = 0;
          i1[d] = 0;
          fr[d] = ZERO;
          continue;
        }
        const real_t g = (p[d] - cf.origin[d]) / cf.dx[d] - HALF;
        real_t       f = std::floor(g);
        real_t       t = g - f;
        int          b = static_cast<int>(f);
        if (b < 0) {
          b = 0;
          t = ZERO;
        } else if (b > cf.n[d] - 2) {
          b = cf.n[d] - 2;
          t = ONE;
        }
        i0[d] = b;
        i1[d] = b + 1;
        fr[d] = t;
      }
      for (int comp = 0; comp < 3; ++comp) {
        const real_t c000 = cf.B[cellLinear(cf, i0[0], i0[1], i0[2]) * 3 + comp];
        const real_t c100 = cf.B[cellLinear(cf, i1[0], i0[1], i0[2]) * 3 + comp];
        const real_t c010 = cf.B[cellLinear(cf, i0[0], i1[1], i0[2]) * 3 + comp];
        const real_t c110 = cf.B[cellLinear(cf, i1[0], i1[1], i0[2]) * 3 + comp];
        const real_t c001 = cf.B[cellLinear(cf, i0[0], i0[1], i1[2]) * 3 + comp];
        const real_t c101 = cf.B[cellLinear(cf, i1[0], i0[1], i1[2]) * 3 + comp];
        const real_t c011 = cf.B[cellLinear(cf, i0[0], i1[1], i1[2]) * 3 + comp];
        const real_t c111 = cf.B[cellLinear(cf, i1[0], i1[1], i1[2]) * 3 + comp];
        const real_t c00  = c000 * (ONE - fr[0]) + c100 * fr[0];
        const real_t c10  = c010 * (ONE - fr[0]) + c110 * fr[0];
        const real_t c01  = c001 * (ONE - fr[0]) + c101 * fr[0];
        const real_t c11  = c011 * (ONE - fr[0]) + c111 * fr[0];
        const real_t c0   = c00 * (ONE - fr[1]) + c10 * fr[1];
        const real_t c1   = c01 * (ONE - fr[1]) + c11 * fr[1];
        B[comp]           = c0 * (ONE - fr[2]) + c1 * fr[2];
      }
      return std::sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2]);
    }

    inline auto insideBox(const CoarseField& cf, const real_t p[3]) -> bool {
      for (int d = 0; d < 3; ++d) {
        const real_t hi = cf.origin[d] + cf.n[d] * cf.dx[d];
        if (p[d] < cf.origin[d] or p[d] > hi) {
          return false;
        }
      }
      return true;
    }

  } // namespace fl_hidden

  /**
   * @brief Trace field lines through the coarse field by bidirectional RK4.
   * @param cf coarse, replicated physical field
   * @param cfg field-line configuration (seed density, step, caps)
   * @param world_per_pixel world units per screen pixel (sets seed/tube scale)
   * @param[out] out_vmin,out_vmax auto color range (min/max |field| along lines)
   * @return global polylines (identical on every rank)
   */
  inline auto traceFieldLines(const CoarseField&     cf,
                              const FieldLineConfig& cfg,
                              real_t                 world_per_pixel,
                              real_t&                out_vmin,
                              real_t&                out_vmax)
    -> std::vector<Polyline> {
    using fl_hidden::insideBox;
    using fl_hidden::sampleCoarse;

    std::vector<Polyline> lines;
    if (cf.n[0] < 1 or cf.n[1] < 1 or cf.n[2] < 1) {
      return lines;
    }

    real_t size[3];
    real_t diag2 = ZERO;
    real_t min_dx = static_cast<real_t>(1e30);
    for (int d = 0; d < 3; ++d) {
      size[d] = cf.n[d] * cf.dx[d];
      diag2   += size[d] * size[d];
      min_dx  = std::min(min_dx, cf.dx[d]);
    }
    const real_t box_diag = std::sqrt(diag2);
    const real_t max_len  = cfg.max_len_frac * box_diag;
    const real_t h        = std::max(cfg.step_frac, static_cast<real_t>(1e-3)) *
                     min_dx;
    const real_t eps = static_cast<real_t>(1e-20);

    // seed lattice: spacing ~ seed_px screen pixels, grown to respect seed_max
    real_t spacing = std::max(cfg.seed_px, ONE) * world_per_pixel;
    int    ns[3];
    auto   countSeeds = [&](real_t s) -> long {
      long tot = 1;
      for (int d = 0; d < 3; ++d) {
        ns[d]  = std::max(1, static_cast<int>(std::floor(size[d] / s)));
        tot   *= ns[d];
      }
      return tot;
    };
    long n_seed = countSeeds(spacing);
    if (n_seed > cfg.seed_max and cfg.seed_max > 0) {
      const real_t grow = std::cbrt(static_cast<real_t>(n_seed) /
                                    static_cast<real_t>(cfg.seed_max));
      spacing *= grow;
      countSeeds(spacing); // recompute ns[] for the grown spacing
    }

    // unit-vector field derivative (×dir) used by RK4; false if |B| ~ 0
    auto deriv = [&](const real_t p[3], real_t dir, real_t out[3]) -> bool {
      real_t       B[3];
      const real_t m = sampleCoarse(cf, p, B);
      if (m < eps) {
        return false;
      }
      const real_t inv = dir / m;
      out[0]           = B[0] * inv;
      out[1]           = B[1] * inv;
      out[2]           = B[2] * inv;
      return true;
    };

    out_vmin = static_cast<real_t>(1e30);
    out_vmax = static_cast<real_t>(-1e30);
    auto track = [&](real_t m) {
      out_vmin = std::min(out_vmin, m);
      out_vmax = std::max(out_vmax, m);
    };

    // integrate one direction (dir = +1 forward, -1 backward) from a seed
    auto integrate = [&](const real_t seed[3], real_t dir) {
      Polyline pl;
      real_t   p[3]   = { seed[0], seed[1], seed[2] };
      real_t   B0[3];
      real_t   m0 = sampleCoarse(cf, p, B0);
      if (m0 < eps) {
        return;
      }
      pl.pts.push_back({ p[0], p[1], p[2] });
      pl.scal.push_back(m0);
      track(m0);
      real_t len = ZERO;
      for (int step = 0; step < cfg.max_steps and len < max_len; ++step) {
        real_t k1[3], k2[3], k3[3], k4[3], q[3];
        if (not deriv(p, dir, k1)) {
          break;
        }
        for (int d = 0; d < 3; ++d) {
          q[d] = p[d] + HALF * h * k1[d];
        }
        if (not deriv(q, dir, k2)) {
          break;
        }
        for (int d = 0; d < 3; ++d) {
          q[d] = p[d] + HALF * h * k2[d];
        }
        if (not deriv(q, dir, k3)) {
          break;
        }
        for (int d = 0; d < 3; ++d) {
          q[d] = p[d] + h * k3[d];
        }
        if (not deriv(q, dir, k4)) {
          break;
        }
        for (int d = 0; d < 3; ++d) {
          p[d] += (h / static_cast<real_t>(6)) *
                  (k1[d] + static_cast<real_t>(2) * k2[d] +
                   static_cast<real_t>(2) * k3[d] + k4[d]);
        }
        if (not insideBox(cf, p)) {
          break;
        }
        real_t       B[3];
        const real_t m = sampleCoarse(cf, p, B);
        pl.pts.push_back({ p[0], p[1], p[2] });
        pl.scal.push_back(m);
        track(m);
        len += h;
      }
      if (pl.pts.size() >= 2) {
        lines.push_back(std::move(pl));
      }
    };

    for (int k = 0; k < ns[2]; ++k) {
      for (int j = 0; j < ns[1]; ++j) {
        for (int i = 0; i < ns[0]; ++i) {
          const real_t seed[3] = {
            cf.origin[0] + (static_cast<real_t>(i) + HALF) * size[0] / ns[0],
            cf.origin[1] + (static_cast<real_t>(j) + HALF) * size[1] / ns[1],
            cf.origin[2] + (static_cast<real_t>(k) + HALF) * size[2] / ns[2]
          };
          integrate(seed, ONE);
          integrate(seed, -ONE);
        }
      }
    }
    if (out_vmin > out_vmax) { // no lines traced
      out_vmin = ZERO;
      out_vmax = ONE;
    }
    return lines;
  }

  /**
   * @brief Bucket the field-line segments overlapping a domain AABB into a CSR
   * grid index and pack them into device Views ready for the ray-march kernel.
   * @param lines global polylines (every rank passes the same set)
   * @param radius world-space tube radius (the ds floor is applied by the caller)
   * @param cfg field-line configuration (colormap / log)
   * @param vmin,vmax tube color range
   * @param lo,hi this domain's world AABB (segments outside it are dropped)
   * @param cf coarse grid geometry, reused as the bucket grid
   * @param[out] n_kept number of segments kept for this domain (for logging)
   */
  inline auto buildTubeSet(const std::vector<Polyline>& lines,
                           real_t                       radius,
                           const FieldLineConfig&       cfg,
                           real_t                       vmin,
                           real_t                       vmax,
                           const real_t                 lo[3],
                           const real_t                 hi[3],
                           const CoarseField&           cf,
                           std::size_t&                 n_kept) -> TubeSet {
    TubeSet ts;
    ts.radius    = radius;
    ts.vmin      = vmin;
    ts.vmax      = (vmax > vmin) ? vmax : (vmin + ONE);
    ts.log_scale = cfg.log_scale and (vmin > ZERO);
    ts.colormap  = cfg.colormap;
    ts.n_lut     = 256;
    // Bucket grid: a LOCAL uniform lattice spanning only THIS domain's AABB
    // (not the whole box), so cell_start stays O(local cells). A bucket cell is
    // ~one coarse cell; the tube radius is far smaller, so a ray sample (always
    // inside [lo,hi]) finds its nearest segment in its own bucket cell.
    for (int d = 0; d < 3; ++d) {
      ts.gdx[d]         = (cf.dx[d] > ZERO) ? cf.dx[d] : ONE;
      ts.gorigin[d]     = lo[d];
      const real_t span = hi[d] - lo[d];
      ts.gnc[d] = std::max(1, static_cast<int>(std::ceil(span / ts.gdx[d])));
    }
    auto lin = [&](int c0, int c1, int c2) -> std::size_t {
      return (static_cast<std::size_t>(c2) * ts.gnc[1] + c1) * ts.gnc[0] + c0;
    };
    // opaque LUT: a tube sample paints a solid color (alpha==1)
    ts.lut = buildLUT(cfg.colormap, ts.n_lut, { { ZERO, ONE }, { ONE, ONE } });

    // 1) keep segments whose radius-padded AABB overlaps this domain AABB.
    //    (We keep the whole segment, not a clipped piece: the kernel only ever
    //    samples within this domain's slab, so a shared segment shows only in
    //    the correct domain's depth range -- no double-draw.)
    std::vector<std::array<real_t, 8>> kept;
    for (const auto& pl : lines) {
      for (std::size_t i = 0; i + 1 < pl.pts.size(); ++i) {
        const auto&  a = pl.pts[i];
        const auto&  b = pl.pts[i + 1];
        bool         overlap = true;
        for (int d = 0; d < 3; ++d) {
          const real_t smin = std::min(a[d], b[d]) - radius;
          const real_t smax = std::max(a[d], b[d]) + radius;
          if (smax < lo[d] or smin > hi[d]) {
            overlap = false;
            break;
          }
        }
        if (overlap) {
          kept.push_back({ a[0], a[1], a[2], b[0], b[1], b[2], pl.scal[i],
                           pl.scal[i + 1] });
        }
      }
    }
    n_kept       = kept.size();
    ts.n_seg     = static_cast<int>(kept.size());
    const std::size_t ncell = static_cast<std::size_t>(ts.gnc[0]) * ts.gnc[1] *
                              ts.gnc[2];

    // 2) CSR bucketing on the coarse grid: count, prefix-sum, scatter. Each
    //    segment is registered in every cell its radius-padded AABB overlaps.
    auto cellOf = [&](real_t x, int d) -> int {
      int c = static_cast<int>(std::floor((x - ts.gorigin[d]) / ts.gdx[d]));
      if (c < 0) {
        c = 0;
      } else if (c > ts.gnc[d] - 1) {
        c = ts.gnc[d] - 1;
      }
      return c;
    };
    auto cellRange = [&](const std::array<real_t, 8>& s, int d, int& c0, int& c1) {
      const real_t smin = std::min(s[d], s[3 + d]) - radius;
      const real_t smax = std::max(s[d], s[3 + d]) + radius;
      c0 = cellOf(smin, d);
      c1 = cellOf(smax, d);
    };

    std::vector<int> count(ncell + 1, 0);
    for (const auto& s : kept) {
      int a0, a1, b0, b1, d0, d1;
      cellRange(s, 0, a0, a1);
      cellRange(s, 1, b0, b1);
      cellRange(s, 2, d0, d1);
      for (int c2 = d0; c2 <= d1; ++c2) {
        for (int c1 = b0; c1 <= b1; ++c1) {
          for (int c0 = a0; c0 <= a1; ++c0) {
            ++count[lin(c0, c1, c2)];
          }
        }
      }
    }
    std::vector<int> start(ncell + 1, 0);
    for (std::size_t c = 0; c < ncell; ++c) {
      start[c + 1] = start[c] + count[c];
    }
    const std::size_t n_insert = static_cast<std::size_t>(start[ncell]);
    std::vector<int>  idx(n_insert, 0);
    std::vector<int>  cursor(start.begin(), start.end()); // running write head
    for (std::size_t si = 0; si < kept.size(); ++si) {
      int a0, a1, b0, b1, d0, d1;
      cellRange(kept[si], 0, a0, a1);
      cellRange(kept[si], 1, b0, b1);
      cellRange(kept[si], 2, d0, d1);
      for (int c2 = d0; c2 <= d1; ++c2) {
        for (int c1 = b0; c1 <= b1; ++c1) {
          for (int c0 = a0; c0 <= a1; ++c0) {
            const std::size_t cl = lin(c0, c1, c2);
            idx[static_cast<std::size_t>(cursor[cl]++)] = static_cast<int>(si);
          }
        }
      }
    }

    // 3) upload to device
    ts.seg = array_t<real_t* [8]>("fl_seg", static_cast<std::size_t>(ts.n_seg));
    if (ts.n_seg > 0) {
      auto seg_h = Kokkos::create_mirror_view(ts.seg);
      for (int s = 0; s < ts.n_seg; ++s) {
        for (int c = 0; c < 8; ++c) {
          seg_h(s, c) = kept[static_cast<std::size_t>(s)][static_cast<std::size_t>(c)];
        }
      }
      Kokkos::deep_copy(ts.seg, seg_h);
    }
    ts.cell_start = array_t<int*>("fl_cell_start", ncell + 1);
    {
      auto h = Kokkos::create_mirror_view(ts.cell_start);
      for (std::size_t c = 0; c <= ncell; ++c) {
        h(c) = start[c];
      }
      Kokkos::deep_copy(ts.cell_start, h);
    }
    ts.seg_idx = array_t<int*>("fl_seg_idx", std::max<std::size_t>(n_insert, 1));
    if (n_insert > 0) {
      auto h = Kokkos::create_mirror_view(ts.seg_idx);
      for (std::size_t k = 0; k < n_insert; ++k) {
        h(k) = idx[k];
      }
      Kokkos::deep_copy(ts.seg_idx, h);
    }
    return ts;
  }

  /** @brief A valid but empty tube set (used when a scene shows no field lines). */
  inline auto emptyTubeSet() -> TubeSet {
    TubeSet ts;
    ts.n_seg      = 0;
    ts.seg        = array_t<real_t* [8]>("fl_seg_empty", 0);
    ts.cell_start = array_t<int*>("fl_cell_start_empty", 1);
    ts.seg_idx    = array_t<int*>("fl_seg_idx_empty", 1);
    ts.lut        = buildLUT("inferno", 2, { { ZERO, ONE }, { ONE, ONE } });
    return ts;
  }

} // namespace out

#endif // OUTPUT_RENDER_FIELDLINES_H
