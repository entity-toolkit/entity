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

  /** @brief A constant-color opaque LUT (monochrome field lines). */
  inline auto buildSolidLUT(real_t r, real_t g, real_t b, int n_lut)
    -> array_t<real_t* [4]> {
    array_t<real_t* [4]> lut { "fl_solid_lut", static_cast<std::size_t>(n_lut) };
    auto                 h = Kokkos::create_mirror_view(lut);
    for (int i = 0; i < n_lut; ++i) {
      h(i, 0) = r; // opaque -> premultiplied == straight RGB
      h(i, 1) = g;
      h(i, 2) = b;
      h(i, 3) = ONE;
    }
    Kokkos::deep_copy(lut, h);
    return lut;
  }

  /** @brief The field-line LUT: a single color if cfg.color is set, else by |B|. */
  inline auto buildLineLUT(const FieldLineConfig& cfg, int n_lut)
    -> array_t<real_t* [4]> {
    if (cfg.color.size() == 3) {
      return buildSolidLUT(cfg.color[0], cfg.color[1], cfg.color[2], n_lut);
    }
    return buildLUT(cfg.colormap, n_lut, { { ZERO, ONE }, { ONE, ONE } });
  }

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
    // opaque LUT: a tube sample paints a solid color (alpha==1), by |B| or a
    // single monochrome color when cfg.color is set
    ts.lut = buildLineLUT(cfg, ts.n_lut);

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

  // ====================================================================== //
  //  2D field lines == contours of the flux function psi                    //
  // ====================================================================== //

  /**
   * @brief A coarse, MPI-replicated copy of the in-plane (Bx, By) field.
   * @note Component-fastest: (c0,c1,comp) lives at (c1*n0 + c0)*2 + comp.
   */
  struct CoarseField2D {
    std::vector<real_t> B;                 // n0*n1*2
    int                 n[2] { 0, 0 };
    real_t              origin[2] { ZERO, ZERO };
    real_t              dx[2] { ONE, ONE };
  };

  /**
   * @brief Integrate the flux function psi(x,y) from the coarse in-plane field.
   * @note psi obeys Bx = d psi/dy, By = -d psi/dx; the trapezoidal cumulative
   * integral (one pass along x at j=0, then up each column) is path-consistent
   * up to div(B) = 0. Because cf is globally replicated, every rank gets the
   * SAME psi -> contour levels are identical everywhere -> seamless lines.
   * @param[out] psi (n0*n1) flux function, c0-fastest
   * @param[out] psi_min,psi_max flux range (for level spacing)
   * @param[out] bmin,bmax |B| range (for contour coloring)
   */
  inline void computeFlux2D(const CoarseField2D& cf,
                            std::vector<real_t>& psi,
                            real_t&              psi_min,
                            real_t&              psi_max,
                            real_t&              bmin,
                            real_t&              bmax) {
    const int nx = cf.n[0], ny = cf.n[1];
    psi.assign(static_cast<std::size_t>(nx) * ny, ZERO);
    auto B = [&](int i, int j, int c) -> real_t {
      return cf.B[(static_cast<std::size_t>(j) * nx + i) * 2 + c];
    };
    auto P = [&](int i, int j) -> real_t& {
      return psi[static_cast<std::size_t>(j) * nx + i];
    };
    // bottom row (j = 0): d psi/dx = -By, trapezoidal in x
    for (int i = 1; i < nx; ++i) {
      P(i, 0) = P(i - 1, 0) - HALF * (B(i - 1, 0, 1) + B(i, 0, 1)) * cf.dx[0];
    }
    // each column: d psi/dy = +Bx, trapezoidal in y
    for (int i = 0; i < nx; ++i) {
      for (int j = 1; j < ny; ++j) {
        P(i, j) = P(i, j - 1) + HALF * (B(i, j - 1, 0) + B(i, j, 0)) * cf.dx[1];
      }
    }
    psi_min = static_cast<real_t>(1e30);
    psi_max = static_cast<real_t>(-1e30);
    bmin    = static_cast<real_t>(1e30);
    bmax    = static_cast<real_t>(-1e30);
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const real_t p = P(i, j);
        psi_min        = std::min(psi_min, p);
        psi_max        = std::max(psi_max, p);
        const real_t bx = B(i, j, 0), by = B(i, j, 1);
        const real_t b  = std::sqrt(bx * bx + by * by);
        bmin            = std::min(bmin, b);
        bmax            = std::max(bmax, b);
      }
    }
    if (psi_min > psi_max) {
      psi_min = ZERO;
      psi_max = ONE;
    }
    if (bmin > bmax) {
      bmin = ZERO;
      bmax = ONE;
    }
  }

  /**
   * @brief Pack the flux function + contour parameters into a device ContourSet.
   * @param line_half_px half the contour line width, in screen pixels
   * @param wpp world units per screen pixel (sets the screen-space line width)
   */
  inline auto buildContourSet(const CoarseField2D&       cf,
                              const std::vector<real_t>& psi,
                              real_t                     psi_min,
                              real_t                     psi_max,
                              real_t                     bmin,
                              real_t                     bmax,
                              const FieldLineConfig&     cfg,
                              real_t                     line_half_px,
                              real_t                     wpp) -> ContourSet {
    ContourSet cs;
    cs.n0           = cf.n[0];
    cs.n1           = cf.n[1];
    cs.origin0      = cf.origin[0];
    cs.origin1      = cf.origin[1];
    cs.dx0          = cf.dx[0];
    cs.dx1          = cf.dx[1];
    const int nlev  = std::max(1, cfg.levels);
    cs.dlevel       = (psi_max > psi_min)
                        ? (psi_max - psi_min) / static_cast<real_t>(nlev)
                        : ONE;
    cs.psi_ref      = psi_min;
    cs.line_half_px = line_half_px;
    cs.wpp          = wpp;
    cs.colormap     = cfg.colormap;
    cs.n_lut        = 256;
    real_t vlo = bmin, vhi = bmax;
    if (cfg.vmax > cfg.vmin) { // explicit |B| color range overrides auto
      vlo = cfg.vmin;
      vhi = cfg.vmax;
    }
    cs.vmin = vlo;
    cs.vmax = (vhi > vlo) ? vhi : (vlo + ONE);
    cs.lut  = buildLineLUT(cfg, cs.n_lut); // by |B| or monochrome (cfg.color)
    const std::size_t n = static_cast<std::size_t>(cs.n0) * cs.n1;
    cs.psi = array_t<real_t*>("fl_psi", std::max<std::size_t>(n, 1));
    if (n > 0) {
      auto h = Kokkos::create_mirror_view(cs.psi);
      for (std::size_t k = 0; k < n; ++k) {
        h(k) = psi[k];
      }
      Kokkos::deep_copy(cs.psi, h);
    }
    cs.enabled = true;
    return cs;
  }

  /** @brief A valid but empty contour set (scene shows no 2D field lines). */
  inline auto emptyContourSet() -> ContourSet {
    ContourSet cs;
    cs.enabled = false;
    cs.psi     = array_t<real_t*>("fl_psi_empty", 1);
    cs.lut     = buildLUT("inferno", 2, { { ZERO, ONE }, { ONE, ONE } });
    return cs;
  }

  // ====================================================================== //
  //  2D spherical / Kerr-Schild == traced meridional streamlines (nt2py)    //
  // ====================================================================== //

  namespace fl_hidden {
    // bilinear sample of the (Br, Btheta) coarse field at physical (r, theta);
    // false if (r, theta) lies outside the grid (the integrator stops there).
    inline auto sampleRTh(const CoarseField2D& cf, real_t r, real_t th, real_t B[2])
      -> bool {
      const real_t rmin = cf.origin[0], thmin = cf.origin[1];
      const real_t rmax = rmin + cf.n[0] * cf.dx[0];
      const real_t thmax = thmin + cf.n[1] * cf.dx[1];
      const real_t tr = HALF * cf.dx[0], tt = HALF * cf.dx[1];
      if (r < rmin - tr or r > rmax + tr or th < thmin - tt or th > thmax + tt) {
        return false;
      }
      int    i0, i1, j0, j1;
      real_t a0 = ZERO, a1 = ZERO;
      if (cf.n[0] <= 1) {
        i0 = 0;
        i1 = 0;
      } else {
        const real_t g = (r - rmin) / cf.dx[0] - HALF;
        const real_t f = std::floor(g);
        int          b = static_cast<int>(f);
        a0             = g - f;
        if (b < 0) {
          b  = 0;
          a0 = ZERO;
        } else if (b > cf.n[0] - 2) {
          b  = cf.n[0] - 2;
          a0 = ONE;
        }
        i0 = b;
        i1 = b + 1;
      }
      if (cf.n[1] <= 1) {
        j0 = 0;
        j1 = 0;
      } else {
        const real_t g = (th - thmin) / cf.dx[1] - HALF;
        const real_t f = std::floor(g);
        int          b = static_cast<int>(f);
        a1             = g - f;
        if (b < 0) {
          b  = 0;
          a1 = ZERO;
        } else if (b > cf.n[1] - 2) {
          b  = cf.n[1] - 2;
          a1 = ONE;
        }
        j0 = b;
        j1 = b + 1;
      }
      for (int c = 0; c < 2; ++c) {
        const real_t c00 = cf.B[(static_cast<std::size_t>(j0) * cf.n[0] + i0) * 2 + c];
        const real_t c10 = cf.B[(static_cast<std::size_t>(j0) * cf.n[0] + i1) * 2 + c];
        const real_t c01 = cf.B[(static_cast<std::size_t>(j1) * cf.n[0] + i0) * 2 + c];
        const real_t c11 = cf.B[(static_cast<std::size_t>(j1) * cf.n[0] + i1) * 2 + c];
        const real_t e0  = c00 * (ONE - a0) + c10 * a0;
        const real_t e1  = c01 * (ONE - a0) + c11 * a0;
        B[c]             = e0 * (ONE - a1) + e1 * a1;
      }
      return true;
    }
  } // namespace fl_hidden

  /**
   * @brief Trace poloidal field lines in the meridional (X, Z) plane (nt2py
   * style): integrate (Fx, Fz) = (Br sin th + Bth cos th, Br cos th - Bth sin th)
   * by bidirectional RK4 through the coarse (r, theta) field. Polylines are
   * returned in meridional world coords (z = 0 so they reuse the 3D tube
   * builder); with `mirror` the X<0 half is added as the theta-reflected copy.
   * @param cf coarse field: component 0 = Br, 1 = Btheta, grid in (r, theta)
   * @param world_per_pixel meridional world units per pixel (seed/length scale)
   */
  inline auto traceFieldLinesMeridional(const CoarseField2D&   cf,
                                        const FieldLineConfig& cfg,
                                        real_t                 world_per_pixel,
                                        bool                   mirror,
                                        real_t&                out_vmin,
                                        real_t&                out_vmax)
    -> std::vector<Polyline> {
    using fl_hidden::sampleRTh;
    std::vector<Polyline> lines;
    if (cf.n[0] < 1 or cf.n[1] < 1) {
      return lines;
    }
    const real_t rmin = cf.origin[0], thmin = cf.origin[1];
    const real_t rmax = rmin + cf.n[0] * cf.dx[0];
    const real_t thmax = thmin + cf.n[1] * cf.dx[1];
    const real_t h   = std::max(cfg.step_frac, static_cast<real_t>(1e-3)) *
                     cf.dx[0]; // step ~ a coarse dr (a length)
    const real_t max_len = cfg.max_len_frac * rmax * static_cast<real_t>(2);
    const real_t eps     = static_cast<real_t>(1e-20);

    auto bmag = [&](real_t X, real_t Z) -> real_t {
      const real_t r = std::sqrt(X * X + Z * Z);
      const real_t th = std::atan2(std::abs(X), Z);
      real_t       B[2];
      if (not sampleRTh(cf, r, th, B)) {
        return ZERO;
      }
      return std::sqrt(B[0] * B[0] + B[1] * B[1]);
    };
    // unit meridional direction (x dir); false if |F| ~ 0 or outside the grid
    auto deriv = [&](const real_t p[2], real_t dir, real_t out[2]) -> bool {
      const real_t X = p[0], Z = p[1];
      const real_t r = std::sqrt(X * X + Z * Z);
      const real_t th = std::atan2(std::abs(X), Z);
      real_t       B[2];
      if (not sampleRTh(cf, r, th, B)) {
        return false;
      }
      const real_t st = std::sin(th), ct = std::cos(th);
      // (Br, Bth) -> meridional Cartesian; sign of the X-component follows X so
      // a line seeded in X>=0 stays in X>=0 (the X<0 half is the mirror image)
      const real_t sgn = (X < ZERO) ? -ONE : ONE;
      const real_t Fx  = sgn * (B[0] * st + B[1] * ct);
      const real_t Fz  = B[0] * ct - B[1] * st;
      const real_t m   = std::sqrt(Fx * Fx + Fz * Fz);
      if (m < eps) {
        return false;
      }
      const real_t inv = dir / m;
      out[0]           = Fx * inv;
      out[1]           = Fz * inv;
      return true;
    };

    out_vmin = static_cast<real_t>(1e30);
    out_vmax = static_cast<real_t>(-1e30);
    auto track = [&](real_t m) {
      out_vmin = std::min(out_vmin, m);
      out_vmax = std::max(out_vmax, m);
    };
    auto inDomain = [&](real_t X, real_t Z) -> bool {
      const real_t r = std::sqrt(X * X + Z * Z);
      const real_t th = std::atan2(std::abs(X), Z);
      return (r >= rmin and r <= rmax and th >= thmin and th <= thmax);
    };

    auto integrate = [&](const real_t seed[2], real_t dir) {
      Polyline pl;
      real_t   p[2] = { seed[0], seed[1] };
      real_t   m0   = bmag(p[0], p[1]);
      if (m0 < eps) {
        return;
      }
      pl.pts.push_back({ p[0], p[1], ZERO });
      pl.scal.push_back(m0);
      track(m0);
      real_t len = ZERO;
      for (int step = 0; step < cfg.max_steps and len < max_len; ++step) {
        real_t k1[2], k2[2], k3[2], k4[2], q[2];
        if (not deriv(p, dir, k1)) {
          break;
        }
        q[0] = p[0] + HALF * h * k1[0];
        q[1] = p[1] + HALF * h * k1[1];
        if (not deriv(q, dir, k2)) {
          break;
        }
        q[0] = p[0] + HALF * h * k2[0];
        q[1] = p[1] + HALF * h * k2[1];
        if (not deriv(q, dir, k3)) {
          break;
        }
        q[0] = p[0] + h * k3[0];
        q[1] = p[1] + h * k3[1];
        if (not deriv(q, dir, k4)) {
          break;
        }
        p[0] += (h / static_cast<real_t>(6)) *
                (k1[0] + static_cast<real_t>(2) * k2[0] +
                 static_cast<real_t>(2) * k3[0] + k4[0]);
        p[1] += (h / static_cast<real_t>(6)) *
                (k1[1] + static_cast<real_t>(2) * k2[1] +
                 static_cast<real_t>(2) * k3[1] + k4[1]);
        if (not inDomain(p[0], p[1])) {
          break;
        }
        const real_t m = bmag(p[0], p[1]);
        pl.pts.push_back({ p[0], p[1], ZERO });
        pl.scal.push_back(m);
        track(m);
        len += h;
      }
      if (pl.pts.size() >= 2) {
        lines.push_back(std::move(pl));
      }
    };

    // seed lattice over the X>=0 meridional half, keeping in-domain seeds
    const real_t Xhi = rmax, Zlo = -rmax, Zhi = rmax;
    real_t       spacing = std::max(cfg.seed_px, ONE) * world_per_pixel;
    auto         gridCount = [&](real_t s) -> long {
      const long nx = std::max(1L, static_cast<long>(std::floor(Xhi / s)));
      const long nz = std::max(1L, static_cast<long>(std::floor((Zhi - Zlo) / s)));
      return nx * nz;
    };
    if (gridCount(spacing) > cfg.seed_max and cfg.seed_max > 0) {
      spacing *= std::sqrt(static_cast<real_t>(gridCount(spacing)) /
                           static_cast<real_t>(cfg.seed_max));
    }
    const long nx = std::max(1L, static_cast<long>(std::floor(Xhi / spacing)));
    const long nz = std::max(1L,
                             static_cast<long>(std::floor((Zhi - Zlo) / spacing)));
    for (long iz = 0; iz < nz; ++iz) {
      for (long ix = 0; ix < nx; ++ix) {
        const real_t X = (static_cast<real_t>(ix) + HALF) * Xhi /
                         static_cast<real_t>(nx);
        const real_t Z = Zlo + (static_cast<real_t>(iz) + HALF) * (Zhi - Zlo) /
                                 static_cast<real_t>(nz);
        if (not inDomain(X, Z)) {
          continue;
        }
        const real_t seed[2] = { X, Z };
        integrate(seed, ONE);
        integrate(seed, -ONE);
      }
    }
    if (out_vmin > out_vmax) {
      out_vmin = ZERO;
      out_vmax = ONE;
    }
    // mirror the traced (X>=0) lines into the X<0 half for a full disk
    if (mirror) {
      const std::size_t n0 = lines.size();
      for (std::size_t i = 0; i < n0; ++i) {
        Polyline m = lines[i];
        for (auto& q : m.pts) {
          q[0] = -q[0];
        }
        lines.push_back(std::move(m));
      }
    }
    return lines;
  }

} // namespace out

#endif // OUTPUT_RENDER_FIELDLINES_H
