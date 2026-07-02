/**
 * @file output/render/slice2d.hpp
 * @brief Header-only Kokkos 2D slice rasterizer (one parallel_for over pixels)
 * @implements
 *   - kernel::SliceRaster_kernel<M>
 * @namespaces:
 *   - kernel::
 * @macros:
 *   - OUTPUT_ENABLED
 * @note
 * The 2D counterpart of the volume ray-march: a 2D simulation has no depth to
 * integrate, so each screen pixel is a single inverse-mapped sample of the
 * prepared scalar, painted opaque. Two coordinate families are handled at
 * compile time via M::CoordType:
 *   - Cartesian (Minkowski 2D): the screen window IS the (x, y) physical plane;
 *     the inverse map is the per-axis code conversion.
 *   - Spherical / Qspherical (2D SR & all 2D GR): the screen window is the
 *     meridional (X, Z) Cartesian plane; a pixel maps to physical
 *     r = sqrt(X^2 + Z^2), theta = atan2(|X|, Z), then per-axis code conversion
 *     (separable once in physical spherical coords). With `mirror`, X<0 is the
 *     theta-reflected half, yielding a full disk from one axisymmetric half.
 *
 * Seamlessness: every rank shares the same global screen window, so a pixel's
 * world point is identical on all ranks. Each pixel's active-region membership
 * (code index in [0, n]) selects exactly one domain in the interior (boundary
 * pixels may be claimed by two, but the halo-filled value is continuous there),
 * so the disjoint sub-images composite without seams regardless of order.
 */

#ifndef OUTPUT_RENDER_SLICE2D_HPP
#define OUTPUT_RENDER_SLICE2D_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"

#include "output/render/renderer.h"

namespace kernel {
  using namespace ntt;

  template <class M>
  class SliceRaster_kernel {
    static constexpr auto D = M::Dim;
    static_assert(D == Dim::_2D, "SliceRaster_kernel is 2D only");

    randacc_ndfield_t<D, 6> Fld;
    const idx_t             comp;
    const M                 metric;

    // global orthographic window in slice-plane world coords (shared by ranks)
    const real_t umin, umax, vmin, vmax;
    const int    W, H;         // full frame size (ray generation / ndc)
    const int    bx0, by0, bw; // screen-bbox offset and width (output stride)
    const bool   mirror;       // spherical: paint the X<0 reflected half too

    // optional physical render-region clip: a pixel is drawn only if its
    // coordinate is inside [rx1lo,rx1hi] x [rx2lo,rx2hi] (x1,x2 == x,y for
    // Cartesian; r,theta for spherical). Off => the whole domain is drawn.
    const real_t rx1lo, rx1hi, rx2lo, rx2hi;
    const bool   region_clip;

    // local-domain active cell counts and View extents (membership + clamping)
    const real_t n1, n2;
    const int    ext0, ext1;

    // transfer function (opaque LUT: premultiplied with alpha == 1)
    array_t<real_t* [4]> lut;
    const int            n_lut;
    const real_t         vlo, vhi;
    const bool           log_scale;

    // 2D field-line contours: iso-levels of the flux function psi on a coarse
    // world grid, colored by |B| = |grad psi|. Cartesian only; drawn where psi
    // is within `cline_half_px` (screen px) of a level. `heatmap_on` false ->
    // standalone contours (no scalar fill).
    array_t<real_t*>     cpsi;
    const int            cn0, cn1;
    const real_t         corigin0, corigin1, cdx0, cdx1;
    const real_t         cdlevel, cpsi_ref, cline_half_px, cwpp;
    array_t<real_t* [4]> clut;
    const int            cn_lut;
    const real_t         cvmin, cvmax;
    const bool           contour_on;

    // spherical/Kerr field lines: traced meridional streamlines drawn as lines,
    // reusing the 3D tube segment-bucket geometry at z == 0 (queried at the
    // pixel's (X, Z) world point). Cartesian uses the contours above instead.
    array_t<real_t* [8]> lseg;
    array_t<int*>        lcell_start;
    array_t<int*>        lseg_idx;
    const int            ln_seg;
    const real_t         line_r2;
    const int            lgnc0, lgnc1, lgnc2;
    const real_t         lg0, lg1, lg2, ldx0, ldx1, ldx2;
    array_t<real_t* [4]> line_lut;
    const int            line_n_lut;
    const real_t         line_vmin, line_vmax;
    const bool           line_on;

    const bool           heatmap_on;

    array_t<real_t* [4]> image; // output, (bw*bh, 4) premultiplied RGBA

  public:
    SliceRaster_kernel(const randacc_ndfield_t<D, 6>& Fld_,
                       idx_t                          comp_,
                       const M&                       metric_,
                       real_t                         umin_,
                       real_t                         umax_,
                       real_t                         vmin_,
                       real_t                         vmax_,
                       int                            W_,
                       int                            H_,
                       int                            bx0_,
                       int                            by0_,
                       int                            bw_,
                       bool                           mirror_,
                       real_t                         rx1lo_,
                       real_t                         rx1hi_,
                       real_t                         rx2lo_,
                       real_t                         rx2hi_,
                       bool                           region_clip_,
                       int                            n1_,
                       int                            n2_,
                       int                            ext0_,
                       int                            ext1_,
                       const array_t<real_t* [4]>&    lut_,
                       int                            n_lut_,
                       real_t                         vlo_,
                       real_t                         vhi_,
                       bool                           log_scale_,
                       const out::ContourSet&         contours_,
                       const out::TubeSet&            lines_,
                       bool                           heatmap_enabled_,
                       const array_t<real_t* [4]>&    image_)
      : Fld { Fld_ }
      , comp { comp_ }
      , metric { metric_ }
      , umin { umin_ }
      , umax { umax_ }
      , vmin { vmin_ }
      , vmax { vmax_ }
      , W { W_ }
      , H { H_ }
      , bx0 { bx0_ }
      , by0 { by0_ }
      , bw { bw_ }
      , mirror { mirror_ }
      , rx1lo { rx1lo_ }
      , rx1hi { rx1hi_ }
      , rx2lo { rx2lo_ }
      , rx2hi { rx2hi_ }
      , region_clip { region_clip_ }
      , n1 { static_cast<real_t>(n1_) }
      , n2 { static_cast<real_t>(n2_) }
      , ext0 { ext0_ }
      , ext1 { ext1_ }
      , lut { lut_ }
      , n_lut { n_lut_ }
      , vlo { vlo_ }
      , vhi { vhi_ }
      , log_scale { log_scale_ }
      , cpsi { contours_.psi }
      , cn0 { contours_.n0 }
      , cn1 { contours_.n1 }
      , corigin0 { contours_.origin0 }
      , corigin1 { contours_.origin1 }
      , cdx0 { contours_.dx0 }
      , cdx1 { contours_.dx1 }
      , cdlevel { contours_.dlevel }
      , cpsi_ref { contours_.psi_ref }
      , cline_half_px { contours_.line_half_px }
      , cwpp { contours_.wpp }
      , clut { contours_.lut }
      , cn_lut { contours_.n_lut }
      , cvmin { contours_.vmin }
      , cvmax { contours_.vmax }
      , contour_on { contours_.enabled }
      , lseg { lines_.seg }
      , lcell_start { lines_.cell_start }
      , lseg_idx { lines_.seg_idx }
      , ln_seg { lines_.n_seg }
      , line_r2 { lines_.radius * lines_.radius }
      , lgnc0 { lines_.gnc[0] }
      , lgnc1 { lines_.gnc[1] }
      , lgnc2 { lines_.gnc[2] }
      , lg0 { lines_.gorigin[0] }
      , lg1 { lines_.gorigin[1] }
      , lg2 { lines_.gorigin[2] }
      , ldx0 { lines_.gdx[0] }
      , ldx1 { lines_.gdx[1] }
      , ldx2 { lines_.gdx[2] }
      , line_lut { lines_.lut }
      , line_n_lut { lines_.n_lut }
      , line_vmin { lines_.vmin }
      , line_vmax { lines_.vmax }
      , line_on { lines_.n_seg > 0 }
      , heatmap_on { heatmap_enabled_ }
      , image { image_ } {}

    // bilinear sample of the prepared scalar at continuous code coords
    // (cc1, cc2), reading the ghost halo for corners just outside the box.
    Inline auto sample(real_t cc1, real_t cc2) const -> real_t {
      const real_t g0 = cc1 - HALF; // cell-center continuous index
      const real_t g1 = cc2 - HALF;
      const real_t f0 = math::floor(g0);
      const real_t f1 = math::floor(g1);
      const real_t t0 = g0 - f0;
      const real_t t1 = g1 - f1;
      int          b0 = static_cast<int>(f0) + static_cast<int>(N_GHOSTS);
      int          b1 = static_cast<int>(f1) + static_cast<int>(N_GHOSTS);
      b0 = (b0 < 0) ? 0 : ((b0 > ext0 - 2) ? ext0 - 2 : b0);
      b1 = (b1 < 0) ? 0 : ((b1 > ext1 - 2) ? ext1 - 2 : b1);
      const real_t c00 = Fld(b0, b1, comp);
      const real_t c10 = Fld(b0 + 1, b1, comp);
      const real_t c01 = Fld(b0, b1 + 1, comp);
      const real_t c11 = Fld(b0 + 1, b1 + 1, comp);
      const real_t c0  = c00 * (ONE - t0) + c10 * t0;
      const real_t c1  = c01 * (ONE - t0) + c11 * t0;
      return c0 * (ONE - t1) + c1 * t1;
    }

    // bilinear sample of the coarse flux function psi at world point (x, y)
    Inline auto sampleFlux(real_t x, real_t y) const -> real_t {
      if (cn0 <= 0 or cn1 <= 0) {
        return ZERO;
      }
      int    i0, i1, j0, j1;
      real_t t0 = ZERO, t1 = ZERO;
      if (cn0 <= 1) {
        i0 = 0;
        i1 = 0;
      } else {
        const real_t g0 = (x - corigin0) / cdx0 - HALF;
        const real_t f0 = math::floor(g0);
        int          b0 = static_cast<int>(f0);
        t0              = g0 - f0;
        if (b0 < 0) {
          b0 = 0;
          t0 = ZERO;
        } else if (b0 > cn0 - 2) {
          b0 = cn0 - 2;
          t0 = ONE;
        }
        i0 = b0;
        i1 = b0 + 1;
      }
      if (cn1 <= 1) {
        j0 = 0;
        j1 = 0;
      } else {
        const real_t g1 = (y - corigin1) / cdx1 - HALF;
        const real_t f1 = math::floor(g1);
        int          b1 = static_cast<int>(f1);
        t1              = g1 - f1;
        if (b1 < 0) {
          b1 = 0;
          t1 = ZERO;
        } else if (b1 > cn1 - 2) {
          b1 = cn1 - 2;
          t1 = ONE;
        }
        j0 = b1;
        j1 = b1 + 1;
      }
      const real_t c00 = cpsi(j0 * cn0 + i0);
      const real_t c10 = cpsi(j0 * cn0 + i1);
      const real_t c01 = cpsi(j1 * cn0 + i0);
      const real_t c11 = cpsi(j1 * cn0 + i1);
      const real_t c0  = c00 * (ONE - t0) + c10 * t0;
      const real_t c1  = c01 * (ONE - t0) + c11 * t0;
      return c0 * (ONE - t1) + c1 * t1;
    }

    // is meridional world point (x, z) within the line width of any traced
    // streamline segment? Same bucketed distance-to-segment test as the 3D
    // tube kernel, restricted to the z == 0 plane. On a hit, `scalar` is |B|.
    Inline auto inLine(real_t x, real_t z, real_t& scalar) const -> bool {
      if (ln_seg <= 0) {
        return false;
      }
      const int c0 = static_cast<int>(math::floor((x - lg0) / ldx0));
      const int c1 = static_cast<int>(math::floor((z - lg1) / ldx1));
      const int c2 = static_cast<int>(math::floor((ZERO - lg2) / ldx2));
      if (c0 < 0 or c0 >= lgnc0 or c1 < 0 or c1 >= lgnc1 or c2 < 0 or
          c2 >= lgnc2) {
        return false;
      }
      const int    lin  = (c2 * lgnc1 + c1) * lgnc0 + c0;
      const int    kb   = lcell_start(lin);
      const int    ke   = lcell_start(lin + 1);
      real_t       best = line_r2;
      bool         hit  = false;
      for (int k = kb; k < ke; ++k) {
        const int    s  = lseg_idx(k);
        const real_t ax = lseg(s, 0), az = lseg(s, 1); // (X, Z) in slots 0,1
        const real_t bx = lseg(s, 3), bz = lseg(s, 4);
        const real_t ex = bx - ax, ez = bz - az;
        const real_t wx = x - ax, wz = z - az;
        const real_t ee = ex * ex + ez * ez;
        real_t       tt = (ee > ZERO) ? (wx * ex + wz * ez) / ee : ZERO;
        tt              = (tt < ZERO) ? ZERO : ((tt > ONE) ? ONE : tt);
        const real_t cx = ax + tt * ex, cz = az + tt * ez;
        const real_t dx = x - cx, dz = z - cz;
        const real_t d2 = dx * dx + dz * dz;
        if (d2 < best) {
          best   = d2;
          scalar = lseg(s, 6) * (ONE - tt) + lseg(s, 7) * tt;
          hit    = true;
        }
      }
      return hit;
    }

    Inline void operator()(cellidx_t lpx, cellidx_t lpy) const {
      const auto pix = static_cast<std::size_t>(lpy) *
                         static_cast<std::size_t>(bw) +
                       static_cast<std::size_t>(lpx);
      const int gpx = bx0 + static_cast<int>(lpx);
      const int gpy = by0 + static_cast<int>(lpy);
      // default transparent
      image(pix, 0) = ZERO;
      image(pix, 1) = ZERO;
      image(pix, 2) = ZERO;
      image(pix, 3) = ZERO;

      // pixel center -> slice-plane world coords (v flipped so +v is up)
      const real_t u = umin + (static_cast<real_t>(gpx) + HALF) /
                                static_cast<real_t>(W) * (umax - umin);
      const real_t v = vmax - (static_cast<real_t>(gpy) + HALF) /
                                static_cast<real_t>(H) * (vmax - vmin);

      // world -> continuous local code coords, with an optional physical
      // render-region clip (so a crop hides domain data outside the region, not
      // just reframes the view)
      real_t cc1, cc2;
      if constexpr (M::CoordType == Coord::Cartesian) {
        if (region_clip and
            (u < rx1lo or u > rx1hi or v < rx2lo or v > rx2hi)) {
          return;
        }
        cc1 = metric.template convert<1, Crd::Ph, Crd::Cd>(u);
        cc2 = metric.template convert<2, Crd::Ph, Crd::Cd>(v);
      } else {
        if (not mirror and u < ZERO) {
          return; // only the X>=0 meridional half is physical
        }
        const real_t r  = math::sqrt(u * u + v * v);
        const real_t th = math::atan2(math::abs(u), v); // in [0, pi]
        if (region_clip and
            (r < rx1lo or r > rx1hi or th < rx2lo or th > rx2hi)) {
          return;
        }
        cc1 = metric.template convert<1, Crd::Ph, Crd::Cd>(r);
        cc2 = metric.template convert<2, Crd::Ph, Crd::Cd>(th);
      }
      // active-region membership (inclusive so interiors gap-free, boundaries
      // shared harmlessly); outside -> leave transparent
      if (cc1 < ZERO or cc1 > n1 or cc2 < ZERO or cc2 > n2) {
        return;
      }

      real_t cr = ZERO, cg = ZERO, cb = ZERO;
      bool   painted = false;
      if (heatmap_on) {
        const real_t s = sample(cc1, cc2);
        // normalize through the transfer-function range
        const real_t inv_range = (vhi > vlo) ? (ONE / (vhi - vlo)) : ZERO;
        real_t       uu;
        if (log_scale) {
          const real_t log_vlo = math::log10(vlo);
          uu = (s > ZERO) ? (math::log10(s) - log_vlo) * inv_range : -ONE;
        } else {
          uu = (s - vlo) * inv_range;
        }
        if (uu < ZERO) {
          uu = ZERO;
        } else if (uu > ONE) {
          uu = ONE;
        }
        int idx = static_cast<int>(uu * static_cast<real_t>(n_lut - 1) + HALF);
        if (idx < 0) {
          idx = 0;
        } else if (idx > n_lut - 1) {
          idx = n_lut - 1;
        }
        // opaque LUT: premultiplied with alpha == 1, so this is straight RGB
        cr      = lut(idx, 0);
        cg      = lut(idx, 1);
        cb      = lut(idx, 2);
        painted = true;
      }
      // field-line contours: iso-levels of the flux function psi (Cartesian
      // only, where (u, v) IS the (x, y) world plane). Drawn over the heatmap
      // and colored by |B| = |grad psi|; a screen-space line width keeps the
      // contours ~constant thickness regardless of local gradient.
      if constexpr (M::CoordType == Coord::Cartesian) {
        if (contour_on) {
          const real_t psi0 = sampleFlux(u, v);
          const real_t pl   = sampleFlux(u - cwpp, v);
          const real_t pr   = sampleFlux(u + cwpp, v);
          const real_t pd   = sampleFlux(u, v - cwpp);
          const real_t pup  = sampleFlux(u, v + cwpp);
          const real_t gx   = (pr - pl) / (TWO * cwpp);
          const real_t gy   = (pup - pd) / (TWO * cwpp);
          const real_t g    = math::sqrt(gx * gx + gy * gy); // |B|
          const real_t tlev = (cdlevel > ZERO) ? (psi0 - cpsi_ref) / cdlevel
                                               : ZERO;
          const real_t nlev = math::floor(tlev + HALF); // nearest level index
          const real_t d_world = math::abs(psi0 - (cpsi_ref + nlev * cdlevel));
          const real_t eps     = static_cast<real_t>(1e-30);
          const real_t d_screen = (g > eps) ? (d_world / (g * cwpp))
                                            : static_cast<real_t>(1e30);
          if (d_screen <= cline_half_px) {
            const real_t invr = (cvmax > cvmin) ? (ONE / (cvmax - cvmin)) : ZERO;
            real_t       uu   = (g - cvmin) * invr;
            if (uu < ZERO) {
              uu = ZERO;
            } else if (uu > ONE) {
              uu = ONE;
            }
            int idx = static_cast<int>(uu * static_cast<real_t>(cn_lut - 1) +
                                       HALF);
            if (idx < 0) {
              idx = 0;
            } else if (idx > cn_lut - 1) {
              idx = cn_lut - 1;
            }
            cr      = clut(idx, 0);
            cg      = clut(idx, 1);
            cb      = clut(idx, 2);
            painted = true;
          }
        }
      } else {
        // spherical/Kerr: traced meridional streamlines. (u, v) is the (X, Z)
        // world point; draw a line where it falls within a segment's width.
        if (line_on) {
          real_t sB;
          if (inLine(u, v, sB)) {
            const real_t invr = (line_vmax > line_vmin)
                                  ? (ONE / (line_vmax - line_vmin))
                                  : ZERO;
            real_t uu = (sB - line_vmin) * invr;
            if (uu < ZERO) {
              uu = ZERO;
            } else if (uu > ONE) {
              uu = ONE;
            }
            int idx = static_cast<int>(uu * static_cast<real_t>(line_n_lut - 1) +
                                       HALF);
            if (idx < 0) {
              idx = 0;
            } else if (idx > line_n_lut - 1) {
              idx = line_n_lut - 1;
            }
            cr      = line_lut(idx, 0);
            cg      = line_lut(idx, 1);
            cb      = line_lut(idx, 2);
            painted = true;
          }
        }
      }
      if (painted) {
        image(pix, 0) = cr;
        image(pix, 1) = cg;
        image(pix, 2) = cb;
        image(pix, 3) = ONE;
      }
    }
  };

} // namespace kernel

#endif // OUTPUT_RENDER_SLICE2D_HPP
