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

    // local-domain active cell counts and View extents (membership + clamping)
    const real_t n1, n2;
    const int    ext0, ext1;

    // transfer function (opaque LUT: premultiplied with alpha == 1)
    array_t<real_t* [4]> lut;
    const int            n_lut;
    const real_t         vlo, vhi;
    const bool           log_scale;

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
                       int                            n1_,
                       int                            n2_,
                       int                            ext0_,
                       int                            ext1_,
                       const array_t<real_t* [4]>&    lut_,
                       int                            n_lut_,
                       real_t                         vlo_,
                       real_t                         vhi_,
                       bool                           log_scale_,
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
      , n1 { static_cast<real_t>(n1_) }
      , n2 { static_cast<real_t>(n2_) }
      , ext0 { ext0_ }
      , ext1 { ext1_ }
      , lut { lut_ }
      , n_lut { n_lut_ }
      , vlo { vlo_ }
      , vhi { vhi_ }
      , log_scale { log_scale_ }
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

      // world -> continuous local code coords
      real_t cc1, cc2;
      if constexpr (M::CoordType == Coord::Cartesian) {
        cc1 = metric.template convert<1, Crd::Ph, Crd::Cd>(u);
        cc2 = metric.template convert<2, Crd::Ph, Crd::Cd>(v);
      } else {
        if (not mirror and u < ZERO) {
          return; // only the X>=0 meridional half is physical
        }
        const real_t r  = math::sqrt(u * u + v * v);
        const real_t th = math::atan2(math::abs(u), v); // in [0, pi]
        cc1 = metric.template convert<1, Crd::Ph, Crd::Cd>(r);
        cc2 = metric.template convert<2, Crd::Ph, Crd::Cd>(th);
      }
      // active-region membership (inclusive so interiors gap-free, boundaries
      // shared harmlessly); outside -> leave transparent
      if (cc1 < ZERO or cc1 > n1 or cc2 < ZERO or cc2 > n2) {
        return;
      }

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
      image(pix, 0) = lut(idx, 0);
      image(pix, 1) = lut(idx, 1);
      image(pix, 2) = lut(idx, 2);
      image(pix, 3) = ONE;
    }
  };

} // namespace kernel

#endif // OUTPUT_RENDER_SLICE2D_HPP
