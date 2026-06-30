/**
 * @file output/render/raymarch.hpp
 * @brief Header-only Kokkos volume ray-march kernel (one parallel_for over pixels)
 * @implements
 *   - kernel::VolumeRayMarch_kernel<M>
 * @namespaces:
 *   - kernel::
 * @macros:
 *   - OUTPUT_ENABLED
 * @note
 * Pure Kokkos: the only device entities are Views, the (trivially-copyable)
 * metric, and the POD camera. Runs in Kokkos::DefaultExecutionSpace, inheriting
 * whatever backend entity was built with (HIP / CUDA / SYCL / OpenMP).
 *
 * Seamlessness: every rank marches at GLOBAL sample positions t_k = k*ds
 * measured from the shared camera eye, with a FIXED world-space step `ds`
 * identical on all ranks. Each global sample therefore lands in exactly one
 * domain (half-open membership via the per-domain slab interval [t_enter,
 * t_exit)), so the ordered cross-domain "over" composite reproduces the single
 * full-ray integral exactly. Trilinear sampling reads into the 1-cell ghost
 * halo entity already exchanges, so the per-rank field is C0-continuous up to
 * the shared face.
 */

#ifndef OUTPUT_RENDER_RAYMARCH_HPP
#define OUTPUT_RENDER_RAYMARCH_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"

#include "output/render/renderer.h"

namespace kernel {
  using namespace ntt;

  template <class M>
  class VolumeRayMarch_kernel {
    static constexpr auto D = M::Dim;

    randacc_ndfield_t<D, 6> Fld;
    const idx_t             comp;
    const M                 metric;
    const out::CameraDevice cam;

    // local-domain world AABB and View extents (for index clamping)
    const real_t lo0, lo1, lo2, hi0, hi1, hi2;
    const int    ext0, ext1, ext2;

    const int    W, H;        // full frame size (for ray generation / ndc)
    const int    bx0, by0, bw; // screen-bbox offset and width (output stride)
    const real_t ds;          // fixed world step (global, identical on all ranks)
    const int    max_steps;   // safety cap on the marching loop

    // transfer function
    array_t<real_t* [4]> lut;
    const int            n_lut;
    const real_t         vmin, vmax;
    const bool           log_scale;
    const real_t         early_alpha;

    array_t<real_t* [4]> image; // output, (bw*bh, 4) premultiplied RGBA

  public:
    VolumeRayMarch_kernel(const randacc_ndfield_t<D, 6>& Fld_,
                          idx_t                          comp_,
                          const M&                       metric_,
                          const out::CameraDevice&       cam_,
                          const real_t                   lo[3],
                          const real_t                   hi[3],
                          int                            ext0_,
                          int                            ext1_,
                          int                            ext2_,
                          int                            W_,
                          int                            H_,
                          int                            bx0_,
                          int                            by0_,
                          int                            bw_,
                          real_t                         ds_,
                          int                            max_steps_,
                          const array_t<real_t* [4]>&    lut_,
                          int                            n_lut_,
                          real_t                         vmin_,
                          real_t                         vmax_,
                          bool                           log_scale_,
                          real_t                         early_alpha_,
                          const array_t<real_t* [4]>&    image_)
      : Fld { Fld_ }
      , comp { comp_ }
      , metric { metric_ }
      , cam { cam_ }
      , lo0 { lo[0] }
      , lo1 { lo[1] }
      , lo2 { lo[2] }
      , hi0 { hi[0] }
      , hi1 { hi[1] }
      , hi2 { hi[2] }
      , ext0 { ext0_ }
      , ext1 { ext1_ }
      , ext2 { ext2_ }
      , W { W_ }
      , H { H_ }
      , bx0 { bx0_ }
      , by0 { by0_ }
      , bw { bw_ }
      , ds { ds_ }
      , max_steps { max_steps_ }
      , lut { lut_ }
      , n_lut { n_lut_ }
      , vmin { vmin_ }
      , vmax { vmax_ }
      , log_scale { log_scale_ }
      , early_alpha { early_alpha_ }
      , image { image_ } {}

    // trilinear sample of the prepared scalar at world point p, reading the
    // ghost halo for corners just outside the active box.
    Inline auto sample(real_t px, real_t py, real_t pz) const -> real_t {
      // world -> code (cell-center continuous index = code index - 1/2)
      const real_t g0 = metric.template convert<1, Crd::Ph, Crd::Cd>(px) - HALF;
      const real_t g1 = metric.template convert<2, Crd::Ph, Crd::Cd>(py) - HALF;
      const real_t g2 = metric.template convert<3, Crd::Ph, Crd::Cd>(pz) - HALF;
      const real_t f0 = math::floor(g0);
      const real_t f1 = math::floor(g1);
      const real_t f2 = math::floor(g2);
      const real_t t0 = g0 - f0;
      const real_t t1 = g1 - f1;
      const real_t t2 = g2 - f2;
      // base View index of the lower corner (active cell i -> View i + N_GHOSTS)
      int b0 = static_cast<int>(f0) + static_cast<int>(N_GHOSTS);
      int b1 = static_cast<int>(f1) + static_cast<int>(N_GHOSTS);
      int b2 = static_cast<int>(f2) + static_cast<int>(N_GHOSTS);
      // clamp so both corners (b, b+1) stay in range [0, ext-1]
      b0 = (b0 < 0) ? 0 : ((b0 > ext0 - 2) ? ext0 - 2 : b0);
      b1 = (b1 < 0) ? 0 : ((b1 > ext1 - 2) ? ext1 - 2 : b1);
      b2 = (b2 < 0) ? 0 : ((b2 > ext2 - 2) ? ext2 - 2 : b2);
      const real_t c000 = Fld(b0, b1, b2, comp);
      const real_t c100 = Fld(b0 + 1, b1, b2, comp);
      const real_t c010 = Fld(b0, b1 + 1, b2, comp);
      const real_t c110 = Fld(b0 + 1, b1 + 1, b2, comp);
      const real_t c001 = Fld(b0, b1, b2 + 1, comp);
      const real_t c101 = Fld(b0 + 1, b1, b2 + 1, comp);
      const real_t c011 = Fld(b0, b1 + 1, b2 + 1, comp);
      const real_t c111 = Fld(b0 + 1, b1 + 1, b2 + 1, comp);
      const real_t c00  = c000 * (ONE - t0) + c100 * t0;
      const real_t c10  = c010 * (ONE - t0) + c110 * t0;
      const real_t c01  = c001 * (ONE - t0) + c101 * t0;
      const real_t c11  = c011 * (ONE - t0) + c111 * t0;
      const real_t c0   = c00 * (ONE - t1) + c10 * t1;
      const real_t c1   = c01 * (ONE - t1) + c11 * t1;
      return c0 * (ONE - t2) + c1 * t2;
    }

    Inline void operator()(cellidx_t lpx, cellidx_t lpy) const {
      // local bbox index -> output pixel; global pixel -> ray generation
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

      // ---- ray generation ------------------------------------------------ //
      const real_t fx = TWO * (static_cast<real_t>(gpx) + HALF) /
                          static_cast<real_t>(W) - ONE;
      const real_t fy = ONE - TWO * (static_cast<real_t>(gpy) + HALF) /
                                static_cast<real_t>(H);
      real_t ox, oy, oz, dx, dy, dz;
      if (cam.orthographic) {
        const real_t sx = fx * cam.half_w;
        const real_t sy = fy * cam.half_h;
        ox = cam.eye[0] + sx * cam.right[0] + sy * cam.up[0];
        oy = cam.eye[1] + sx * cam.right[1] + sy * cam.up[1];
        oz = cam.eye[2] + sx * cam.right[2] + sy * cam.up[2];
        dx = cam.forward[0];
        dy = cam.forward[1];
        dz = cam.forward[2];
      } else {
        const real_t nx = fx * cam.aspect * cam.tan_half_fov;
        const real_t ny = fy * cam.tan_half_fov;
        dx = cam.forward[0] + nx * cam.right[0] + ny * cam.up[0];
        dy = cam.forward[1] + nx * cam.right[1] + ny * cam.up[1];
        dz = cam.forward[2] + nx * cam.right[2] + ny * cam.up[2];
        const real_t inv = ONE / math::sqrt(dx * dx + dy * dy + dz * dz);
        dx *= inv;
        dy *= inv;
        dz *= inv;
        ox = cam.eye[0];
        oy = cam.eye[1];
        oz = cam.eye[2];
      }

      // ---- ray-AABB slab test against [lo, hi] --------------------------- //
      real_t       t_enter = ZERO;
      real_t       t_exit  = static_cast<real_t>(1e30);
      const real_t eps     = static_cast<real_t>(1e-12);
      // axis 0
      if (math::abs(dx) < eps) {
        if (ox < lo0 or ox > hi0) {
          return;
        }
      } else {
        real_t t1 = (lo0 - ox) / dx;
        real_t t2 = (hi0 - ox) / dx;
        if (t1 > t2) {
          const real_t tmp = t1;
          t1               = t2;
          t2               = tmp;
        }
        t_enter = (t1 > t_enter) ? t1 : t_enter;
        t_exit  = (t2 < t_exit) ? t2 : t_exit;
      }
      // axis 1
      if (math::abs(dy) < eps) {
        if (oy < lo1 or oy > hi1) {
          return;
        }
      } else {
        real_t t1 = (lo1 - oy) / dy;
        real_t t2 = (hi1 - oy) / dy;
        if (t1 > t2) {
          const real_t tmp = t1;
          t1               = t2;
          t2               = tmp;
        }
        t_enter = (t1 > t_enter) ? t1 : t_enter;
        t_exit  = (t2 < t_exit) ? t2 : t_exit;
      }
      // axis 2
      if (math::abs(dz) < eps) {
        if (oz < lo2 or oz > hi2) {
          return;
        }
      } else {
        real_t t1 = (lo2 - oz) / dz;
        real_t t2 = (hi2 - oz) / dz;
        if (t1 > t2) {
          const real_t tmp = t1;
          t1               = t2;
          t2               = tmp;
        }
        t_enter = (t1 > t_enter) ? t1 : t_enter;
        t_exit  = (t2 < t_exit) ? t2 : t_exit;
      }
      if (t_enter >= t_exit) {
        return;
      }

      // ---- march at global sample positions t_k = k*ds ------------------- //
      const real_t inv_range = (vmax > vmin) ? (ONE / (vmax - vmin)) : ZERO;
      const real_t log_vmin  = log_scale ? math::log10(vmin) : ZERO;
      // first global sample index inside this segment: t_k >= t_enter
      const real_t k0    = math::ceil(t_enter / ds);
      real_t       t     = k0 * ds;
      real_t       acc_r = ZERO, acc_g = ZERO, acc_b = ZERO, acc_a = ZERO;
      int          steps = 0;
      while (t < t_exit and steps < max_steps) {
        const real_t s = sample(ox + t * dx, oy + t * dy, oz + t * dz);
        // normalize through the transfer function range
        real_t u;
        if (log_scale) {
          u = (s > ZERO) ? (math::log10(s) - log_vmin) * inv_range
                         : -ONE;
        } else {
          u = (s - vmin) * inv_range;
        }
        if (u < ZERO) {
          u = ZERO;
        } else if (u > ONE) {
          u = ONE;
        }
        int idx = static_cast<int>(u * static_cast<real_t>(n_lut - 1) + HALF);
        if (idx < 0) {
          idx = 0;
        } else if (idx > n_lut - 1) {
          idx = n_lut - 1;
        }
        const real_t cr = lut(idx, 0); // premultiplied
        const real_t cg = lut(idx, 1);
        const real_t cb = lut(idx, 2);
        const real_t ca = lut(idx, 3);
        const real_t w  = ONE - acc_a;
        acc_r += w * cr;
        acc_g += w * cg;
        acc_b += w * cb;
        acc_a += w * ca;
        if (acc_a >= early_alpha) {
          break;
        }
        t += ds;
        ++steps;
      }

      image(pix, 0) = acc_r;
      image(pix, 1) = acc_g;
      image(pix, 2) = acc_b;
      image(pix, 3) = acc_a;
    }
  };

} // namespace kernel

#endif // OUTPUT_RENDER_RAYMARCH_HPP
