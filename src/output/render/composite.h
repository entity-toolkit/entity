/**
 * @file output/render/composite.h
 * @brief Front-to-back visibility ordering for the structured decomposition
 *        and the premultiplied "over" compositing operator.
 * @implements
 *   - out::compositeOrderKey
 *   - out::overComposite
 * @namespaces:
 *   - out::
 * @note
 * entity decomposes the global box into a regular Dx x Dy x Dz grid of domains
 * (domain index == MPI rank). For a camera viewing the box from outside, the
 * correct global front-to-back order is a deterministic per-axis ordering by
 * which side of each split plane the camera sits on -- no general depth sort,
 * no cyclic overlap. Ordered premultiplied "over" of the non-overlapping,
 * correctly-ordered per-domain segments reconstructs the single-image ray
 * integral, hence is seamless.
 */

#ifndef OUTPUT_RENDER_COMPOSITE_H
#define OUTPUT_RENDER_COMPOSITE_H

#include "global.h"

#include "utils/numeric.h"

#include "output/render/renderer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace out {

  /**
   * @brief Total-order sort key placing nearer domains first (front-to-back).
   * @param offset integer grid coordinate of the domain (offset_ndomains)
   * @param ndoms  number of domains per axis (ndomains_per_dim)
   * @param forward camera view direction (world == code axes for Minkowski)
   * @return a single key; ascending key == front-to-back. Smaller is nearer.
   *
   * For axis d: if the camera looks toward +d (forward[d] >= 0), the smaller
   * grid index is nearer, so key_d = offset_d. Otherwise key_d is reversed.
   * The per-axis keys are packed lexicographically (axis 0 most significant).
   */
  inline auto compositeOrderKey(const std::vector<unsigned int>& offset,
                                const std::vector<unsigned int>& ndoms,
                                const real_t                     forward[3])
    -> uint64_t {
    uint64_t key = 0;
    for (std::size_t d = 0; d < ndoms.size(); ++d) {
      const unsigned int Dd  = ndoms[d];
      const unsigned int od  = offset[d];
      const unsigned int kd  = (forward[d] >= ZERO) ? od : (Dd - 1u - od);
      key                    = key * static_cast<uint64_t>(Dd) +
            static_cast<uint64_t>(kd);
    }
    return key;
  }

  /**
   * @brief Accumulate one segment into a front-to-back running composite.
   * @param acc 4-element premultiplied RGBA accumulator (modified in place)
   * @param seg 4-element premultiplied RGBA of the next (further) segment
   *
   * acc holds everything in front of seg. The "over" operator:
   *   C_acc += (1 - A_acc) * C_seg ;  A_acc += (1 - A_acc) * A_seg
   * Associative with identity (0,0,0,0); segments must be supplied front first.
   */
  inline void overComposite(real_t acc[4], const real_t seg[4]) {
    const real_t one_minus_a = ONE - acc[3];
    acc[0] += one_minus_a * seg[0];
    acc[1] += one_minus_a * seg[1];
    acc[2] += one_minus_a * seg[2];
    acc[3] += one_minus_a * seg[3];
  }

  /**
   * @brief Project a world point to a (fractional) screen pixel, inverting the
   * ray-march kernel's ray generation.
   * @return false if the point is behind a perspective camera (no projection)
   */
  inline auto projectToScreen(const CameraDevice& cam,
                              int                 W,
                              int                 H,
                              const real_t        p[3],
                              real_t&             outx,
                              real_t&             outy) -> bool {
    const real_t dx = p[0] - cam.eye[0];
    const real_t dy = p[1] - cam.eye[1];
    const real_t dz = p[2] - cam.eye[2];
    const real_t cx = dx * cam.right[0] + dy * cam.right[1] + dz * cam.right[2];
    const real_t cy = dx * cam.up[0] + dy * cam.up[1] + dz * cam.up[2];
    real_t       fx, fy;
    if (cam.orthographic) {
      fx = cx / cam.half_w;
      fy = cy / cam.half_h;
    } else {
      const real_t cz = dx * cam.forward[0] + dy * cam.forward[1] +
                        dz * cam.forward[2];
      if (cz <= static_cast<real_t>(1e-6)) {
        return false;
      }
      fx = (cx / cz) / (cam.aspect * cam.tan_half_fov);
      fy = (cy / cz) / cam.tan_half_fov;
    }
    outx = (fx + ONE) * HALF * static_cast<real_t>(W) - HALF;
    outy = (ONE - fy) * HALF * static_cast<real_t>(H) - HALF;
    return true;
  }

  /**
   * @brief Screen-space bounding box (in pixels) of a world-space AABB.
   * @param lo,hi world AABB corners
   * @param[out] bx0,by0,bw,bh clamped pixel bbox (top-left + size)
   * @return false if the box projects to an empty on-screen region
   * @note Falls back to the full frame if any corner is behind the camera.
   */
  inline auto screenBBox(const CameraDevice& cam,
                         int                 W,
                         int                 H,
                         const real_t        lo[3],
                         const real_t        hi[3],
                         int&                bx0,
                         int&                by0,
                         int&                bw,
                         int&                bh) -> bool {
    real_t minx = static_cast<real_t>(1e30), miny = static_cast<real_t>(1e30);
    real_t maxx = static_cast<real_t>(-1e30), maxy = static_cast<real_t>(-1e30);
    for (int c = 0; c < 8; ++c) {
      const real_t p[3] = { (c & 1) ? hi[0] : lo[0],
                            (c & 2) ? hi[1] : lo[1],
                            (c & 4) ? hi[2] : lo[2] };
      real_t       sx, sy;
      if (not projectToScreen(cam, W, H, p, sx, sy)) {
        bx0 = 0;
        by0 = 0;
        bw  = W;
        bh  = H;
        return true; // conservative fallback
      }
      minx = std::min(minx, sx);
      maxx = std::max(maxx, sx);
      miny = std::min(miny, sy);
      maxy = std::max(maxy, sy);
    }
    const int pad = 2;
    int       x0  = static_cast<int>(std::floor(minx)) - pad;
    int       x1  = static_cast<int>(std::ceil(maxx)) + pad;
    int       y0  = static_cast<int>(std::floor(miny)) - pad;
    int       y1  = static_cast<int>(std::ceil(maxy)) + pad;
    x0 = std::max(0, std::min(W, x0));
    x1 = std::max(0, std::min(W, x1));
    y0 = std::max(0, std::min(H, y0));
    y1 = std::max(0, std::min(H, y1));
    bx0 = x0;
    by0 = y0;
    bw  = x1 - x0;
    bh  = y1 - y0;
    return (bw > 0 and bh > 0);
  }

  /**
   * @brief Composite two sparse sub-images: `front` OVER `back`.
   * @return a sub-image spanning the union of the two bounding boxes
   * @note premultiplied "over": out = front + (1 - front.a) * back. Associative,
   *       so a tree of these reproduces the sequential front-to-back composite.
   */
  inline auto overSub(const SubImage& f, const SubImage& b) -> SubImage {
    if (f.w == 0 or f.h == 0) {
      return b;
    }
    if (b.w == 0 or b.h == 0) {
      return f;
    }
    const int ux0 = std::min(f.x0, b.x0);
    const int uy0 = std::min(f.y0, b.y0);
    const int ux1 = std::max(f.x0 + f.w, b.x0 + b.w);
    const int uy1 = std::max(f.y0 + f.h, b.y0 + b.h);
    SubImage  r;
    r.x0 = ux0;
    r.y0 = uy0;
    r.w  = ux1 - ux0;
    r.h  = uy1 - uy0;
    r.rgba.assign(static_cast<std::size_t>(r.w) * r.h * 4, ZERO);
    // place `back`
    for (int y = 0; y < b.h; ++y) {
      for (int x = 0; x < b.w; ++x) {
        const std::size_t ri = (static_cast<std::size_t>(b.y0 + y - uy0) * r.w +
                                (b.x0 + x - ux0)) *
                               4;
        const std::size_t bi = (static_cast<std::size_t>(y) * b.w + x) * 4;
        r.rgba[ri + 0] = b.rgba[bi + 0];
        r.rgba[ri + 1] = b.rgba[bi + 1];
        r.rgba[ri + 2] = b.rgba[bi + 2];
        r.rgba[ri + 3] = b.rgba[bi + 3];
      }
    }
    // `front` OVER the (back-filled) result
    for (int y = 0; y < f.h; ++y) {
      for (int x = 0; x < f.w; ++x) {
        const std::size_t ri = (static_cast<std::size_t>(f.y0 + y - uy0) * r.w +
                                (f.x0 + x - ux0)) *
                               4;
        const std::size_t fi  = (static_cast<std::size_t>(y) * f.w + x) * 4;
        const real_t      inv = ONE - f.rgba[fi + 3];
        r.rgba[ri + 0] = f.rgba[fi + 0] + inv * r.rgba[ri + 0];
        r.rgba[ri + 1] = f.rgba[fi + 1] + inv * r.rgba[ri + 1];
        r.rgba[ri + 2] = f.rgba[fi + 2] + inv * r.rgba[ri + 2];
        r.rgba[ri + 3] = f.rgba[fi + 3] + inv * r.rgba[ri + 3];
      }
    }
    return r;
  }

  /* ====================================================================== */
  /*  Interior-eye dome: fisheye projection + depth-resolved (A-buffer)     */
  /*  composite. Used when a single global front-to-back order does not     */
  /*  exist (camera inside the box). See renderer.h::FragImage.             */
  /* ====================================================================== */

  /**
   * @brief Forward azimuthal-equidistant fisheye projection: world point -> the
   * (fractional) dome pixel, inverting the dome ray generation.
   * @return false if the point is outside the dome field of view.
   * @note Uses the same ndc<->pixel convention as projectToScreen (so a square
   * frame gives a centered disk of radius 1); the dome kernel forces aspect 1.
   */
  inline auto projectToScreenDome(const CameraDevice& cam,
                                  int                 W,
                                  int                 H,
                                  const real_t        p[3],
                                  real_t&             outx,
                                  real_t&             outy) -> bool {
    real_t       v[3] = { p[0] - cam.eye[0], p[1] - cam.eye[1], p[2] - cam.eye[2] };
    const real_t n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n < static_cast<real_t>(1e-20)) {
      outx = HALF * static_cast<real_t>(W) - HALF; // eye itself -> disk center
      outy = HALF * static_cast<real_t>(H) - HALF;
      return true;
    }
    const real_t inv = ONE / n;
    v[0] *= inv;
    v[1] *= inv;
    v[2] *= inv;
    real_t cz = v[0] * cam.forward[0] + v[1] * cam.forward[1] +
                v[2] * cam.forward[2];
    cz                 = (cz < -ONE) ? -ONE : ((cz > ONE) ? ONE : cz);
    const real_t theta = std::acos(cz);
    if (theta > cam.dome_half_fov) {
      return false; // outside the dome FOV
    }
    const real_t r  = theta / cam.dome_half_fov; // 0..1 image radius
    const real_t cx = v[0] * cam.right[0] + v[1] * cam.right[1] +
                      v[2] * cam.right[2];
    const real_t cy = v[0] * cam.up[0] + v[1] * cam.up[1] + v[2] * cam.up[2];
    const real_t phi = std::atan2(cy, cx);
    const real_t fx  = r * std::cos(phi), fy = r * std::sin(phi);
    outx = (fx + ONE) * HALF * static_cast<real_t>(W) - HALF;
    outy = (ONE - fy) * HALF * static_cast<real_t>(H) - HALF;
    return true;
  }

  /**
   * @brief Conservative screen-space bbox of a world AABB under the dome fisheye.
   * @return false (empty) if the box is entirely outside the dome FOV.
   * @note Never under-covers (that would drop fragments and reintroduce seams):
   *   - eye inside the (inclusive) AABB           -> full frame
   *   - all edge samples outside the FOV          -> empty
   *   - some in / some out (straddles the horizon)-> full frame
   *   - footprint contains the zenith or wraps the disk center (max angular gap
   *     between projected samples < pi)           -> full frame
   *   - otherwise                                 -> tight bbox of the samples
   */
  inline auto screenBBoxDome(const CameraDevice& cam,
                             int                 W,
                             int                 H,
                             const real_t        lo[3],
                             const real_t        hi[3],
                             int&                bx0,
                             int&                by0,
                             int&                bw,
                             int&                bh) -> bool {
    auto fullFrame = [&]() {
      bx0 = 0;
      by0 = 0;
      bw  = W;
      bh  = H;
      return true;
    };
    // eye inside the domain -> covers all azimuths + the zenith -> full frame
    if (cam.eye[0] >= lo[0] and cam.eye[0] <= hi[0] and cam.eye[1] >= lo[1] and
        cam.eye[1] <= hi[1] and cam.eye[2] >= lo[2] and cam.eye[2] <= hi[2]) {
      return fullFrame();
    }
    // the zenith ray (disk center) piercing this domain also means it covers the
    // center -> full frame. A ray/AABB slab test from the eye along `forward`
    // catches the case edge-sampling can miss (a slab pierced through a face,
    // where the minr / azimuth-gap tests stay just under threshold -> a hole at
    // the frame center). Only the forward half-line (t >= 0) is considered.
    {
      const real_t reps = static_cast<real_t>(1e-12);
      real_t       te = ZERO, tx = static_cast<real_t>(1e30);
      bool         hit = true;
      for (int d = 0; d < 3; ++d) {
        const real_t o = cam.eye[d], dd = cam.forward[d];
        if (dd > -reps and dd < reps) {
          if (o < lo[d] or o > hi[d]) {
            hit = false;
            break;
          }
        } else {
          real_t t1 = (lo[d] - o) / dd, t2 = (hi[d] - o) / dd;
          if (t1 > t2) {
            const real_t tmp = t1;
            t1               = t2;
            t2               = tmp;
          }
          te = (t1 > te) ? t1 : te;
          tx = (t2 < tx) ? t2 : tx;
        }
      }
      if (hit and te <= tx and tx >= ZERO) {
        return fullFrame();
      }
    }
    real_t      minx = static_cast<real_t>(1e30), miny = static_cast<real_t>(1e30);
    real_t      maxx = static_cast<real_t>(-1e30), maxy = static_cast<real_t>(-1e30);
    real_t      minr = static_cast<real_t>(1e30);
    int         n_in = 0, n_out = 0;
    // azimuths of in-FOV samples, for the "wraps the center" (largest-gap) test
    std::vector<real_t> phis;
    phis.reserve(12 * 17);
    const int NS = 48; // samples per AABB edge (a straight edge maps to a
                       // curved fisheye arc, so sample densely to bound it)
    auto      addPoint = [&](const real_t p[3]) {
      real_t sx, sy;
      if (not projectToScreenDome(cam, W, H, p, sx, sy)) {
        ++n_out;
        return;
      }
      ++n_in;
      minx        = std::min(minx, sx);
      maxx        = std::max(maxx, sx);
      miny        = std::min(miny, sy);
      maxy        = std::max(maxy, sy);
      const real_t fx = TWO * (sx + HALF) / static_cast<real_t>(W) - ONE;
      const real_t fy = ONE - TWO * (sy + HALF) / static_cast<real_t>(H);
      minr        = std::min(minr, std::sqrt(fx * fx + fy * fy));
      phis.push_back(std::atan2(fy, fx));
    };
    // sample all 12 edges of the AABB
    for (int axis = 0; axis < 3; ++axis) {
      for (int c = 0; c < 4; ++c) {
        // the two AABB axes perpendicular to `axis` are fixed to lo/hi per `c`
        const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
        real_t    p[3];
        p[a1] = (c & 1) ? hi[a1] : lo[a1];
        p[a2] = (c & 2) ? hi[a2] : lo[a2];
        for (int s = 0; s <= NS; ++s) {
          const real_t t = static_cast<real_t>(s) / static_cast<real_t>(NS);
          p[axis]        = lo[axis] + (hi[axis] - lo[axis]) * t;
          addPoint(p);
        }
      }
    }
    if (n_in == 0) {
      bw = 0;
      bh = 0;
      return false; // entirely outside the FOV
    }
    if (n_out > 0) {
      return fullFrame(); // straddles the FOV boundary -> conservative
    }
    if (minr < static_cast<real_t>(1e-3)) {
      return fullFrame(); // footprint reaches the zenith (disk center)
    }
    // largest cyclic gap between azimuths: if < pi the samples wrap the center,
    // so the axis-aligned bbox of the boundary would miss the interior.
    std::sort(phis.begin(), phis.end());
    real_t       maxgap = ZERO;
    const real_t twopi  = static_cast<real_t>(2.0 * 3.14159265358979323846);
    for (std::size_t i = 0; i + 1 < phis.size(); ++i) {
      maxgap = std::max(maxgap, phis[i + 1] - phis[i]);
    }
    if (not phis.empty()) {
      maxgap = std::max(maxgap, (phis.front() + twopi) - phis.back());
    }
    // a small tolerance past pi keeps borderline wraps conservative
    if (maxgap < static_cast<real_t>(3.14159265358979323846 + 0.05)) {
      return fullFrame();
    }
    // pad generously: the fisheye arc between edge samples can bulge a few px
    const int pad = 4;
    int       x0  = static_cast<int>(std::floor(minx)) - pad;
    int       x1  = static_cast<int>(std::ceil(maxx)) + pad;
    int       y0  = static_cast<int>(std::floor(miny)) - pad;
    int       y1  = static_cast<int>(std::ceil(maxy)) + pad;
    x0  = std::max(0, std::min(W, x0));
    x1  = std::max(0, std::min(W, x1));
    y0  = std::max(0, std::min(H, y0));
    y1  = std::max(0, std::min(H, y1));
    bx0 = x0;
    by0 = y0;
    bw  = x1 - x0;
    bh  = y1 - y0;
    return (bw > 0 and bh > 0);
  }

  /**
   * @brief Collapse one pixel's depth-sorted fragment list [k0, k1) into a
   * single premultiplied RGBA via front-to-back "over".
   */
  inline void fragOver(const std::vector<real_t>& depth,
                       const std::vector<real_t>& rgba,
                       uint32_t                   k0,
                       uint32_t                   k1,
                       real_t                     out[4]) {
    (void)depth; // fragments are already ascending in depth
    real_t acc[4] = { ZERO, ZERO, ZERO, ZERO };
    for (uint32_t k = k0; k < k1; ++k) {
      overComposite(acc, &rgba[static_cast<std::size_t>(k) * 4]);
      if (acc[3] >= ONE) {
        break;
      }
    }
    out[0] = acc[0];
    out[1] = acc[1];
    out[2] = acc[2];
    out[3] = acc[3];
  }

  /**
   * @brief Merge two depth-sorted fragment images: union the bboxes and, per
   * pixel, merge the two ascending fragment lists by depth, then drop fragments
   * once the accumulated alpha reaches `cull_alpha` (exact when cull_alpha == 1:
   * only provably-occluded fragments are removed, so the result is independent
   * of how the tree is grouped -> associative + commutative).
   */
  inline auto mergeFrag(const FragImage& a, const FragImage& b, real_t cull_alpha)
    -> FragImage {
    if (a.w == 0 or a.h == 0) {
      return b;
    }
    if (b.w == 0 or b.h == 0) {
      return a;
    }
    const int ux0 = std::min(a.x0, b.x0);
    const int uy0 = std::min(a.y0, b.y0);
    const int ux1 = std::max(a.x0 + a.w, b.x0 + b.w);
    const int uy1 = std::max(a.y0 + a.h, b.y0 + b.h);
    FragImage r;
    r.x0 = ux0;
    r.y0 = uy0;
    r.w  = ux1 - ux0;
    r.h  = uy1 - uy0;
    const std::size_t np = static_cast<std::size_t>(r.w) * r.h;
    r.offs.assign(np + 1, 0u);

    // fetch a source image's fragment range at global pixel (gx, gy)
    auto range = [](const FragImage& s, int gx, int gy, uint32_t& k0, uint32_t& k1) {
      const int lx = gx - s.x0, ly = gy - s.y0;
      if (lx < 0 or ly < 0 or lx >= s.w or ly >= s.h) {
        k0 = 0;
        k1 = 0;
        return;
      }
      const std::size_t p = static_cast<std::size_t>(ly) * s.w + lx;
      k0                  = s.offs[p];
      k1                  = s.offs[p + 1];
    };

    // pass 1: per-pixel surviving-fragment count (merge + occlusion cull)
    for (int gy = uy0; gy < uy1; ++gy) {
      for (int gx = ux0; gx < ux1; ++gx) {
        uint32_t ak0, ak1, bk0, bk1;
        range(a, gx, gy, ak0, ak1);
        range(b, gx, gy, bk0, bk1);
        uint32_t ia = ak0, ib = bk0, cnt = 0;
        real_t   A  = ZERO;
        while ((ia < ak1 or ib < bk1) and A < cull_alpha) {
          const bool takeA = (ib >= bk1) or
                             (ia < ak1 and a.depth[ia] <= b.depth[ib]);
          const real_t al = takeA ? a.rgba[static_cast<std::size_t>(ia) * 4 + 3]
                                  : b.rgba[static_cast<std::size_t>(ib) * 4 + 3];
          A += (ONE - A) * al;
          ++cnt;
          if (takeA) {
            ++ia;
          } else {
            ++ib;
          }
        }
        const std::size_t pix = static_cast<std::size_t>(gy - uy0) * r.w +
                                (gx - ux0);
        r.offs[pix + 1] = cnt;
      }
    }
    // prefix-sum to offsets
    for (std::size_t p = 0; p < np; ++p) {
      r.offs[p + 1] += r.offs[p];
    }
    const std::size_t nfrag = r.offs[np];
    r.depth.assign(nfrag, ZERO);
    r.rgba.assign(nfrag * 4, ZERO);

    // pass 2: fill the merged, culled fragments
    for (int gy = uy0; gy < uy1; ++gy) {
      for (int gx = ux0; gx < ux1; ++gx) {
        uint32_t ak0, ak1, bk0, bk1;
        range(a, gx, gy, ak0, ak1);
        range(b, gx, gy, bk0, bk1);
        const std::size_t pix = static_cast<std::size_t>(gy - uy0) * r.w +
                                (gx - ux0);
        uint32_t ia = ak0, ib = bk0, o = r.offs[pix];
        const uint32_t oend = r.offs[pix + 1];
        while (o < oend) {
          const bool takeA = (ib >= bk1) or
                             (ia < ak1 and a.depth[ia] <= b.depth[ib]);
          if (takeA) {
            r.depth[o]         = a.depth[ia];
            const std::size_t s = static_cast<std::size_t>(ia) * 4;
            const std::size_t d = static_cast<std::size_t>(o) * 4;
            r.rgba[d + 0]      = a.rgba[s + 0];
            r.rgba[d + 1]      = a.rgba[s + 1];
            r.rgba[d + 2]      = a.rgba[s + 2];
            r.rgba[d + 3]      = a.rgba[s + 3];
            ++ia;
          } else {
            r.depth[o]         = b.depth[ib];
            const std::size_t s = static_cast<std::size_t>(ib) * 4;
            const std::size_t d = static_cast<std::size_t>(o) * 4;
            r.rgba[d + 0]      = b.rgba[s + 0];
            r.rgba[d + 1]      = b.rgba[s + 1];
            r.rgba[d + 2]      = b.rgba[s + 2];
            r.rgba[d + 3]      = b.rgba[s + 3];
            ++ib;
          }
          ++o;
        }
      }
    }
    return r;
  }

} // namespace out

#endif // OUTPUT_RENDER_COMPOSITE_H
