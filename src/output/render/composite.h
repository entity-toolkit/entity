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

} // namespace out

#endif // OUTPUT_RENDER_COMPOSITE_H
