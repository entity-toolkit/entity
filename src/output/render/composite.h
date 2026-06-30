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

} // namespace out

#endif // OUTPUT_RENDER_COMPOSITE_H
