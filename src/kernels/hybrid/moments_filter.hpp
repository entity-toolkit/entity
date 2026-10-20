#ifndef KERNELS_HYBRID_MOMENTS_FILTER_HPP
#define KERNELS_HYBRID_MOMENTS_FILTER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::hybrid {
  using namespace ntt;

  /**
   * Binomial (1-2-1) smoothing of the deposited hybrid moments.
   *
   * The Ohm's-law E is built directly from the deposited number density N and
   * momentum density V (aux::3 and aux::012). With cold, fast beams the per-cell
   * shot noise in V (~ v_drift / sqrt(ppc)) is injected straight into E via the
   * motional (-u x B) and Hall (J x B / N) terms and drives a grid-scale
   * numerical instability. Smoothing the moments before the field solve removes
   * the grid-scale (Nyquist) noise, exactly as current filtering does in the
   * explicit PIC engines.
   *
   * Filters comps 0..3 (V0, V1, V2, N) of the 6-component `array` (aux), reading
   * from `buffer` (a copy of `array`). Cartesian, interior stencil only: the
   * i +/- 1 reads reach into the ghost layer, which the caller refills (periodic
   * / MPI) between passes via CommunicateFields(AUX). N_GHOSTS >= 1 is required
   * (always true for SHAPE_ORDER >= 0).
   *
   * @note Conductor/reflecting walls are NOT special-cased here (unlike
   *       DigitalFilter_kernel); the plain stencil smooths against whatever the
   *       wall ghost moments hold. Hybrid wall setups should revisit this if they
   *       enable filtering.
   */
  template <Dimension D>
  class MomentsFilter_kernel {
    ndfield_t<D, 6>       array;
    const ndfield_t<D, 6> buffer;

    static constexpr unsigned short NCOMP = 4; // V0, V1, V2, N

  public:
    MomentsFilter_kernel(const ndfield_t<D, 6>& array,
                         const ndfield_t<D, 6>& buffer)
      : array { array }
      , buffer { buffer } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
#pragma unroll
        for (auto c { 0u }; c < NCOMP; ++c) {
          array(i1, c) = INV_2 * buffer(i1, c) +
                         INV_4 * (buffer(i1 - 1, c) + buffer(i1 + 1, c));
        }
      } else {
        raise::KernelError(HERE, "MomentsFilter_kernel: 1D called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
#pragma unroll
        for (auto c { 0u }; c < NCOMP; ++c) {
          array(i1, i2, c) = INV_4 * buffer(i1, i2, c) +
                             INV_8 * (buffer(i1 - 1, i2, c) +
                                      buffer(i1 + 1, i2, c) +
                                      buffer(i1, i2 - 1, c) +
                                      buffer(i1, i2 + 1, c)) +
                             INV_16 * (buffer(i1 - 1, i2 - 1, c) +
                                       buffer(i1 + 1, i2 + 1, c) +
                                       buffer(i1 - 1, i2 + 1, c) +
                                       buffer(i1 + 1, i2 - 1, c));
        }
      } else {
        raise::KernelError(HERE, "MomentsFilter_kernel: 2D called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
#pragma unroll
        for (auto c { 0u }; c < NCOMP; ++c) {
          array(i1, i2, i3, c) =
            INV_8 * buffer(i1, i2, i3, c) +
            INV_16 * (buffer(i1 - 1, i2, i3, c) + buffer(i1 + 1, i2, i3, c) +
                      buffer(i1, i2 - 1, i3, c) + buffer(i1, i2 + 1, i3, c) +
                      buffer(i1, i2, i3 - 1, c) + buffer(i1, i2, i3 + 1, c)) +
            INV_32 * (buffer(i1 - 1, i2 - 1, i3, c) +
                      buffer(i1 + 1, i2 + 1, i3, c) +
                      buffer(i1 - 1, i2 + 1, i3, c) +
                      buffer(i1 + 1, i2 - 1, i3, c) +
                      buffer(i1, i2 - 1, i3 - 1, c) +
                      buffer(i1, i2 + 1, i3 + 1, c) +
                      buffer(i1, i2 - 1, i3 + 1, c) +
                      buffer(i1, i2 + 1, i3 - 1, c) +
                      buffer(i1 - 1, i2, i3 - 1, c) +
                      buffer(i1 + 1, i2, i3 + 1, c) +
                      buffer(i1 - 1, i2, i3 + 1, c) +
                      buffer(i1 + 1, i2, i3 - 1, c)) +
            INV_64 * (buffer(i1 - 1, i2 - 1, i3 - 1, c) +
                      buffer(i1 + 1, i2 + 1, i3 + 1, c) +
                      buffer(i1 - 1, i2 + 1, i3 + 1, c) +
                      buffer(i1 + 1, i2 - 1, i3 - 1, c) +
                      buffer(i1 - 1, i2 - 1, i3 + 1, c) +
                      buffer(i1 + 1, i2 + 1, i3 - 1, c) +
                      buffer(i1 - 1, i2 + 1, i3 - 1, c) +
                      buffer(i1 + 1, i2 - 1, i3 + 1, c));
        }
      } else {
        raise::KernelError(HERE, "MomentsFilter_kernel: 3D called for D != 3");
      }
    }
  };

} // namespace kernel::hybrid

#endif // KERNELS_HYBRID_MOMENTS_FILTER_HPP
