/**
 * @file kernels/utils.hpp
 * @brief Commonly used generic kernels
 * @implements
 *   - kernel::ComputeSum_kernel<>
 *   - kernel::ComputeDivergence_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_UTILS_HPP
#define KERNELS_UTILS_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace kernel {

  template <Dimension D, unsigned short N>
  class ComputeSum_kernel {
    const ndfield_t<D, N> buff;
    const idx_t           buff_idx;

  public:
    ComputeSum_kernel(const ndfield_t<D, N>& buff, idx_t buff_idx)
      : buff { buff }
      , buff_idx { buff_idx } {
      raise::ErrorIf(buff_idx >= N, "Invalid component index", HERE);
    }

    Inline void operator()(index_t i1, real_t& lsum) const {
      if constexpr (D == Dim::_1D) {
        lsum += buff(i1, buff_idx);
      } else {
        raise::KernelError(
          HERE,
          "1D implementation of ComputeSum_kernel called for non-1D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, real_t& lsum) const {
      if (D == Dim::_2D) {
        lsum += buff(i1, i2, buff_idx);
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of ComputeSum_kernel called for non-2D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3, real_t& lsum) const {
      if (D == Dim::_3D) {
        lsum += buff(i1, i2, i3, buff_idx);
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of ComputeSum_kernel called for non-3D");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_UTILS_HPP
