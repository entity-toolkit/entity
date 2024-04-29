/**
 * @file kernels/faraday_mink.hpp
 * @brief Algorithms for Faraday's law in cartesian Minkowski space
 * @implements
 *   - kernel::mink::Faraday_kernel<>
 * @namespaces:
 *   - kernel::mink
 */

#ifndef KERNELS_FARADAY_MINK_HPP
#define KERNELS_FARADAY_MINK_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace kernel::mink {
  using namespace ntt;

  /**
   * @brief Algorithm for the Faraday's law: `dB/dt = -curl E` in Minkowski space.
   */
  template <Dimension D>
  class Faraday_kernel {
    ndfield_t<D, 6> EB;
    const real_t    coeff1;
    const real_t    coeff2;

  public:
    /**
     * ! 1D: coeff1 = dt / dx
     * ! 2D: coeff1 = dt / dx^2, coeff2 = dt
     * ! 3D: coeff1 = dt / dx
     */
    Faraday_kernel(const ndfield_t<D, 6>& EB, real_t coeff1, real_t coeff2)
      : EB { EB }
      , coeff1 { coeff1 }
      , coeff2 { coeff2 } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        EB(i1, em::bx2) += coeff1 * (EB(i1 + 1, em::ex3) - EB(i1, em::ex3));
        EB(i1, em::bx3) += coeff1 * (EB(i1, em::ex2) - EB(i1 + 1, em::ex2));
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        EB(i1, i2, em::bx1) += coeff1 *
                               (EB(i1, i2, em::ex3) - EB(i1, i2 + 1, em::ex3));
        EB(i1, i2, em::bx2) += coeff1 *
                               (EB(i1 + 1, i2, em::ex3) - EB(i1, i2, em::ex3));
        EB(i1, i2, em::bx3) += coeff2 *
                               (EB(i1, i2 + 1, em::ex1) - EB(i1, i2, em::ex1) +
                                EB(i1, i2, em::ex2) - EB(i1 + 1, i2, em::ex2));

      } else {
        raise::KernelError(HERE, "Faraday_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        EB(i1, i2, i3, em::bx1) += coeff1 * (EB(i1, i2, i3 + 1, em::ex2) -
                                             EB(i1, i2, i3, em::ex2) +
                                             EB(i1, i2, i3, em::ex3) -
                                             EB(i1, i2 + 1, i3, em::ex3));
        EB(i1, i2, i3, em::bx2) += coeff1 * (EB(i1 + 1, i2, i3, em::ex3) -
                                             EB(i1, i2, i3, em::ex3) +
                                             EB(i1, i2, i3, em::ex1) -
                                             EB(i1, i2, i3 + 1, em::ex1));
        EB(i1, i2, i3, em::bx3) += coeff1 * (EB(i1, i2 + 1, i3, em::ex1) -
                                             EB(i1, i2, i3, em::ex1) +
                                             EB(i1, i2, i3, em::ex2) -
                                             EB(i1 + 1, i2, i3, em::ex2));

      } else {
        raise::KernelError(HERE, "Faraday_kernel: 3D implementation called for D != 3");
      }
    }
  };
} // namespace kernel::mink

#endif // KERNELS_FARADAY_MINK_HPP
