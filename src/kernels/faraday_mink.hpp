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
#include "utils/numeric.h"

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
    const real_t    deltax;
    const real_t    deltay;
    const real_t    betaxy;
    const real_t    betayx;
    const real_t    deltaz;
    const real_t    betaxz;
    const real_t    betazx;
    const real_t    betayz;
    const real_t    betazy;

  public:
    /**
     * ! 1D: coeff1 = dt / dx
     * ! 2D: coeff1 = dt / dx^2, coeff2 = dt
     * ! 3D: coeff1 = dt / dx
     */
    Faraday_kernel(const ndfield_t<D, 6>& EB,
                   real_t                 coeff1,
                   real_t                 coeff2,
                   real_t                 deltax = ZERO,
                   real_t                 deltay = ZERO,
                   real_t                 betaxy = ZERO,
                   real_t                 betayx = ZERO,
                   real_t                 deltaz = ZERO,
                   real_t                 betaxz = ZERO,
                   real_t                 betazx = ZERO,
                   real_t                 betayz = ZERO,
                   real_t                 betazy = ZERO)
      : EB { EB }
      , coeff1 { coeff1 }
      , coeff2 { coeff2 }
      , deltax { deltax }
      , deltay { deltay }
      , betaxy { betaxy }
      , betayx { betayx }
      , deltaz { deltaz }
      , betaxz { betaxz }
      , betazx { betazx }
      , betayz { betayz }
      , betazy { betazy } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto alphax = ONE - THREE * deltax;
        // clang-format off
        EB(i1, em::bx2) += coeff1 * (
                        + alphax * (EB(i1 + 1, em::ex3) - EB(i1    , em::ex3))
                        + deltax * (EB(i1 + 2, em::ex3) - EB(i1 - 1, em::ex3)));
        EB(i1, em::bx3) += coeff1 * (
                        - alphax * (EB(i1 + 1, em::ex2) - EB(i1    , em::ex2))
                        - deltax * (EB(i1 + 2, em::ex2) - EB(i1 - 1, em::ex2)));
        // clang-format on
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const auto alphax = ONE - TWO * betaxy - THREE * deltax;
        const auto alphay = ONE - TWO * betayx - THREE * deltay;
        // clang-format off
        EB(i1, i2, em::bx1) += coeff1 * (
                            - alphay * (EB(i1    , i2 + 1, em::ex3) - EB(i1    , i2    , em::ex3))
                            - deltay * (EB(i1    , i2 + 2, em::ex3) - EB(i1    , i2 - 1, em::ex3))
                            - betayx * (EB(i1 + 1, i2 + 1, em::ex3) - EB(i1 + 1, i2    , em::ex3))
                            - betayx * (EB(i1 - 1, i2 + 1, em::ex3) - EB(i1 - 1, i2    , em::ex3)));
        EB(i1, i2, em::bx2) += coeff1 * (
                            + alphax * (EB(i1 + 1, i2    , em::ex3) - EB(i1    , i2    , em::ex3))
                            + deltax * (EB(i1 + 2, i2    , em::ex3) - EB(i1 - 1, i2    , em::ex3))
                            + betaxy * (EB(i1 + 1, i2 + 1, em::ex3) - EB(i1    , i2 + 1, em::ex3))
                            + betaxy * (EB(i1 + 1, i2 - 1, em::ex3) - EB(i1    , i2 - 1, em::ex3)));
        EB(i1, i2, em::bx3) += coeff2 * (
                            + alphay * (EB(i1    , i2 + 1, em::ex1) - EB(i1    , i2    , em::ex1))
                            + deltay * (EB(i1    , i2 + 2, em::ex1) - EB(i1    , i2 - 1, em::ex1))
                            + betayx * (EB(i1 + 1, i2 + 1, em::ex1) - EB(i1 + 1, i2    , em::ex1))
                            + betayx * (EB(i1 - 1, i2 + 1, em::ex1) - EB(i1 - 1, i2    , em::ex1))
                            - alphax * (EB(i1 + 1, i2    , em::ex2) - EB(i1    , i2    , em::ex2))
                            - deltax * (EB(i1 + 2, i2    , em::ex2) - EB(i1 - 1, i2    , em::ex2))
                            - betaxy * (EB(i1 + 1, i2 + 1, em::ex2) - EB(i1    , i2 + 1, em::ex2))
                            - betaxy * (EB(i1 + 1, i2 - 1, em::ex2) - EB(i1    , i2 - 1, em::ex2)));
        // clang-format on
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        const auto alphax = ONE - TWO * betaxy - TWO * betaxz - THREE * deltax;
        const auto alphay = ONE - TWO * betayx - TWO * betayz - THREE * deltay;
        const auto alphaz = ONE - TWO * betazx - TWO * betazy - THREE * deltaz;
        // clang-format off
        EB(i1, i2, i3, em::bx1) += coeff1 * (
                        + alphaz * (EB(i1    , i2    , i3 + 1, em::ex2) - EB(i1    , i2    , i3    , em::ex2))
                        + deltaz * (EB(i1    , i2    , i3 + 2, em::ex2) - EB(i1    , i2    , i3 - 1, em::ex2))
                        + betazx * (EB(i1 + 1, i2    , i3 + 1, em::ex2) - EB(i1 + 1, i2    , i3    , em::ex2))
                        + betazx * (EB(i1 - 1, i2    , i3 + 1, em::ex2) - EB(i1 - 1, i2    , i3    , em::ex2))
                        + betazy * (EB(i1    , i2 + 1, i3 + 1, em::ex2) - EB(i1    , i2 + 1, i3    , em::ex2))
                        + betazy * (EB(i1    , i2 - 1, i3 + 1, em::ex2) - EB(i1    , i2 - 1, i3    , em::ex2))
                        - alphay * (EB(i1    , i2 + 1, i3    , em::ex3) - EB(i1    , i2    , i3    , em::ex3))
                        - deltay * (EB(i1    , i2 + 2, i3    , em::ex3) - EB(i1    , i2 - 1, i3    , em::ex3))
                        - betayx * (EB(i1 + 1, i2 + 1, i3    , em::ex3) - EB(i1 + 1, i2    , i3    , em::ex3))
                        - betayx * (EB(i1 - 1, i2 + 1, i3    , em::ex3) - EB(i1 - 1, i2    , i3    , em::ex3))
                        - betayz * (EB(i1    , i2 + 1, i3 + 1, em::ex3) - EB(i1    , i2    , i3 + 1, em::ex3))
                        - betayz * (EB(i1    , i2 + 1, i3 - 1, em::ex3) - EB(i1    , i2    , i3 - 1, em::ex3)));
        EB(i1, i2, i3, em::bx2) += coeff1 * (
                        + alphax * (EB(i1 + 1, i2    , i3    , em::ex3) - EB(i1    , i2    , i3    , em::ex3))
                        + deltax * (EB(i1 + 2, i2    , i3    , em::ex3) - EB(i1 - 1, i2    , i3    , em::ex3))
                        + betaxy * (EB(i1 + 1, i2 + 1, i3    , em::ex3) - EB(i1    , i2 + 1, i3    , em::ex3))
                        + betaxy * (EB(i1 + 1, i2 - 1, i3    , em::ex3) - EB(i1    , i2 - 1, i3    , em::ex3))
                        + betaxz * (EB(i1 + 1, i2    , i3 + 1, em::ex3) - EB(i1    , i2    , i3 + 1, em::ex3))
                        + betaxz * (EB(i1 + 1, i2    , i3 - 1, em::ex3) - EB(i1    , i2    , i3 - 1, em::ex3))
                        - alphaz * (EB(i1    , i2    , i3 + 1, em::ex1) - EB(i1    , i2    , i3    , em::ex1))
                        - deltaz * (EB(i1    , i2    , i3 + 2, em::ex1) - EB(i1    , i2    , i3 - 1, em::ex1))
                        - betazx * (EB(i1 + 1, i2    , i3 + 1, em::ex1) - EB(i1 + 1, i2    , i3    , em::ex1))
                        - betazx * (EB(i1 - 1, i2    , i3 + 1, em::ex1) - EB(i1 - 1, i2    , i3    , em::ex1))
                        - betazy * (EB(i1    , i2 + 1, i3 + 1, em::ex1) - EB(i1    , i2 + 1, i3    , em::ex1))
                        - betazy * (EB(i1    , i2 - 1, i3 + 1, em::ex1) - EB(i1    , i2 - 1, i3    , em::ex1)));
        EB(i1, i2, i3, em::bx3) += coeff1 * (
                        + alphay * (EB(i1    , i2 + 1, i3    , em::ex1) - EB(i1    , i2    , i3    , em::ex1))
                        + deltay * (EB(i1    , i2 + 2, i3    , em::ex1) - EB(i1    , i2 - 1, i3    , em::ex1))
                        + betayx * (EB(i1 + 1, i2 + 1, i3    , em::ex1) - EB(i1 + 1, i2    , i3    , em::ex1))
                        + betayx * (EB(i1 - 1, i2 + 1, i3    , em::ex1) - EB(i1 - 1, i2    , i3    , em::ex1))
                        + betayz * (EB(i1    , i2 + 1, i3 + 1, em::ex1) - EB(i1    , i2    , i3 + 1, em::ex1))
                        + betayz * (EB(i1    , i2 + 1, i3 - 1, em::ex1) - EB(i1    , i2    , i3 - 1, em::ex1))
                        - alphax * (EB(i1 + 1, i2    , i3    , em::ex2) - EB(i1    , i2    , i3    , em::ex2))
                        - deltax * (EB(i1 + 2, i2    , i3    , em::ex2) - EB(i1 - 1, i2    , i3    , em::ex2))
                        - betaxy * (EB(i1 + 1, i2 + 1, i3    , em::ex2) - EB(i1    , i2 + 1, i3    , em::ex2))
                        - betaxy * (EB(i1 + 1, i2 - 1, i3    , em::ex2) - EB(i1    , i2 - 1, i3    , em::ex2))
                        - betaxz * (EB(i1 + 1, i2    , i3 + 1, em::ex2) - EB(i1    , i2    , i3 + 1, em::ex2))
                        - betaxz * (EB(i1 + 1, i2    , i3 - 1, em::ex2) - EB(i1    , i2    , i3 - 1, em::ex2)));
        // clang-format on
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 3D implementation called for D != 3");
      }
    }
  };
} // namespace kernel::mink

#endif // KERNELS_FARADAY_MINK_HPP
