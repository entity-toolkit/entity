/**
 * @file kernels/ampere_mink.hpp
 * @brief Algorithms for Ampere's law in cartesian Minkowski space
 * @implements
 *   - kernel::mink::Ampere_kernel<>
 *   - kernel::mink::CurrentsAmpere_kernel<>
 * @namespaces:
 *   - kernel::mink::
 */

#ifndef KERNELS_AMPERE_MINK_HPP
#define KERNELS_AMPERE_MINK_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::mink {
  using namespace ntt;

  struct NoCurrent_t {
    NoCurrent_t() {}
  };

  /**
   * @brief Algorithm for the Ampere's law: `dE/dt = curl B` in Minkowski space.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class Ampere_kernel {
    ndfield_t<D, 6> EB;
    const real_t    coeff1;
    const real_t    coeff2;

  public:
    /**
     * ! 1D: coeff1 = dt / dx
     * ! 2D: coeff1 = dt / dx^2, coeff2 = dt
     * ! 3D: coeff1 = dt / dx
     */
    Ampere_kernel(const ndfield_t<D, 6>& EB, real_t coeff1, real_t coeff2)
      : EB { EB }
      , coeff1 { coeff1 }
      , coeff2 { coeff2 } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        EB(i1, em::ex2) += coeff1 * (EB(i1 - 1, em::bx3) - EB(i1, em::bx3));
        EB(i1, em::ex3) += coeff1 * (EB(i1, em::bx2) - EB(i1 - 1, em::bx2));
      } else {
        raise::KernelError(HERE, "Ampere_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        EB(i1, i2, em::ex1) += coeff1 *
                               (EB(i1, i2, em::bx3) - EB(i1, i2 - 1, em::bx3));
        EB(i1, i2, em::ex2) += coeff1 *
                               (EB(i1 - 1, i2, em::bx3) - EB(i1, i2, em::bx3));
        EB(i1, i2, em::ex3) += coeff2 *
                               (EB(i1, i2 - 1, em::bx1) - EB(i1, i2, em::bx1) +
                                EB(i1, i2, em::bx2) - EB(i1 - 1, i2, em::bx2));
      } else {
        raise::KernelError(HERE, "Ampere_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        EB(i1, i2, i3, em::ex1) += coeff1 * (EB(i1, i2, i3 - 1, em::bx2) -
                                             EB(i1, i2, i3, em::bx2) +
                                             EB(i1, i2, i3, em::bx3) -
                                             EB(i1, i2 - 1, i3, em::bx3));
        EB(i1, i2, i3, em::ex2) += coeff1 * (EB(i1 - 1, i2, i3, em::bx3) -
                                             EB(i1, i2, i3, em::bx3) +
                                             EB(i1, i2, i3, em::bx1) -
                                             EB(i1, i2, i3 - 1, em::bx1));
        EB(i1, i2, i3, em::ex3) += coeff1 * (EB(i1, i2 - 1, i3, em::bx1) -
                                             EB(i1, i2, i3, em::bx1) +
                                             EB(i1, i2, i3, em::bx2) -
                                             EB(i1 - 1, i2, i3, em::bx2));
      } else {
        raise::KernelError(HERE, "Ampere_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Add the currents to the E field (Minkowski).
   * @brief `coeff` includes metric coefficient.
   * @tparam D Dimension.
   */
  template <Dimension D, class C = NoCurrent_t>
  class CurrentsAmpere_kernel {
    static constexpr auto ExtCurrent = not std::is_same<C, NoCurrent_t>::value;
    ndfield_t<D, 6>       E;
    ndfield_t<D, 3>       J;
    // coeff = -dt * q0 / (B0 * V0)
    const real_t          coeff;
    const real_t          ppc0;
    const C               ext_current;
    real_t                x1min { ZERO };
    real_t                x2min { ZERO };
    real_t                x3min { ZERO };
    real_t                dx;

  public:
    CurrentsAmpere_kernel(const ndfield_t<D, 6>&    E,
                          const ndfield_t<D, 3>     J,
                          real_t                    coeff,
                          real_t                    ppc0,
                          const C&                  ext_current,
                          const std::vector<real_t> xmin,
                          real_t                    dx)
      : E { E }
      , J { J }
      , coeff { coeff }
      , ppc0 { ppc0 }
      , ext_current { ext_current }
      , x1min { xmin.size() > 0 ? xmin[0] : ZERO }
      , x2min { xmin.size() > 1 ? xmin[1] : ZERO }
      , x3min { xmin.size() > 2 ? xmin[2] : ZERO }
      , dx { dx } {}

    CurrentsAmpere_kernel(const ndfield_t<D, 6>& E,
                          const ndfield_t<D, 3>  J,
                          real_t                 coeff,
                          real_t                 inv_n0)
      : CurrentsAmpere_kernel { E, J, coeff, inv_n0, NoCurrent_t {}, {}, ZERO } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        if constexpr (ExtCurrent) {
          const auto i1_ = COORD(i1);
          J(i1, cur::jx1) += ppc0 * ext_current.jx1({ (i1_ + HALF) * dx + x1min });
          J(i1, cur::jx2) += ppc0 * ext_current.jx2({ i1_ * dx + x1min });
          J(i1, cur::jx3) += ppc0 * ext_current.jx3({ i1_ * dx + x1min });
        }
        E(i1, em::ex1) += J(i1, cur::jx1) * coeff;
        E(i1, em::ex2) += J(i1, cur::jx2) * coeff;
        E(i1, em::ex3) += J(i1, cur::jx3) * coeff;

        J(i1, cur::jx1) /= ppc0;
        J(i1, cur::jx2) /= ppc0;
        J(i1, cur::jx3) /= ppc0;
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpere_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        if constexpr (ExtCurrent) {
          const auto i1_ = COORD(i1);
          const auto i2_ = COORD(i2);
          J(i1, i2, cur::jx1) += ppc0 * ext_current.jx1({ (i1_ + HALF) * dx + x1min,
                                                          i2_ * dx + x2min });
          J(i1, i2, cur::jx2) += ppc0 *
                                 ext_current.jx2({ i1_ * dx + x1min,
                                                   (i2_ + HALF) * dx + x2min });
          J(i1, i2, cur::jx3) += ppc0 * ext_current.jx3({ i1_ * dx + x1min,
                                                          i2_ * dx + x2min });
        }
        E(i1, i2, em::ex1) += J(i1, i2, cur::jx1) * coeff;
        E(i1, i2, em::ex2) += J(i1, i2, cur::jx2) * coeff;
        E(i1, i2, em::ex3) += J(i1, i2, cur::jx3) * coeff;

        J(i1, i2, cur::jx1) /= ppc0;
        J(i1, i2, cur::jx2) /= ppc0;
        J(i1, i2, cur::jx3) /= ppc0;
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpere_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        if constexpr (ExtCurrent) {
          const auto i1_           = COORD(i1);
          const auto i2_           = COORD(i2);
          const auto i3_           = COORD(i3);
          J(i1, i2, i3, cur::jx1) += ppc0 *
                                     ext_current.jx1({ (i1_ + HALF) * dx + x1min,
                                                       i2_ * dx + x2min,
                                                       i3_ * dx + x3min });
          J(i1, i2, i3, cur::jx2) += ppc0 *
                                     ext_current.jx2({ i1_ * dx + x1min,
                                                       (i2_ + HALF) * dx + x2min,
                                                       i3_ * dx + x3min });
          J(i1, i2, i3, cur::jx3) += ppc0 * ext_current.jx3(
                                              { i1_ * dx + x1min,
                                                i2_ * dx + x2min,
                                                (i3_ + HALF) * dx + x3min });
        }
        E(i1, i2, i3, em::ex1) += J(i1, i2, i3, cur::jx1) * coeff;
        E(i1, i2, i3, em::ex2) += J(i1, i2, i3, cur::jx2) * coeff;
        E(i1, i2, i3, em::ex3) += J(i1, i2, i3, cur::jx3) * coeff;

        J(i1, i2, i3, cur::jx1) /= ppc0;
        J(i1, i2, i3, cur::jx2) /= ppc0;
        J(i1, i2, i3, cur::jx3) /= ppc0;
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpere_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::mink

#endif // KERNELS_AMPERE_MINK_HPP
