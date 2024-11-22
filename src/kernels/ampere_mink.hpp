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

namespace kernel::mink {
  using namespace ntt;

  struct NoCurrent_t {
    NoCurrent_t() {}
  };

  /**
   * @brief
   * A helper struct which combines the atmospheric gravity
   * with (optionally) custom user-defined force
   * @tparam D Dimension
   * @tparam C Coordinate system
   * @tparam F Additional force
   * @tparam Atm Toggle for atmospheric gravity
   * @note when `Atm` is true, `g` contains a vector of gravity acceleration
   * @note when `Atm` is true, sign of `ds` indicates the direction of the boundary
   * !TODO: compensate for the species mass when applying atmospheric force
   */
  template <Dimension D, Coord::type C, class Cu = NoCurrent_t>
  struct Current {
    static constexpr auto ExtCurrent = not std::is_same<Cu, NoCurrent_t>::value;
    const Cu      pgen_current;

    Current(const Cu& pgen_current)
      : pgen_current { pgen_current } {}

    Inline auto jx1(const coord_t<D>&     x_Ph) const -> real_t {
      real_t j_x1 = ZERO;
      if constexpr (ExtCurrent) {
          j_x1 += pgen_current.jx1(x_Ph);
      }

      return j_x1;
    }

    Inline auto jx2(const coord_t<D>&     x_Ph) const -> real_t {
      real_t j_x2 = ZERO;
      if constexpr (ExtCurrent) {
          j_x2 += pgen_current.jx2(x_Ph);
      }
      return j_x2;
    }

    Inline auto jx3(const coord_t<D>&     x_Ph) const -> real_t {
      real_t j_x3 = ZERO;
      if constexpr (ExtCurrent) {
        j_x3 += pgen_current.jx3(x_Ph);
      }

      return j_x3;
    }
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
  template <class M, class Cu = NoCurrent_t>
  struct CurrentsAmpere_kernel {
    static constexpr auto        D     = M::Dim;
    static constexpr auto ExtCurrent = not std::is_same<Cu, NoCurrent_t>::value;
    ndfield_t<D, 6> E;
    ndfield_t<D, 3> J;
    const M           metric;
    // coeff = -dt * q0 * n0 / (B0 * V0)
    const real_t    coeff;
    const real_t    inv_n0;
    const real_t    V0;
    const Cu current;

  public:
    CurrentsAmpere_kernel(const ndfield_t<D, 6>& E,
                          const ndfield_t<D, 3>  J,
                          const M&               metric,
                          real_t                 coeff,
                          real_t                 inv_n0,
                          real_t                 V0,
                          const Cu&              current)
      : E { E }
      , J { J }
      , coeff { coeff }
      , inv_n0 { inv_n0 } 
      , V0 { V0 }
      , metric { metric }
      , current {current} {}

    CurrentsAmpere_kernel(const ndfield_t<D, 6>& E,
                          const ndfield_t<D, 3>  J,
                          const M&              metric,
                          real_t                 coeff,
                          real_t                 inv_n0)
      : CurrentsAmpere_kernel(E, J, metric, coeff, inv_n0, ZERO, NoCurrent_t {}) {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        J(i1, cur::jx1) *= inv_n0;
        J(i1, cur::jx2) *= inv_n0;
        J(i1, cur::jx3) *= inv_n0;

        E(i1, em::ex1) += J(i1, cur::jx1) * coeff;
        E(i1, em::ex2) += J(i1, cur::jx2) * coeff;
        E(i1, em::ex3) += J(i1, cur::jx3) * coeff;
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpere_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
      vec_t<Dim::_3D> current_Cd { ZERO };

      if constexpr (ExtCurrent) {
        coord_t<Dim::_2D> xp_Ph { ZERO };
        coord_t<Dim::_2D> xp_Cd { i1, i2 };
        xp_Ph[0] = metric.template convert<1, Crd::Cd, Crd::Ph>(xp_Cd[0]);
        xp_Ph[1] = metric.template convert<2, Crd::Cd, Crd::Ph>(xp_Cd[1]);
        metric.template transform_xyz<Idx::XYZ, Idx::U>(
          xp_Ph,
          { current.jx1(xp_Ph),
            current.jx2(xp_Ph),
            current.jx3(xp_Ph) },
          current_Cd);
      }

        // J(i1, i2, cur::jx1) *= inv_n0;
        // J(i1, i2, cur::jx2) *= inv_n0;
        // J(i1, i2, cur::jx3) *= inv_n0;

        J(i1, i2, cur::jx1) *= ZERO;
        J(i1, i2, cur::jx2) *= ZERO;
        J(i1, i2, cur::jx3) *= ZERO;

        J(i1, i2, cur::jx1) += current_Cd[0] * V0;
        J(i1, i2, cur::jx2) += current_Cd[1] * V0;
        J(i1, i2, cur::jx3) += current_Cd[2] * V0;

        E(i1, i2, em::ex1) += J(i1, i2, cur::jx1) * coeff;
        E(i1, i2, em::ex2) += J(i1, i2, cur::jx2) * coeff;
        E(i1, i2, em::ex3) += J(i1, i2, cur::jx3) * coeff;

      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpere_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        J(i1, i2, i3, cur::jx1) *= inv_n0;
        J(i1, i2, i3, cur::jx2) *= inv_n0;
        J(i1, i2, i3, cur::jx3) *= inv_n0;

        E(i1, i2, i3, em::ex1) += J(i1, i2, i3, cur::jx1) * coeff;
        E(i1, i2, i3, em::ex2) += J(i1, i2, i3, cur::jx2) * coeff;
        E(i1, i2, i3, em::ex3) += J(i1, i2, i3, cur::jx3) * coeff;
        
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpere_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::mink

#endif // KERNELS_AMPERE_MINK_HPP
