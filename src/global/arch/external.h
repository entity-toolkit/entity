/**
 * @file arch/external.h
 * @brief Defines a helper class for the external force and EM fields
 * @implements
 *   - ext::NoForce_t
 *   - ext::Force<>
 * @namespaces:
 *   - ext::
 */

#ifndef GLOBAL_ARCH_EXTERNAL_H
#define GLOBAL_ARCH_EXTERNAL_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"

namespace ext {
  using namespace ntt;

  struct NoForce_t {
    static constexpr auto ExtForce  = false;
    static constexpr auto ExtFields = false;

    NoForce_t() {}
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
  template <Dimension D, Coord::type C, class F = NoForce_t, bool Atm = false>
  struct Force {
    static constexpr auto definesFx1 = traits::has_member<traits::fx1_t, F>::value;
    static constexpr auto definesFx2 = traits::has_member<traits::fx2_t, F>::value;
    static constexpr auto definesFx3 = traits::has_member<traits::fx3_t, F>::value;

    static constexpr auto ExtForce = definesFx1 || definesFx2 || definesFx3;

    static constexpr auto definesEx1 = traits::has_member<traits::ex1_t, F>::value;
    static constexpr auto definesEx2 = traits::has_member<traits::ex2_t, F>::value;
    static constexpr auto definesEx3 = traits::has_member<traits::ex3_t, F>::value;
    static constexpr auto definesBx1 = traits::has_member<traits::bx1_t, F>::value;
    static constexpr auto definesBx2 = traits::has_member<traits::bx2_t, F>::value;
    static constexpr auto definesBx3 = traits::has_member<traits::bx3_t, F>::value;

    static constexpr auto ExtFields = definesEx1 || definesEx2 || definesEx3 ||
                                      definesBx1 || definesBx2 || definesBx3;

    static_assert(
      ExtFields or ExtForce or Atm,
      "Force initialized with neither PGen force/ext fields nor gravity");

    const F      pgen_force;
    const real_t gx1, gx2, gx3, x_surf, ds;

    Force(const F& pgen_force, const vec_t<Dim::_3D>& g, real_t x_surf, real_t ds)
      : pgen_force { pgen_force }
      , gx1 { g[0] }
      , gx2 { g[1] }
      , gx3 { g[2] }
      , x_surf { x_surf }
      , ds { ds } {}

    Force(const F& pgen_force)
      : Force {
        pgen_force,
        {ZERO, ZERO, ZERO},
        ZERO,
        ZERO
    } {
      raise::ErrorIf(Atm, "Atmospheric gravity not provided", HERE);
    }

    Force(const vec_t<Dim::_3D>& g, real_t x_surf, real_t ds)
      : Force { NoForce_t {}, g, x_surf, ds } {
      raise::ErrorIf(ExtForce or ExtFields,
                     "External force/fields not provided",
                     HERE);
    }

    Inline auto fx1(const unsigned short& sp,
                    const real_t&         time,
                    bool                  apply_ext_force,
                    const coord_t<D>&     x_Ph) const -> real_t {
      real_t f_x1 = ZERO;
      if constexpr (definesFx1) {
        if (apply_ext_force) {
          f_x1 += pgen_force.fx1(sp, time, x_Ph);
        }
      }
      if constexpr (Atm) {
        if (gx1 != ZERO) {
          if ((ds > ZERO and x_Ph[0] >= x_surf + ds) or
              (ds < ZERO and x_Ph[0] <= x_surf + ds)) {
            return f_x1;
          }
          if constexpr (C == Coord::Cart) {
            return f_x1 + gx1;
          } else {
            return f_x1 + gx1 * SQR(x_surf / x_Ph[0]);
          }
        }
      }
      return f_x1;
    }

    Inline auto fx2(const unsigned short& sp,
                    const real_t&         time,
                    bool                  apply_ext_force,
                    const coord_t<D>&     x_Ph) const -> real_t {
      real_t f_x2 = ZERO;
      if constexpr (definesFx2) {
        if (apply_ext_force) {
          f_x2 += pgen_force.fx2(sp, time, x_Ph);
        }
      }
      if constexpr (Atm and (D == Dim::_2D or D == Dim::_3D)) {
        if (gx2 != ZERO) {
          if ((ds > ZERO and x_Ph[1] >= x_surf + ds) or
              (ds < ZERO and x_Ph[1] <= x_surf + ds)) {
            return f_x2;
          }
          if constexpr (C == Coord::Cart) {
            return f_x2 + gx2;
          } else {
            raise::KernelError(HERE, "Invalid force for coordinate system");
          }
        }
      }
      return f_x2;
    }

    Inline auto fx3(const unsigned short& sp,
                    const real_t&         time,
                    bool                  apply_ext_force,
                    const coord_t<D>&     x_Ph) const -> real_t {
      real_t f_x3 = ZERO;
      if constexpr (definesFx3) {
        if (apply_ext_force) {
          f_x3 += pgen_force.fx3(sp, time, x_Ph);
        }
      }
      if constexpr (Atm and D == Dim::_3D) {
        if (gx3 != ZERO) {
          if ((ds > ZERO and x_Ph[2] >= x_surf + ds) or
              (ds < ZERO and x_Ph[2] <= x_surf + ds)) {
            return f_x3;
          }
          if constexpr (C == Coord::Cart) {
            return f_x3 + gx3;
          } else {
            raise::KernelError(HERE, "Invalid force for coordinate system");
          }
        }
      }
      return f_x3;
    }

    Inline auto ex1(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      if constexpr (definesEx1) {
        return pgen_force.ex1(sp, time, x_Ph);
      } else {
        return ZERO;
      }
    }

    Inline auto ex2(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      if constexpr (definesEx2) {
        return pgen_force.ex2(sp, time, x_Ph);
      } else {
        return ZERO;
      }
    }

    Inline auto ex3(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      if constexpr (definesEx3) {
        return pgen_force.ex3(sp, time, x_Ph);
      } else {
        return ZERO;
      }
    }

    Inline auto bx1(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      if constexpr (definesBx1) {
        return pgen_force.bx1(sp, time, x_Ph);
      } else {
        return ZERO;
      }
    }

    Inline auto bx2(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      if constexpr (definesBx2) {
        return pgen_force.bx2(sp, time, x_Ph);
      } else {
        return ZERO;
      }
    }

    Inline auto bx3(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      if constexpr (definesBx3) {
        return pgen_force.bx3(sp, time, x_Ph);
      } else {
        return ZERO;
      }
    }
  };

} // namespace ext

#endif // ARCHETYPES_EXTERNAL_H
