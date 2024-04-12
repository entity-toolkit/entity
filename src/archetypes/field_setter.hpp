/**
 * @file archetypes/field_setter.hpp
 * @brief Defines a class which populates the EM fields with user-defined values
 * @implements
 *   - ntt::SetEMFields
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/traits.h
 *   - archetypes/kokkos_aliases.h
 *   - utils/numeric.h
 * @namespaces:
 *   - ntt::
 * @note
 * The functor accepts a class I as a template argument, which must contain
 * one of the ex1, ex2, ex3, bx1, bx2, bx3 methods (dx1, dx2, dx3 for GR).
 * * `coord_t<D>` --> [ I ] --> `real_t` --> [ SetEMFields ] --> ...
 * *      ^                        ^                              ^
 * *  (physical)           (SR: hatted basis )            (SR & GR: cntrv)
 * *                       (GR: phys. cntrv  )
 * @note
 * The functor automatically takes care of the staggering, ghost cells and
 * conversion to contravariant basis.
 */

#ifndef ARCHETYPES_FIELD_SETTER_HPP
#define ARCHETYPES_FIELD_SETTER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>

namespace ntt {

  template <class I, SimEngine::type S, class M>
  class SetEMFields {
    static constexpr Dimension D           = M::Dim;
    static constexpr bool      defines_ex1 = traits::has_ex1<I>::value;
    static constexpr bool      defines_ex2 = traits::has_ex2<I>::value;
    static constexpr bool      defines_ex3 = traits::has_ex3<I>::value;
    static constexpr bool      defines_bx1 = traits::has_bx1<I>::value;
    static constexpr bool      defines_bx2 = traits::has_bx2<I>::value;
    static constexpr bool      defines_bx3 = traits::has_bx3<I>::value;
    static constexpr bool      defines_dx1 = traits::has_dx1<I>::value;
    static constexpr bool      defines_dx2 = traits::has_dx2<I>::value;
    static constexpr bool      defines_dx3 = traits::has_dx3<I>::value;

    static_assert(defines_ex1 || defines_ex2 || defines_ex3 || defines_bx1 ||
                    defines_bx2 || defines_bx3 || defines_dx1 || defines_dx2 ||
                    defines_dx3,
                  "No field initializer defined");
    static_assert(M::is_metric, "M must be a metric class");

    ndfield_t<M::Dim, 6> EM;
    const I              finit;
    const M              metric;

  public:
    SetEMFields(ndfield_t<M::Dim, 6>& EM, const I& finit, const M& metric) :
      EM { EM },
      finit { finit },
      metric { metric } {}

    ~SetEMFields() = default;

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto        i1_ = COORD(i1);
        coord_t<Dim::_1D> x_Phys { ZERO };
        if constexpr ((S == SimEngine::SRPIC) && defines_ex1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
          EM(i1, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_ + HALF },
            finit.ex1(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_ex2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
          EM(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_ },
            finit.ex2(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_ex3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
          EM(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_ },
            finit.ex3(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_bx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
          EM(i1, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_ },
            finit.bx1(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_bx2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
          EM(i1, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_ + HALF },
            finit.bx2(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_bx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
          EM(i1, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_ + HALF },
            finit.bx3(x_Phys));
        }
        if constexpr (S == SimEngine::GRPIC) {
          raise::KernelError(HERE, "Invalid SimEngine");
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const auto        i1_ = COORD(i1);
        const auto        i2_ = COORD(i2);
        coord_t<Dim::_2D> x_Phys { ZERO };
        // srpic
        if constexpr ((S == SimEngine::SRPIC) && defines_ex1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Phys);
          EM(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_ + HALF, i2_ },
            finit.ex1(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_ex2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Phys);
          EM(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_, i2_ + HALF },
            finit.ex2(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_ex3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Phys);
          EM(i1, i2, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_, i2_ },
            finit.ex3(x_Phys));
        }
        if constexpr (defines_bx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Phys);
          EM(i1, i2, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_, i2_ + HALF },
            finit.bx1(x_Phys));
        }
        if constexpr (defines_bx2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Phys);
          EM(i1, i2, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_ + HALF, i2_ },
            finit.bx2(x_Phys));
        }
        if constexpr (defines_bx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF },
                                                    x_Phys);
          EM(i1, i2, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_ + HALF, i2_ + HALF },
            finit.bx3(x_Phys));
        }
        // grpic
        if constexpr ((S == SimEngine::GRPIC) && defines_dx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Phys);
          EM(i1, i2, em::dx1) = metric.template transform<1, Idx::PU, Idx::U>(
            { i1_ + HALF, i2_ },
            finit.dx1(x_Phys));
        }
        if constexpr ((S == SimEngine::GRPIC) && defines_dx2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Phys);
          EM(i1, i2, em::dx2) = metric.template transform<2, Idx::PU, Idx::U>(
            { i1_, i2_ + HALF },
            finit.dx2(x_Phys));
        }
        if constexpr ((S == SimEngine::GRPIC) && defines_dx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Phys);
          EM(i1, i2, em::dx3) = metric.template transform<3, Idx::PU, Idx::U>(
            { i1_, i2_ },
            finit.dx3(x_Phys));
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        const auto        i1_ = COORD(i1);
        const auto        i2_ = COORD(i2);
        const auto        i3_ = COORD(i3);
        coord_t<Dim::_3D> x_Phys { ZERO };
        // srpic
        if constexpr ((S == SimEngine::SRPIC) && defines_ex1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ },
                                                    x_Phys);
          EM(i1, i2, i3, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_ + HALF, i2_, i3_ },
            finit.ex1(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_ex2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ },
                                                    x_Phys);
          EM(i1, i2, i3, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_, i2_ + HALF, i3_ },
            finit.ex2(x_Phys));
        }
        if constexpr ((S == SimEngine::SRPIC) && defines_ex3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF },
                                                    x_Phys);
          EM(i1, i2, i3, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_, i2_, i3_ + HALF },
            finit.ex3(x_Phys));
        }
        if constexpr (defines_bx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ + HALF },
                                                    x_Phys);
          EM(i1, i2, i3, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_, i2_ + HALF, i3_ + HALF },
            finit.bx1(x_Phys));
        }
        if constexpr (defines_bx2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ + HALF },
                                                    x_Phys);
          EM(i1, i2, i3, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_ + HALF, i2_, i3_ + HALF },
            finit.bx2(x_Phys));
        }
        if constexpr (defines_bx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF, i3_ },
                                                    x_Phys);
          EM(i1, i2, i3, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_ + HALF, i2_ + HALF, i3_ },
            finit.bx3(x_Phys));
        }
        // grpic
        if constexpr ((S == SimEngine::GRPIC) && defines_dx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ },
                                                    x_Phys);
          EM(i1, i2, i3, em::dx1) = metric.template transform<1, Idx::PU, Idx::U>(
            { i1_ + HALF, i2_, i3_ },
            finit.dx1(x_Phys));
        }
        if constexpr ((S == SimEngine::GRPIC) && defines_dx2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ },
                                                    x_Phys);
          EM(i1, i2, i3, em::dx2) = metric.template transform<2, Idx::PU, Idx::U>(
            { i1_, i2_ + HALF, i3_ },
            finit.dx2(x_Phys));
        }
        if constexpr ((S == SimEngine::GRPIC) && defines_dx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF },
                                                    x_Phys);
          EM(i1, i2, i3, em::dx3) = metric.template transform<3, Idx::PU, Idx::U>(
            { i1_, i2_, i3_ + HALF },
            finit.dx3(x_Phys));
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }
  };

} // namespace ntt

#endif // ARCHETYPES_FIELD_SETTER_HPP