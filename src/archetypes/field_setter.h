/**
 * @file archetypes/field_setter.h
 * @brief Defines kernels which populate the EM fields with user-defined values
 * conditionally or unconditionally
 * @implements
 *   - arch::SetEMFields_kernel<>
 *   - arch::CustomSetEMFields_kernel<>
 * @namespaces:
 *   - arch::
 * @note
 * The functors accept a class F as a template argument, which must contain
 * one of the ex1, ex2, ex3, bx1, bx2, bx3 methods (dx1, dx2, dx3 for GR).
 * * `coord_t<D>` --> [ I ] --> `real_t` --> [ SetEMFields_kernel ] --> ...
 * *      ^                        ^                                     ^
 * *  (physical)           (SR: tetrad basis )                  (SR & GR: cntrv)
 * *                       (GR: phys. cntrv  )
 * @note
 * The functors automatically take care of the staggering, ghost cells and
 * conversion to contravariant basis.
 */

#ifndef ARCHETYPES_FIELD_SETTER_HPP
#define ARCHETYPES_FIELD_SETTER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/archetypes.h"
#include "traits/engine.h"
#include "traits/metric.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>

namespace arch {
  using namespace ntt;

  template <SimEngine::type S, MetricClass M, FieldSetterClass<S, M::Dim> FS>
  class SetEMFields_kernel {
    static constexpr auto D      = M::Dim;
    static constexpr auto HasEx1 = ::traits::fieldsetter::HasEx1<FS, D>;
    static constexpr auto HasEx2 = ::traits::fieldsetter::HasEx2<FS, D>;
    static constexpr auto HasEx3 = ::traits::fieldsetter::HasEx3<FS, D>;
    static constexpr auto HasBx1 = ::traits::fieldsetter::HasBx1<FS, D>;
    static constexpr auto HasBx2 = ::traits::fieldsetter::HasBx2<FS, D>;
    static constexpr auto HasBx3 = ::traits::fieldsetter::HasBx3<FS, D>;
    static constexpr auto HasDx1 = ::traits::fieldsetter::HasDx1<FS, D>;
    static constexpr auto HasDx2 = ::traits::fieldsetter::HasDx2<FS, D>;
    static constexpr auto HasDx3 = ::traits::fieldsetter::HasDx3<FS, D>;

    ndfield_t<M::Dim, 6> EM;
    const FS             finit;
    const M              metric;

  public:
    SetEMFields_kernel(ndfield_t<M::Dim, 6>& EM, const FS& finit, const M& metric)
      : EM { EM }
      , finit { finit }
      , metric { metric } {}

    ~SetEMFields_kernel() = default;

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto        i1_ = COORD(i1);
        coord_t<Dim::_1D> x_Phys { ZERO };
        if constexpr (::traits::engine::UserFieldsInTetradBasis<S>) {
          if constexpr (HasEx1) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
            EM(i1, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF },
              finit.ex1(x_Phys));
          }
          if constexpr (HasEx2) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
            EM(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ },
              finit.ex2(x_Phys));
          }
          if constexpr (HasEx3) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
            EM(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ },
              finit.ex3(x_Phys));
          }
          if constexpr (HasBx1) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Phys);
            EM(i1, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ },
              finit.bx1(x_Phys));
          }
          if constexpr (HasBx2) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
            EM(i1, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF },
              finit.bx2(x_Phys));
          }
          if constexpr (HasBx3) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Phys);
            EM(i1, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF },
              finit.bx3(x_Phys));
          }
        } else {
          raise::KernelError(HERE, "Invalid SimEngine");
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);
        // srpic
        if constexpr (::traits::engine::UserFieldsInTetradBasis<S>) {
          coord_t<Dim::_2D> x_Phys { ZERO };
          if constexpr (HasEx1) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Phys);
            EM(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ },
              finit.ex1(x_Phys));
          }
          if constexpr (HasEx2) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Phys);
            EM(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF },
              finit.ex2(x_Phys));
          }
          if constexpr (HasEx3) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Phys);
            EM(i1, i2, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_ },
              finit.ex3(x_Phys));
          }
          if constexpr (HasBx1) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Phys);
            EM(i1, i2, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF },
              finit.bx1(x_Phys));
          }
          if constexpr (HasBx2) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Phys);
            EM(i1, i2, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ },
              finit.bx2(x_Phys));
          }
          if constexpr (HasBx3) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF },
                                                      x_Phys);
            EM(i1, i2, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF },
              finit.bx3(x_Phys));
          }
        } else if constexpr (::traits::engine::UserFieldsInContravariantBasis<S>) {
          // grpic
          if constexpr (HasDx1 && HasDx2 && HasDx3) {
            const real_t x1_0 { metric.template convert<1, Crd::Cd, Crd::Ph>(i1_) };
            const real_t x1_H { metric.template convert<1, Crd::Cd, Crd::Ph>(
              i1_ + HALF) };
            const real_t x2_0 { metric.template convert<2, Crd::Cd, Crd::Ph>(i2_) };
            const real_t x2_H { metric.template convert<2, Crd::Cd, Crd::Ph>(
              i2_ + HALF) };
            { // dx1
              EM(i1, i2, em::dx1) = finit.dx1({ x1_H, x2_0 });
            }
            { // dx2
              EM(i1, i2, em::dx2) = finit.dx2({ x1_0, x2_H });
            }
            { // dx3
              EM(i1, i2, em::dx3) = finit.dx3({ x1_0, x2_0 });
            }
          }
          if constexpr (HasBx1 && HasBx2 && HasBx3) {
            const real_t x1_0 { metric.template convert<1, Crd::Cd, Crd::Ph>(i1_) };
            const real_t x1_H { metric.template convert<1, Crd::Cd, Crd::Ph>(
              i1_ + HALF) };
            const real_t x2_0 { metric.template convert<2, Crd::Cd, Crd::Ph>(i2_) };
            const real_t x2_H { metric.template convert<2, Crd::Cd, Crd::Ph>(
              i2_ + HALF) };
            { // bx1
              EM(i1, i2, em::bx1) = finit.bx1({ x1_0, x2_H });
            }
            { // bx2
              EM(i1, i2, em::bx2) = finit.bx2({ x1_H, x2_0 });
            }
            { // bx3
              EM(i1, i2, em::bx3) = finit.bx3({ x1_H, x2_H });
            }
          }
        } else {
          raise::KernelError(HERE, "Invalid SimEngine");
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        const auto        i1_ = COORD(i1);
        const auto        i2_ = COORD(i2);
        const auto        i3_ = COORD(i3);
        coord_t<Dim::_3D> x_Phys { ZERO };
        if constexpr (::traits::engine::UserFieldsInTetradBasis<S>) {
          // srpic
          if constexpr (HasEx1) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ },
                                                      x_Phys);
            EM(i1, i2, i3, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ },
              finit.ex1(x_Phys));
          }
          if constexpr (HasEx2) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ },
                                                      x_Phys);
            EM(i1, i2, i3, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ },
              finit.ex2(x_Phys));
          }
          if constexpr (HasEx3) {
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF },
                                                      x_Phys);
            EM(i1, i2, i3, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_, i3_ + HALF },
              finit.ex3(x_Phys));
          }
          if constexpr (HasBx1) {
            metric.template convert<Crd::Cd, Crd::Ph>(
              { i1_, i2_ + HALF, i3_ + HALF },
              x_Phys);
            EM(i1, i2, i3, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ + HALF },
              finit.bx1(x_Phys));
          }
          if constexpr (HasBx2) {
            metric.template convert<Crd::Cd, Crd::Ph>(
              { i1_ + HALF, i2_, i3_ + HALF },
              x_Phys);
            EM(i1, i2, i3, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ + HALF },
              finit.bx2(x_Phys));
          }
          if constexpr (HasBx3) {
            metric.template convert<Crd::Cd, Crd::Ph>(
              { i1_ + HALF, i2_ + HALF, i3_ },
              x_Phys);
            EM(i1, i2, i3, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF, i3_ },
              finit.bx3(x_Phys));
          }
        } else if constexpr (::traits::engine::UserFieldsInContravariantBasis<S>) {
          // grpic
          const real_t x1_0 { metric.template convert<1, Crd::Cd, Crd::Ph>(i1_) };
          const real_t x1_H { metric.template convert<1, Crd::Cd, Crd::Ph>(
            i1_ + HALF) };
          const real_t x2_0 { metric.template convert<2, Crd::Cd, Crd::Ph>(i2_) };
          const real_t x2_H { metric.template convert<2, Crd::Cd, Crd::Ph>(
            i2_ + HALF) };
          const real_t x3_0 { metric.template convert<3, Crd::Cd, Crd::Ph>(i3_) };
          const real_t x3_H { metric.template convert<3, Crd::Cd, Crd::Ph>(
            i3_ + HALF) };

          if constexpr (HasDx1 && HasDx2 && HasDx3) {
            { // dx1
              EM(i1, i2, i3, em::dx1) = finit.dx1({ x1_H, x2_0, x3_0 });
            }
            { // dx2
              EM(i1, i2, i3, em::dx2) = finit.dx2({ x1_0, x2_H, x3_0 });
            }
            { // dx3
              EM(i1, i2, i3, em::dx3) = finit.dx3({ x1_0, x2_0, x3_H });
            }
          }
          if constexpr (HasBx1 && HasBx2 && HasBx3) {
            { // bx1
              EM(i1, i2, i3, em::bx1) = finit.bx1({ x1_0, x2_H, x3_H });
            }
            { // bx2
              EM(i1, i2, i3, em::bx2) = finit.bx2({ x1_H, x2_0, x3_H });
            }
            { // bx3
              EM(i1, i2, i3, em::bx3) = finit.bx3({ x1_H, x2_H, x3_0 });
            }
          }
        } else {
          raise::KernelError(HERE, "Invalid SimEngine");
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }
  };

  template <SimEngine::type S, SRMetricClass M, ConditionalSRFieldSetterClass<M::Dim> FS>
  struct CustomSetEMFields_kernel {
    static_assert(::traits::engine::UserFieldsInTetradBasis<S>,
                  "CustomSetEMFields_kernel is only implemented for SimEngines "
                  "with user-defined fields in tetrad basis");
    static constexpr auto D = M::Dim;
    static constexpr auto HasEx1 = ::traits::fieldsetter::HasConditionalEx1<FS, D>;
    static constexpr auto HasEx2 = ::traits::fieldsetter::HasConditionalEx2<FS, D>;
    static constexpr auto HasEx3 = ::traits::fieldsetter::HasConditionalEx3<FS, D>;
    static constexpr auto HasBx1 = ::traits::fieldsetter::HasConditionalBx1<FS, D>;
    static constexpr auto HasBx2 = ::traits::fieldsetter::HasConditionalBx2<FS, D>;
    static constexpr auto HasBx3 = ::traits::fieldsetter::HasConditionalBx3<FS, D>;

    M               metric;
    ndfield_t<D, 6> fields;
    ndfield_t<D, 6> buffer;

    const FS fieldsetter;

    CustomSetEMFields_kernel(const M&         metric,
                             ndfield_t<D, 6>& fields,
                             const ndfield_t<D, 6>& /*buffer*/,
                             const FS& fieldsetter)
      : metric { metric }
      , fields { fields }
      , fieldsetter { fieldsetter } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto i1_ = COORD(i1);
        coord_t<D> x_Ph { ZERO };

        vec_t<Dim::_3D> e_U { ZERO };
        vec_t<Dim::_3D> b_U { ZERO };

        vec_t<Dim::_3D> e_T { ZERO };
        vec_t<Dim::_3D> b_T { ZERO };

        if constexpr (HasEx1 or HasBx2 or HasBx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Ph);

          e_U[0] = buffer(i1, em::ex1);
          e_U[1] = HALF * (buffer(i1, em::ex2) + buffer(i1 + 1, em::ex2));
          e_U[2] = HALF * (buffer(i1, em::ex3) + buffer(i1 + 1, em::ex3));

          b_U[0] = HALF * (buffer(i1, em::bx1) + buffer(i1 + 1, em::bx1));
          b_U[1] = buffer(i1, em::bx2);
          b_U[2] = buffer(i1, em::bx3);

          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF }, e_U, e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF }, b_U, b_T);

          if constexpr (HasEx1) {
            const auto ex1_setter = fieldsetter.ex1(x_Ph, e_T, b_T);

            if (ex1_setter.first) {
              fields(i1, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
                { i1_ + HALF },
                ex1_setter.second);
            }
          }
          if constexpr (HasBx2) {
            const auto bx2_setter = fieldsetter.bx2(x_Ph, e_T, b_T);

            if (bx2_setter.first) {
              fields(i1, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
                { i1_ + HALF },
                bx2_setter.second);
            }
          }
          if constexpr (HasBx3) {
            const auto bx3_setter = fieldsetter.bx3(x_Ph, e_T, b_T);

            if (bx3_setter.first) {
              fields(i1, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
                { i1_ + HALF },
                bx3_setter.second);
            }
          }
        }
        if constexpr (HasEx2 or HasEx3 or HasBx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Ph);

          e_U[0] = HALF * (buffer(i1, em::ex1) + buffer(i1 - 1, em::ex1));
          e_U[1] = buffer(i1, em::ex2);
          e_U[2] = buffer(i1, em::ex3);

          b_U[0] = buffer(i1, em::bx1);
          b_U[1] = HALF * (buffer(i1, em::bx2) + buffer(i1 - 1, em::bx2));
          b_U[2] = HALF * (buffer(i1, em::bx3) + buffer(i1 - 1, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_ }, e_U, e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ }, b_U, b_T);

          if constexpr (HasEx2) {
            const auto ex2_setter = fieldsetter.ex2(x_Ph, e_T, b_T);

            if (ex2_setter.first) {
              fields(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
                { i1_ },
                ex2_setter.second);
            }
          }
          if constexpr (HasEx3) {
            const auto ex3_setter = fieldsetter.ex3(x_Ph, e_T, b_T);

            if (ex3_setter.first) {
              fields(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
                { i1_ },
                ex3_setter.second);
            }
          }
          if constexpr (HasBx1) {
            const auto bx1_setter = fieldsetter.bx1(x_Ph, e_T, b_T);

            if (bx1_setter.first) {
              fields(i1, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
                { i1_ },
                bx1_setter.second);
            }
          }
        }
      } else {
        raise::KernelError(HERE, "CustomEMFields_kernel 1D called for 2D/3D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);
        coord_t<D> x_Ph { ZERO };

        vec_t<Dim::_3D> e_U { ZERO };
        vec_t<Dim::_3D> b_U { ZERO };

        vec_t<Dim::_3D> e_T { ZERO };
        vec_t<Dim::_3D> b_T { ZERO };

        if constexpr (HasEx1 or HasBx2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Ph);

          e_U[0] = buffer(i1, i2, em::ex1);
          e_U[1] = INV_4 * (buffer(i1, i2, em::ex2) + buffer(i1 + 1, i2, em::ex2) +
                            buffer(i1, i2 - 1, em::ex2) +
                            buffer(i1 + 1, i2 - 1, em::ex2));
          e_U[2] = HALF * (buffer(i1, i2, em::ex3) + buffer(i1 + 1, i2, em::ex3));

          b_U[0] = INV_4 * (buffer(i1, i2, em::bx1) + buffer(i1 + 1, i2, em::bx1) +
                            buffer(i1, i2 - 1, em::bx1) +
                            buffer(i1 + 1, i2 - 1, em::bx1));
          b_U[1] = buffer(i1, i2, em::bx2);
          b_U[2] = HALF * (buffer(i1, i2, em::bx3) + buffer(i1, i2 - 1, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_ }, e_U, e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_ }, b_U, b_T);
          if constexpr (HasEx1) {
            const auto ex1_setter = fieldsetter.ex1(x_Ph, e_T, b_T);

            if (ex1_setter.first) {
              fields(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
                { i1_ + HALF, i2_ },
                ex1_setter.second);
            }
          }
          if constexpr (HasBx2) {
            const auto bx2_setter = fieldsetter.bx2(x_Ph, e_T, b_T);

            if (bx2_setter.first) {
              fields(i1, i2, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
                { i1_ + HALF, i2_ },
                bx2_setter.second);
            }
          }
        }
        if constexpr (HasEx2 or HasBx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Ph);

          e_U[0] = INV_4 * (buffer(i1, i2, em::ex1) + buffer(i1 - 1, i2, em::ex1) +
                            buffer(i1, i2 + 1, em::ex1) +
                            buffer(i1 - 1, i2 + 1, em::ex1));
          e_U[1] = buffer(i1, i2, em::ex2);
          e_U[2] = HALF * (buffer(i1, i2, em::ex3) + buffer(i1, i2 + 1, em::ex3));

          b_U[0] = buffer(i1, i2, em::bx1);
          b_U[1] = INV_4 * (buffer(i1, i2, em::bx2) + buffer(i1 - 1, i2, em::bx2) +
                            buffer(i1, i2 + 1, em::bx2) +
                            buffer(i1 - 1, i2 + 1, em::bx2));
          b_U[2] = HALF * (buffer(i1, i2, em::bx3) + buffer(i1 - 1, i2, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_, i2_ + HALF }, e_U, e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_, i2_ + HALF }, b_U, b_T);
          if constexpr (HasEx2) {
            const auto ex2_setter = fieldsetter.ex2(x_Ph, e_T, b_T);

            if (ex2_setter.first) {
              fields(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
                { i1_, i2_ + HALF },
                ex2_setter.second);
            }
          }
          if constexpr (HasBx1) {
            const auto bx1_setter = fieldsetter.bx1(x_Ph, e_T, b_T);

            if (bx1_setter.first) {
              fields(i1, i2, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
                { i1_, i2_ + HALF },
                bx1_setter.second);
            }
          }
        }
        if constexpr (HasEx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Ph);

          e_U[0] = HALF * (buffer(i1, i2, em::ex1) + buffer(i1 - 1, i2, em::ex1));
          e_U[1] = HALF * (buffer(i1, i2, em::ex2) + buffer(i1, i2 - 1, em::ex2));
          e_U[2] = buffer(i1, i2, em::ex3);

          b_U[0] = HALF * (buffer(i1, i2, em::bx1) + buffer(i1, i2 - 1, em::bx1));
          b_U[1] = HALF * (buffer(i1, i2, em::bx2) + buffer(i1 - 1, i2, em::bx2));
          b_U[2] = INV_4 * (buffer(i1, i2, em::bx3) + buffer(i1 - 1, i2, em::bx3) +
                            buffer(i1, i2 - 1, em::bx3) +
                            buffer(i1 - 1, i2 - 1, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_, i2_ }, e_U, e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_, i2_ }, b_U, b_T);
          const auto ex3_setter = fieldsetter.ex3(x_Ph, e_T, b_T);

          if (ex3_setter.first) {
            fields(i1, i2, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_ },
              ex3_setter.second);
          }
        }
        if constexpr (HasBx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF },
                                                    x_Ph);

          e_U[0] = HALF * (buffer(i1, i2, em::ex1) + buffer(i1, i2 + 1, em::ex1));
          e_U[1] = HALF * (buffer(i1, i2, em::ex2) + buffer(i1 + 1, i2, em::ex2));
          e_U[2] = INV_4 * (buffer(i1, i2, em::ex3) + buffer(i1 + 1, i2, em::ex3) +
                            buffer(i1, i2 + 1, em::ex3) +
                            buffer(i1 + 1, i2 + 1, em::ex3));

          b_U[0] = HALF * (buffer(i1, i2, em::bx1) + buffer(i1 + 1, i2, em::bx1));
          b_U[1] = HALF * (buffer(i1, i2, em::bx2) + buffer(i1, i2 + 1, em::bx2));
          b_U[2] = buffer(i1, i2, em::bx3);

          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_ + HALF },
                                                    e_U,
                                                    e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_ + HALF },
                                                    b_U,
                                                    b_T);
          const auto bx3_setter = fieldsetter.bx3(x_Ph, e_T, b_T);

          if (bx3_setter.first) {
            fields(i1, i2, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF },
              bx3_setter.second);
          }
        }
      } else {
        raise::KernelError(HERE, "CustomEMFields_kernel 2D called for 1D/3D");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);
        const auto i3_ = COORD(i3);
        coord_t<D> x_Ph { ZERO };

        vec_t<Dim::_3D> e_U { ZERO };
        vec_t<Dim::_3D> b_U { ZERO };

        vec_t<Dim::_3D> e_T { ZERO };
        vec_t<Dim::_3D> b_T { ZERO };
        if constexpr (HasEx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ }, x_Ph);

          e_U[0] = buffer(i1, i2, i3, em::ex1);
          e_U[1] = INV_4 * (buffer(i1, i2, i3, em::ex2) +
                            buffer(i1 + 1, i2, i3, em::ex2) +
                            buffer(i1, i2 - 1, i3, em::ex2) +
                            buffer(i1 + 1, i2 - 1, i3, em::ex2));
          e_U[2] = INV_4 * (buffer(i1, i2, i3, em::ex3) +
                            buffer(i1 + 1, i2, i3, em::ex3) +
                            buffer(i1, i2, i3 - 1, em::ex3) +
                            buffer(i1 + 1, i2, i3 - 1, em::ex3));

          b_U[0] = INV_8 * (buffer(i1, i2, i3, em::bx1) +
                            buffer(i1 + 1, i2, i3, em::bx1) +
                            buffer(i1, i2 - 1, i3, em::bx1) +
                            buffer(i1 + 1, i2 - 1, i3, em::bx1) +
                            buffer(i1, i2, i3 - 1, em::bx1) +
                            buffer(i1 + 1, i2, i3 - 1, em::bx1) +
                            buffer(i1, i2 - 1, i3 - 1, em::bx1) +
                            buffer(i1 + 1, i2 - 1, i3 - 1, em::bx1));
          b_U[1] = HALF * (buffer(i1, i2, i3, em::bx2) +
                           buffer(i1, i2, i3 - 1, em::bx2));
          b_U[2] = HALF * (buffer(i1, i2, i3, em::bx3) +
                           buffer(i1, i2 - 1, i3, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_, i3_ },
                                                    e_U,
                                                    e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_, i3_ },
                                                    b_U,
                                                    b_T);
          const auto ex1_setter = fieldsetter.ex1(x_Ph, e_T, b_T);

          if (ex1_setter.first) {
            fields(i1, i2, i3, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ },
              ex1_setter.second);
          }
        }
        if constexpr (HasEx2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ }, x_Ph);

          e_U[0] = INV_4 * (buffer(i1, i2, i3, em::ex1) +
                            buffer(i1 - 1, i2, i3, em::ex1) +
                            buffer(i1, i2 + 1, i3, em::ex1) +
                            buffer(i1 - 1, i2 + 1, i3, em::ex1));
          e_U[1] = buffer(i1, i2, i3, em::ex2);
          e_U[2] = INV_4 * (buffer(i1, i2, i3, em::ex3) +
                            buffer(i1, i2 + 1, i3, em::ex3) +
                            buffer(i1, i2, i3 - 1, em::ex3) +
                            buffer(i1, i2 + 1, i3 - 1, em::ex3));

          b_U[0] = HALF * (buffer(i1, i2, i3, em::bx1) +
                           buffer(i1, i2, i3 - 1, em::bx1));
          b_U[1] = INV_8 * (buffer(i1, i2, i3, em::bx2) +
                            buffer(i1 - 1, i2, i3, em::bx2) +
                            buffer(i1, i2 + 1, i3, em::bx2) +
                            buffer(i1 - 1, i2 + 1, i3, em::bx2) +
                            buffer(i1, i2, i3 - 1, em::bx2) +
                            buffer(i1 - 1, i2, i3 - 1, em::bx2) +
                            buffer(i1, i2 + 1, i3 - 1, em::bx2) +
                            buffer(i1 - 1, i2 + 1, i3 - 1, em::bx2));
          b_U[2] = HALF * (buffer(i1, i2, i3, em::bx3) +
                           buffer(i1 - 1, i2, i3, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_, i2_ + HALF, i3_ },
                                                    e_U,
                                                    e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_, i2_ + HALF, i3_ },
                                                    b_U,
                                                    b_T);
          const auto ex2_setter = fieldsetter.ex2(x_Ph, e_T, b_T);

          if (ex2_setter.first) {
            fields(i1, i2, i3, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ },
              ex2_setter.second);
          }
        }
        if constexpr (HasEx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF }, x_Ph);

          e_U[0] = INV_4 * (buffer(i1, i2, i3, em::ex1) +
                            buffer(i1 - 1, i2, i3, em::ex1) +
                            buffer(i1, i2, i3 + 1, em::ex1) +
                            buffer(i1 - 1, i2, i3 + 1, em::ex1));
          e_U[1] = INV_4 * (buffer(i1, i2, i3, em::ex2) +
                            buffer(i1, i2 - 1, i3, em::ex2) +
                            buffer(i1, i2, i3 + 1, em::ex2) +
                            buffer(i1, i2 - 1, i3 + 1, em::ex2));
          e_U[2] = buffer(i1, i2, i3, em::ex3);

          b_U[0] = HALF * (buffer(i1, i2, i3, em::bx1) +
                           buffer(i1, i2 - 1, i3, em::bx1));
          b_U[1] = HALF * (buffer(i1, i2, i3, em::bx2) +
                           buffer(i1 - 1, i2, i3, em::bx2));
          b_U[2] = INV_8 * (buffer(i1, i2, i3, em::bx3) +
                            buffer(i1 - 1, i2, i3, em::bx3) +
                            buffer(i1, i2, i3 + 1, em::bx3) +
                            buffer(i1 - 1, i2, i3 + 1, em::bx3) +
                            buffer(i1, i2 - 1, i3, em::bx3) +
                            buffer(i1 - 1, i2 - 1, i3, em::bx3) +
                            buffer(i1, i2 - 1, i3 + 1, em::bx3) +
                            buffer(i1 - 1, i2 - 1, i3 + 1, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_, i2_, i3_ + HALF },
                                                    e_U,
                                                    e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_, i2_, i3_ + HALF },
                                                    b_U,
                                                    b_T);
          const auto ex3_setter = fieldsetter.ex3(x_Ph, e_T, b_T);

          if (ex3_setter.first) {
            fields(i1, i2, i3, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_, i3_ + HALF },
              ex3_setter.second);
          }
        }
        if constexpr (HasBx1) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ + HALF },
                                                    x_Ph);

          e_U[0] = INV_8 * (buffer(i1, i2, i3, em::ex1) +
                            buffer(i1 - 1, i2, i3, em::ex1) +
                            buffer(i1, i2 + 1, i3, em::ex1) +
                            buffer(i1 - 1, i2 + 1, i3, em::ex1) +
                            buffer(i1, i2, i3 + 1, em::ex1) +
                            buffer(i1 - 1, i2, i3 + 1, em::ex1) +
                            buffer(i1, i2 + 1, i3 + 1, em::ex1) +
                            buffer(i1 - 1, i2 + 1, i3 + 1, em::ex1));
          e_U[1] = HALF * (buffer(i1, i2, i3, em::ex2) +
                           buffer(i1, i2, i3 + 1, em::ex2));
          e_U[2] = HALF * (buffer(i1, i2, i3, em::ex3) +
                           buffer(i1, i2 + 1, i3, em::ex3));

          b_U[0] = buffer(i1, i2, i3, em::bx1);
          b_U[1] = INV_4 * (buffer(i1, i2, i3, em::bx2) +
                            buffer(i1 - 1, i2, i3, em::bx2) +
                            buffer(i1, i2 + 1, i3, em::bx2) +
                            buffer(i1 - 1, i2 + 1, i3, em::bx2));
          b_U[2] = INV_4 * (buffer(i1, i2, i3, em::bx3) +
                            buffer(i1 - 1, i2, i3, em::bx3) +
                            buffer(i1, i2, i3 + 1, em::bx3) +
                            buffer(i1 - 1, i2, i3 + 1, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_, i2_ + HALF, i3_ + HALF },
                                                    e_U,
                                                    e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_, i2_ + HALF, i3_ + HALF },
                                                    b_U,
                                                    b_T);
          const auto bx1_setter = fieldsetter.bx1(x_Ph, e_T, b_T);

          if (bx1_setter.first) {
            fields(i1, i2, i3, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ + HALF },
              bx1_setter.second);
          }
        }
        if constexpr (HasBx2) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ + HALF },
                                                    x_Ph);

          e_U[0] = HALF * (buffer(i1, i2, i3, em::ex1) +
                           buffer(i1, i2, i3 + 1, em::ex1));
          e_U[1] = INV_8 * (buffer(i1, i2, i3, em::ex2) +
                            buffer(i1 + 1, i2, i3, em::ex2) +
                            buffer(i1, i2 - 1, i3, em::ex2) +
                            buffer(i1 + 1, i2 - 1, i3, em::ex2) +
                            buffer(i1, i2, i3 + 1, em::ex2) +
                            buffer(i1 + 1, i2, i3 + 1, em::ex2) +
                            buffer(i1, i2 - 1, i3 + 1, em::ex2) +
                            buffer(i1 + 1, i2 - 1, i3 + 1, em::ex2));
          e_U[2] = HALF * (buffer(i1, i2, i3, em::ex3) +
                           buffer(i1 + 1, i2, i3, em::ex3));

          b_U[0] = INV_4 * (buffer(i1, i2, i3, em::bx1) +
                            buffer(i1 + 1, i2, i3, em::bx1) +
                            buffer(i1, i2 - 1, i3, em::bx1) +
                            buffer(i1 + 1, i2 - 1, i3, em::bx1));
          b_U[1] = buffer(i1, i2, i3, em::bx2);
          b_U[2] = INV_4 * (buffer(i1, i2, i3, em::bx3) +
                            buffer(i1, i2 - 1, i3, em::bx3) +
                            buffer(i1, i2, i3 + 1, em::bx3) +
                            buffer(i1, i2 - 1, i3 + 1, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_, i3_ + HALF },
                                                    e_U,
                                                    e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_, i3_ + HALF },
                                                    b_U,
                                                    b_T);
          const auto bx2_setter = fieldsetter.bx2(x_Ph, e_T, b_T);

          if (bx2_setter.first) {
            fields(i1, i2, i3, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ + HALF },
              bx2_setter.second);
          }
        }
        if constexpr (HasBx3) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF, i3_ },
                                                    x_Ph);

          e_U[0] = HALF * (buffer(i1, i2, i3, em::ex1) +
                           buffer(i1, i2 + 1, i3, em::ex1));
          e_U[1] = HALF * (buffer(i1, i2, i3, em::ex2) +
                           buffer(i1 + 1, i2, i3, em::ex2));
          e_U[2] = INV_8 * (buffer(i1, i2, i3, em::ex3) +
                            buffer(i1 + 1, i2, i3, em::ex3) +
                            buffer(i1, i2 + 1, i3, em::ex3) +
                            buffer(i1 + 1, i2 + 1, i3, em::ex3) +
                            buffer(i1, i2, i3 - 1, em::ex3) +
                            buffer(i1 + 1, i2, i3 - 1, em::ex3) +
                            buffer(i1, i2 + 1, i3 - 1, em::ex3) +
                            buffer(i1 + 1, i2 + 1, i3 - 1, em::ex3));

          b_U[0] = INV_4 * (buffer(i1, i2, i3, em::bx1) +
                            buffer(i1 + 1, i2, i3, em::bx1) +
                            buffer(i1, i2, i3 - 1, em::bx1) +
                            buffer(i1 + 1, i2, i3 - 1, em::bx1));
          b_U[1] = INV_4 * (buffer(i1, i2, i3, em::bx2) +
                            buffer(i1, i2 + 1, i3, em::bx2) +
                            buffer(i1, i2, i3 - 1, em::bx2) +
                            buffer(i1, i2 + 1, i3 - 1, em::bx2));
          b_U[2] = buffer(i1, i2, i3, em::bx3);

          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_ + HALF, i3_ },
                                                    e_U,
                                                    e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF, i2_ + HALF, i3_ },
                                                    b_U,
                                                    b_T);
          const auto bx3_setter = fieldsetter.bx3(x_Ph, e_T, b_T);

          if (bx3_setter.first) {
            fields(i1, i2, i3, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF, i3_ },
              bx3_setter.second);
          }
        }
      } else {
        raise::KernelError(HERE, "CustomEMFields_kernel 3D called for 1D/2D");
      }
    }
  };

} // namespace arch

#endif // ARCHETYPES_FIELD_SETTER_HPP
