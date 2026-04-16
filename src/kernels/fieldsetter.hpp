/**
 * @file kernels/fieldsetter.hpp
 * @brief Kernel for updating the EM fields with a custom class
 * @implements
 *   - kernel::CustomFieldsetter<>
 * @namespaces:
 *   - kernel::
 */
#ifndef KERNELS_FIELDSETTER_HPP
#define KERNELS_FIELDSETTER_HPP

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

namespace kernel {
  using namespace ntt;

  template <SimEngine S, class M, class F>
    requires metric::traits::HasD<M> && metric::traits::HasTransform<M> &&
             metric::traits::HasTransform_i<M> && (S != SimEngine::GRPIC) &&
             (::traits::fieldsetter::HasConditionalEx1<F, M::Dim> ||
              ::traits::fieldsetter::HasConditionalEx2<F, M::Dim> ||
              ::traits::fieldsetter::HasConditionalEx3<F, M::Dim> ||
              ::traits::fieldsetter::HasConditionalBx1<F, M::Dim> ||
              ::traits::fieldsetter::HasConditionalBx2<F, M::Dim> ||
              ::traits::fieldsetter::HasConditionalBx3<F, M::Dim>)
  struct CustomFieldsetter {
    M                    metric;
    ndfield_t<M::Dim, 6> fields;
    ndfield_t<M::Dim, 6> buffer;

    const F fieldsetter;

    CustomFieldsetter(const M&                    metric,
                      ndfield_t<M::Dim, 6>&       fields,
                      const ndfield_t<M::Dim, 6>& buffer,
                      const F&                    fieldsetter)
      : metric { metric }
      , fields { fields }
      , fieldsetter { fieldsetter } {}

    Inline void operator()(index_t i1) const {
      if constexpr (M::Dim == Dim::_1D) {
        const auto      i1_ = COORD(i1);
        coord_t<M::Dim> x_Ph { ZERO };

        vec_t<Dim::_3D> e_U { ZERO };
        vec_t<Dim::_3D> b_U { ZERO };

        vec_t<Dim::_3D> e_T { ZERO };
        vec_t<Dim::_3D> b_T { ZERO };

        if constexpr (::traits::fieldsetter::HasConditionalEx1<F, M::Dim> or
                      ::traits::fieldsetter::HasConditionalBx2<F, M::Dim> or
                      ::traits::fieldsetter::HasConditionalBx3<F, M::Dim>) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Ph);

          e_U[0] = buffer(i1, em::ex1);
          e_U[1] = HALF * (buffer(i1, em::ex2) + buffer(i1 + 1, em::ex2));
          e_U[2] = HALF * (buffer(i1, em::ex3) + buffer(i1 + 1, em::ex3));

          b_U[0] = HALF * (buffer(i1, em::bx1) + buffer(i1 + 1, em::bx1));
          b_U[1] = buffer(i1, em::bx2);
          b_U[2] = buffer(i1, em::bx3);

          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF }, e_U, e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ + HALF }, b_U, b_T);

          if constexpr (::traits::fieldsetter::HasConditionalEx1<F, M::Dim>) {
            const auto ex1_setter = fieldsetter.ex1(x_Ph, e_T, b_T);

            if (ex1_setter.first) {
              fields(i1, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
                { i1_ + HALF },
                ex1_setter.second);
            }
          }
          if constexpr (::traits::fieldsetter::HasConditionalBx2<F, M::Dim>) {
            const auto bx2_setter = fieldsetter.bx2(x_Ph, e_T, b_T);

            if (bx2_setter.first) {
              fields(i1, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
                { i1_ + HALF },
                bx2_setter.second);
            }
          }
          if constexpr (::traits::fieldsetter::HasConditionalBx3<F, M::Dim>) {
            const auto bx3_setter = fieldsetter.bx3(x_Ph, e_T, b_T);

            if (bx3_setter.first) {
              fields(i1, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
                { i1_ + HALF },
                bx3_setter.second);
            }
          }
        }
        if constexpr (::traits::fieldsetter::HasConditionalEx2<F, M::Dim> or
                      ::traits::fieldsetter::HasConditionalEx3<F, M::Dim> or
                      ::traits::fieldsetter::HasConditionalBx1<F, M::Dim>) {
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Ph);

          e_U[0] = HALF * (buffer(i1, em::ex1) + buffer(i1 - 1, em::ex1));
          e_U[1] = buffer(i1, em::ex2);
          e_U[2] = buffer(i1, em::ex3);

          b_U[0] = buffer(i1, em::bx1);
          b_U[1] = HALF * (buffer(i1, em::bx2) + buffer(i1 - 1, em::bx2));
          b_U[2] = HALF * (buffer(i1, em::bx3) + buffer(i1 - 1, em::bx3));

          metric.template transform<Idx::U, Idx::T>({ i1_ }, e_U, e_T);
          metric.template transform<Idx::U, Idx::T>({ i1_ }, b_U, b_T);

          if constexpr (::traits::fieldsetter::HasConditionalEx2<F, M::Dim>) {
            const auto ex2_setter = fieldsetter.ex2(x_Ph, e_T, b_T);

            if (ex2_setter.first) {
              fields(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
                { i1_ },
                ex2_setter.second);
            }
          }
          if constexpr (::traits::fieldsetter::HasConditionalEx3<F, M::Dim>) {
            const auto ex3_setter = fieldsetter.ex3(x_Ph, e_T, b_T);

            if (ex3_setter.first) {
              fields(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
                { i1_ },
                ex3_setter.second);
            }
          }
          if constexpr (::traits::fieldsetter::HasConditionalBx1<F, M::Dim>) {
            const auto bx1_setter = fieldsetter.bx1(x_Ph, e_T, b_T);

            if (bx1_setter.first) {
              fields(i1, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
                { i1_ },
                bx1_setter.second);
            }
          }
        }
      } else {
        raise::KernelError(HERE, "CustomFieldsetter_kernel 1D called for 2D/3D");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (M::Dim == Dim::_2D) {
        const auto      i1_ = COORD(i1);
        const auto      i2_ = COORD(i2);
        coord_t<M::Dim> x_Ph { ZERO };

        vec_t<Dim::_3D> e_U { ZERO };
        vec_t<Dim::_3D> b_U { ZERO };

        vec_t<Dim::_3D> e_T { ZERO };
        vec_t<Dim::_3D> b_T { ZERO };

        if constexpr (::traits::fieldsetter::HasConditionalEx1<F, M::Dim> or
                      ::traits::fieldsetter::HasConditionalBx2<F, M::Dim>) {
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
          if constexpr (::traits::fieldsetter::HasConditionalEx1<F, M::Dim>) {
            const auto ex1_setter = fieldsetter.ex1(x_Ph, e_T, b_T);

            if (ex1_setter.first) {
              fields(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
                { i1_ + HALF, i2_ },
                ex1_setter.second);
            }
          }
          if constexpr (::traits::fieldsetter::HasConditionalBx2<F, M::Dim>) {
            const auto bx2_setter = fieldsetter.bx2(x_Ph, e_T, b_T);

            if (bx2_setter.first) {
              fields(i1, i2, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
                { i1_ + HALF, i2_ },
                bx2_setter.second);
            }
          }
        }
        if constexpr (::traits::fieldsetter::HasConditionalEx2<F, M::Dim> or
                      ::traits::fieldsetter::HasConditionalBx1<F, M::Dim>) {
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
          if constexpr (::traits::fieldsetter::HasConditionalEx2<F, M::Dim>) {
            const auto ex2_setter = fieldsetter.ex2(x_Ph, e_T, b_T);

            if (ex2_setter.first) {
              fields(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
                { i1_, i2_ + HALF },
                ex2_setter.second);
            }
          }
          if constexpr (::traits::fieldsetter::HasConditionalBx1<F, M::Dim>) {
            const auto bx1_setter = fieldsetter.bx1(x_Ph, e_T, b_T);

            if (bx1_setter.first) {
              fields(i1, i2, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
                { i1_, i2_ + HALF },
                bx1_setter.second);
            }
          }
        }
        if constexpr (::traits::fieldsetter::HasConditionalEx3<F, M::Dim>) {
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
        if constexpr (::traits::fieldsetter::HasConditionalBx3<F, M::Dim>) {
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
        raise::KernelError(HERE, "CustomFieldsetter_kernel 2D called for 1D/3D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (M::Dim == Dim::_3D) {
        const auto      i1_ = COORD(i1);
        const auto      i2_ = COORD(i2);
        const auto      i3_ = COORD(i3);
        coord_t<M::Dim> x_Ph { ZERO };

        vec_t<Dim::_3D> e_U { ZERO };
        vec_t<Dim::_3D> b_U { ZERO };

        vec_t<Dim::_3D> e_T { ZERO };
        vec_t<Dim::_3D> b_T { ZERO };
        if constexpr (::traits::fieldsetter::HasConditionalEx1<F, M::Dim>) {
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
        if constexpr (::traits::fieldsetter::HasConditionalEx2<F, M::Dim>) {
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
        if constexpr (::traits::fieldsetter::HasConditionalEx3<F, M::Dim>) {
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
        if constexpr (::traits::fieldsetter::HasConditionalBx1<F, M::Dim>) {
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
        if constexpr (::traits::fieldsetter::HasConditionalBx2<F, M::Dim>) {
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
        if constexpr (::traits::fieldsetter::HasConditionalBx3<F, M::Dim>) {
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
        raise::KernelError(HERE, "CustomFieldsetter_kernel 3D called for 1D/2D");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_FIELDSETTER_HPP
