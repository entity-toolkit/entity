/**
 * @file arch/traits.h
 * @brief Defines a set of traits to check if a class satisfies certain conditions
 * @implements
 *   - traits::fieldsetter::HasEx1, ::HasEx2, ::HasEx3 - checks for E functions in field setter class
 *   - traits::fieldsetter::HasBx1, ::HasBx2, ::HasBx3 - checks for B functions in field setter class
 *   - traits::fieldsetter::HasDx1, ::HasDx2, ::HasDx3 - checks for D functions in field setter class
 *   - traits::external::HasFx1, ::HasFx2, ::HasFx3 - checks for F functions in external field class
 *   - traits::external::HasEx1, ::HasEx2, ::HasEx3 - checks for E functions in external field class
 *   - traits::external::HasBx1, ::HasBx2, ::HasBx3 - checks for B functions in external field class
 *   - traits::external::HasExternalE - checks if a class has any external E field method
 *   - traits::external::HasExternalB - checks if a class has any external B field method
 *   - traits::external::HasExternalF - checks if a class has any external force method
 *   - traits::has_method<>
 *   - traits::has_member<>
 *   - traits::run_t, traits::to_string_t
 *   - traits::check_compatibility<>
 *   - traits::compatibility<>
 *   - traits::is_pair<>
 * @namespaces:
 *   - traits::
 *   - traits::pgen::
 * @note realized with SFINAE technique
 */

#ifndef GLOBAL_ARCH_TRAITS_H
#define GLOBAL_ARCH_TRAITS_H

#include "global.h"

#include <Kokkos_Core.hpp>

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

namespace traits {

  namespace fieldsetter {
    // special ::ex1, ::ex2, ::ex3, ::bx1, ::bx2, ::bx3, ::dx1, ::dx2, ::dx3 methods
    template <class T, Dimension D>
    concept HasEx1 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.ex1(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasEx2 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.ex2(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasEx3 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.ex3(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx1 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.bx1(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx2 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.bx2(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx3 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.bx3(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasDx1 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.dx1(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasDx2 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.dx2(x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasDx3 = requires(const T& t, const coord_t<D>& x_Ph) {
      { t.dx3(x_Ph) } -> std::convertible_to<real_t>;
    };
  } // namespace fieldsetter

  namespace external {
    template <class T, Dimension D>
    concept HasFx1 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.fx1(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasFx2 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.fx2(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasFx3 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.fx3(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasEx1 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.ex1(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasEx2 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.ex2(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasEx3 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.ex3(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx1 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.bx1(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx2 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.bx2(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasBx3 = requires(const T&          t,
                              spidx_t           sp,
                              simtime_t         time,
                              const coord_t<D>& x_Ph) {
      { t.bx3(sp, time, x_Ph) } -> std::convertible_to<real_t>;
    };

    template <class T, Dimension D>
    concept HasExternalF = (HasFx1<T, D> or HasFx2<T, D> or HasFx3<T, D>);

    template <class T, Dimension D>
    concept HasExternalE = (HasEx1<T, D> or HasEx2<T, D> or HasEx3<T, D>);

    template <class T, Dimension D>
    concept HasExternalB = (HasBx1<T, D> or HasBx2<T, D> or HasBx3<T, D>);

  } // namespace external

  namespace emission {

    template <class E>
    concept HasPayloadType = requires { typename E::Payload; };

    template <class E, Dimension D>
    concept HasShouldEmit = HasPayloadType<E> and requires(
                                                    const E&               e,
                                                    const coord_t<D>&      x_Cd,
                                                    const coord_t<D>&      x_Ph,
                                                    const vec_t<Dim::_3D>& u_Ph,
                                                    const vec_t<Dim::_3D>& ep,
                                                    const vec_t<Dim::_3D>& bp,
                                                    vec_t<Dim::_3D>& delta_u_Ph,
                                                    typename E::Payload& payload) {
      {
        e.shouldEmit(x_Cd, x_Ph, u_Ph, ep, bp, delta_u_Ph, payload)
      } -> std::convertible_to<Kokkos::pair<bool, bool>>;
    };

    template <class E, Dimension D>
    concept HasEmit = HasPayloadType<E> and
                      requires(const E&                    e,
                               const tuple_t<int, D>&      xi_Cd,
                               const tuple_t<prtldx_t, D>& dxi_Cd,
                               const vec_t<Dim::_3D>&      direction,
                               real_t                      weight,
                               real_t                      phi,
                               const typename E::Payload&  payload) {
                        {
                          e.emit(xi_Cd, dxi_Cd, direction, weight, phi, payload)
                        } -> std::same_as<void>;
                      };

    template <class E, Dimension D>
    concept IsValidEmissionPolicy = HasShouldEmit<E, D> and HasEmit<E, D>;

  } // namespace emission

  template <template <typename> class Trait, typename T, typename = void>
  struct has_method : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_method<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  // trivial overload of `has_method` for readability
  template <template <typename> class Trait, typename T, typename = void>
  struct has_member : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_member<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  // for pgen ext_fields
  template <typename T>
  using species_t = decltype(&T::species);

  template <typename>
  struct always_false : std::false_type {};

  // generic
  template <typename T>
  struct is_pair : std::false_type {};

  template <typename T, typename U>
  struct is_pair<std::pair<T, U>> : std::true_type {};

  // c++20
  namespace params {
    template <class P>
    concept HasToString = requires(const P& params) {
      { params.to_string() } -> std::convertible_to<std::string>;
    };
  } // namespace params

} // namespace traits

#endif // GLOBAL_ARCH_TRAITS_H
