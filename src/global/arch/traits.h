/**
 * @file arch/traits.h
 * @brief Defines a set of traits to check if a class satisfies certain conditions
 * @implements
 *   - traits::has_method<>
 *   - traits::ex1_t, ::ex2_t, ::ex3_t
 *   - traits::bx1_t, ::bx2_t, ::bx3_t
 *   - traits::dx1_t, ::dx2_t, ::dx3_t
 *   - traits::run_t
 *   - traits::check_compatibility<>
 *   - traits::compatibility<>
 * @namespaces:
 *   - traits::
 * @note realized with SFINAE technique
 */

#ifndef GLOBAL_ARCH_TRAITS_H
#define GLOBAL_ARCH_TRAITS_H

#include <type_traits>
#include <utility>

namespace traits {

  // special ::ex1, ::ex2, ::ex3, ::bx1, ::bx2, ::bx3, ::dx1, ::dx2, ::dx3
  template <template <typename> class Trait, typename T, typename = void>
  struct has_method : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_method<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  template <typename T>
  using ex1_t = decltype(&T::ex1);

  template <typename T>
  using ex2_t = decltype(&T::ex2);

  template <typename T>
  using ex3_t = decltype(&T::ex3);

  template <typename T>
  using bx1_t = decltype(&T::bx1);

  template <typename T>
  using bx2_t = decltype(&T::bx2);

  template <typename T>
  using bx3_t = decltype(&T::bx3);

  template <typename T>
  using dx1_t = decltype(&T::dx1);

  template <typename T>
  using dx2_t = decltype(&T::dx2);

  template <typename T>
  using dx3_t = decltype(&T::dx3);

  template <typename T>
  using run_t = decltype(&T::run);

  template <int N>
  struct check_compatibility {
    template <int... Is>
    static constexpr bool value(std::integer_sequence<int, Is...>) {
      return ((Is == N) || ...);
    }
  };

  template <int... Is>
  struct compatible_with {
    static constexpr auto value = std::integer_sequence<int, Is...> {};
  };

} // namespace traits

#endif // GLOBAL_ARCH_TRAITS_H