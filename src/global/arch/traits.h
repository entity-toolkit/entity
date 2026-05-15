/**
 * @file arch/traits.h
 * @brief Defines a set of traits to check if a class satisfies certain conditions
 * @implements
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
