/**
 * @file utils/comparators.h
 * @brief Architecture-agnostic functions for comparing real-valued numbers
 * @implements
 *   - cmp::AlmostEqual<> -> bool // float vs float, double vs double
 *   - cmp::AlmostZero<> -> bool  // float/double
 * @depends:
 *   - arch/kokkos_aliases.h
 * @namespaces:
 *   - cmp::
 */

#ifndef GLOBAL_UTILS_COMPARATORS_H
#define GLOBAL_UTILS_COMPARATORS_H

#include "arch/kokkos_aliases.h"

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace cmp {

  template <class T>
  Inline auto AlmostEqual(T a, T b, T eps = Kokkos::Experimental::epsilon<T>::value)
    -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return (a == b) ||
           (math::abs(a - b) <= math::min(math::abs(a), math::abs(b)) * eps);
  }

  /**
   * @brief Function to compare a number with zero.
   * @tparam T Type of the number.
   * @param a Number
   * @param epsilon Accuracy [optional].
   * @returns true/false.
   */
  template <class T>
  Inline auto AlmostZero(T a, T eps = Kokkos::Experimental::epsilon<T>::value)
    -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return math::abs(a) <= eps;
  }

} // namespace cmp

#endif // GLOBAL_UTILS_COMPARATORS_H