/**
 * @file utils/comparators.h
 * @brief Architecture-agnostic functions for comparing real-valued numbers
 * @implements
 *   - cmp::AlmostEqual<> -> bool // float vs float, double vs double
 *   - cmp::AlmostZero<> -> bool  // float/double
 *   - cmp::AlmostEqual_host<> -> bool // float vs float, double vs double
 *   - cmp::AlmostZero_host<> -> bool  // float/double
 * @namespaces:
 *   - cmp::
 * @note
 * The _host definitions are here for backwards compatibility with
 * older nvcc versions (pre 12.1) where `Inline` call on the host raises a warning
 */

#ifndef GLOBAL_UTILS_COMPARATORS_H
#define GLOBAL_UTILS_COMPARATORS_H

#include "arch/kokkos_aliases.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

namespace cmp {

  template <class T>
  Inline auto AlmostEqual(T a, T b, T eps = Kokkos::Experimental::epsilon<T>::value)
    -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return (a == b) ||
           (math::abs(a - b) <= math::min(math::abs(a), math::abs(b)) * eps);
  }

  template <class T>
  Inline auto AlmostZero(T a, T eps = Kokkos::Experimental::epsilon<T>::value)
    -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return math::abs(a) <= eps;
  }

  template <class T>
  inline auto AlmostEqual_host(T a, T b, T eps = std::numeric_limits<T>::epsilon())
    -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return (a == b) ||
           (std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * eps);
  }

  template <class T>
  inline auto AlmostZero_host(T a, T eps = std::numeric_limits<T>::epsilon())
    -> bool {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return std::abs(a) <= eps;
  }

} // namespace cmp

#endif // GLOBAL_UTILS_COMPARATORS_H
