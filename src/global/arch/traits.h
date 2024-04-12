/**
 * @file arch/traits.h
 * @brief Defines a set of traits to check if a class satisfies certain conditions
 * @implements
 *   - traits::has_ex1
 *   - traits::has_ex2
 *   - traits::has_ex3
 *   - traits::has_bx1
 *   - traits::has_bx2
 *   - traits::has_bx3
 *   - traits::has_dx1
 *   - traits::has_dx2
 *   - traits::has_dx3
 *   - traits::has_x_Code2Phys
 * @namespaces:
 *   - traits::
 * @note realized with SFINAE technique
 */

#include <type_traits>

namespace traits {

  // special ::ex1, ::ex2, ::ex3, ::bx1, ::bx2, ::bx3, ::dx1, ::dx2, ::dx3
  template <typename T, typename = void>
  struct has_ex1 : std::false_type {};

  template <typename T>
  struct has_ex1<T, std::void_t<decltype(&T::ex1)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_ex2 : std::false_type {};

  template <typename T>
  struct has_ex2<T, std::void_t<decltype(&T::ex2)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_ex3 : std::false_type {};

  template <typename T>
  struct has_ex3<T, std::void_t<decltype(&T::ex3)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_bx1 : std::false_type {};

  template <typename T>
  struct has_bx1<T, std::void_t<decltype(&T::bx1)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_bx2 : std::false_type {};

  template <typename T>
  struct has_bx2<T, std::void_t<decltype(&T::bx2)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_bx3 : std::false_type {};

  template <typename T>
  struct has_bx3<T, std::void_t<decltype(&T::bx3)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_dx1 : std::false_type {};

  template <typename T>
  struct has_dx1<T, std::void_t<decltype(&T::dx1)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_dx2 : std::false_type {};

  template <typename T>
  struct has_dx2<T, std::void_t<decltype(&T::dx2)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_dx3 : std::false_type {};

  template <typename T>
  struct has_dx3<T, std::void_t<decltype(&T::dx3)>> : std::true_type {};

} // namespace traits