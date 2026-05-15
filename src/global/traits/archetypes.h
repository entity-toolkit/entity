/**
 * @file traits/archetypes.h
 * @brief Defines a set of traits for commonly used class archetypes
 * @implements
 *   - EnrgDistClass<> - checks if a class can be used as an energy distribution
 *   - SpatialDistClass<> - checks if a class can be used as a spatial distribution
 *   - traits::fieldsetter::HasFx1, ::HasFx2, ::HasFx3 - checks for F functions in field setter class
 *   - traits::fieldsetter::HasEx1, ::HasEx2, ::HasEx3 - checks for E functions in field setter class
 *   - traits::fieldsetter::HasBx1, ::HasBx2, ::HasBx3 - checks for B functions in field setter class
 *   - traits::fieldsetter::HasDx1, ::HasDx2, ::HasDx3 - checks for D functions in field setter class
 *   - traits::fieldsetter::HasConditionalEx1, ::HasConditionalEx2, ::HasConditionalEx3
 *     - checks for conditional E functions in field setter class
 *   - traits::fieldsetter::HasConditionalBx1, ::HasConditionalBx2, ::HasConditionalBx3
 *     - checks for conditional B functions in field setter class
 *   - traits::fieldsetter::HasConditionalDx1, ::HasConditionalDx2, ::HasConditionalDx3
 *     - checks for conditional D functions in field setter class
 *   - FieldSetterClass<>
 *   - SRFieldSetterClass<>
 *   - ConditionalSRFieldSetterClass<>
 */
#ifndef GLOBAL_TRAITS_ARCHETYPES_H
#define GLOBAL_TRAITS_ARCHETYPES_H

#include "enums.h"
#include "global.h"

#include <Kokkos_Pair.hpp>

template <class ED, Dimension D>
concept EnrgDistClass = requires(const ED&         edist,
                                 const coord_t<D>& x_Ph,
                                 vec_t<Dim::_3D>&  v) {
  { edist(x_Ph, v) } -> std::same_as<void>;
};

template <class SD, Dimension D>
concept SpatialDistClass = requires(const SD& sdist, const coord_t<D>& x_Ph) {
  { sdist(x_Ph) } -> std::convertible_to<real_t>;
};

namespace traits::fieldsetter {
  template <class T, Dimension D>
  concept HasFx1 = requires(const T& t, const coord_t<D>& x_Ph) {
    { t.fx1(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class T, Dimension D>
  concept HasFx2 = requires(const T& t, const coord_t<D>& x_Ph) {
    { t.fx2(x_Ph) } -> std::convertible_to<real_t>;
  };

  template <class T, Dimension D>
  concept HasFx3 = requires(const T& t, const coord_t<D>& x_Ph) {
    { t.fx3(x_Ph) } -> std::convertible_to<real_t>;
  };

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

  template <class T, Dimension D>
  concept HasConditionalEx1 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& e_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.ex1(x_Ph, e_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };

  template <class T, Dimension D>
  concept HasConditionalEx2 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& e_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.ex2(x_Ph, e_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };

  template <class T, Dimension D>
  concept HasConditionalEx3 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& e_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.ex3(x_Ph, e_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };

  template <class T, Dimension D>
  concept HasConditionalBx1 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& e_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.bx1(x_Ph, e_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };

  template <class T, Dimension D>
  concept HasConditionalBx2 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& e_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.bx2(x_Ph, e_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };

  template <class T, Dimension D>
  concept HasConditionalBx3 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& e_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.bx3(x_Ph, e_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };

  template <class T, Dimension D>
  concept HasConditionalDx1 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& d_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.dx1(x_Ph, d_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };

  template <class T, Dimension D>
  concept HasConditionalDx2 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& d_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.dx2(x_Ph, d_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };

  template <class T, Dimension D>
  concept HasConditionalDx3 = requires(const T&               t,
                                       const coord_t<D>&      x_Ph,
                                       const vec_t<Dim::_3D>& d_Ph,
                                       const vec_t<Dim::_3D>& b_Ph) {
    {
      t.dx3(x_Ph, d_Ph, b_Ph)
    } -> std::convertible_to<Kokkos::pair<bool, real_t>>;
  };
} // namespace traits::fieldsetter

template <class FS, ntt::SimEngine::type S, Dimension D>
concept FieldSetterClass =
  ((S == ntt::SimEngine::SRPIC) and
   (::traits::fieldsetter::HasEx1<FS, D> or ::traits::fieldsetter::HasEx2<FS, D> or
    ::traits::fieldsetter::HasEx3<FS, D> or ::traits::fieldsetter::HasBx1<FS, D> or
    ::traits::fieldsetter::HasBx2<FS, D> or ::traits::fieldsetter::HasBx3<FS, D>)) or
  ((S == ntt::SimEngine::GRPIC) and
   ((::traits::fieldsetter::HasDx1<FS, D> and ::traits::fieldsetter::HasDx2<FS, D> and
     ::traits::fieldsetter::HasDx3<FS, D>) or
    (::traits::fieldsetter::HasBx1<FS, D> and ::traits::fieldsetter::HasBx2<FS, D> and
     ::traits::fieldsetter::HasBx3<FS, D>)));

template <class FS, Dimension D>
concept SRFieldSetterClass = (::traits::fieldsetter::HasEx1<FS, D> or
                              ::traits::fieldsetter::HasEx2<FS, D> or
                              ::traits::fieldsetter::HasEx3<FS, D> or
                              ::traits::fieldsetter::HasBx1<FS, D> or
                              ::traits::fieldsetter::HasBx2<FS, D> or
                              ::traits::fieldsetter::HasBx3<FS, D>);

template <class FS, Dimension D>
concept ConditionalSRFieldSetterClass =
  ::traits::fieldsetter::HasConditionalEx1<FS, D> or
  ::traits::fieldsetter::HasConditionalEx2<FS, D> or
  ::traits::fieldsetter::HasConditionalEx3<FS, D> or
  ::traits::fieldsetter::HasConditionalBx1<FS, D> or
  ::traits::fieldsetter::HasConditionalBx2<FS, D> or
  ::traits::fieldsetter::HasConditionalBx3<FS, D>;

#endif // GLOBAL_TRAITS_ARCHETYPES_H