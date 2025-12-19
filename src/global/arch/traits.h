/**
 * @file arch/traits.h
 * @brief Defines a set of traits to check if a class satisfies certain conditions
 * @implements
 *   - traits::has_method<>
 *   - traits::has_member<>
 *   - traits::ex1_t, ::ex2_t, ::ex3_t
 *   - traits::bx1_t, ::bx2_t, ::bx3_t
 *   - traits::dx1_t, ::dx2_t, ::dx3_t
 *   - traits::run_t, traits::to_string_t
 *   - traits::pgen::init_flds_t
 *   - traits::pgen::ext_force_t
 *   - traits::pgen::ext_current_t
 *   - traits::pgen::atm_fields_t
 *   - traits::pgen::match_fields_const_t
 *   - traits::pgen::match_fields_t
 *   - traits::pgen::fix_fields_const_t
 *   - traits::pgen::fix_fields_t
 *   - traits::pgen::init_prtls_t
 *   - traits::pgen::custom_fields_t
 *   - traits::pgen::custom_field_output_t
 *   - traits::pgen::custom_poststep_t
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

#include <type_traits>
#include <utility>

namespace traits {

  template <template <typename> class Trait, typename T, typename = void>
  struct has_method : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_method<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  // trivial overload of `has_method` for readability
  template <template <typename> class Trait, typename T, typename = void>
  struct has_member : std::false_type {};

  template <template <typename> class Trait, typename T>
  struct has_member<Trait, T, std::void_t<Trait<T>>> : std::true_type {};

  // for fieldsetter
  // special ::ex1, ::ex2, ::ex3, ::bx1, ::bx2, ::bx3, ::dx1, ::dx2, ::dx3
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

  // for simengine
  template <typename T>
  using run_t = decltype(&T::run);

  // for params
  template <typename T>
  using to_string_t = decltype(&T::to_string);

  // for pgen
  namespace pgen {
    template <typename T>
    using init_flds_t = decltype(&T::init_flds);

    template <typename T>
    using init_prtls_t = decltype(&T::InitPrtls);

    template <typename T>
    using ext_force_t = decltype(&T::ext_force);

    template <typename T>
    using ext_current_t = decltype(&T::ext_current);

    template <typename T>
    using atm_fields_t = decltype(&T::AtmFields);

    template <typename T>
    using match_fields_t = decltype(&T::MatchFields);

    template <typename T>
    using match_fields_in_x1_t = decltype(&T::MatchFieldsInX1);

    template <typename T>
    using match_fields_in_x2_t = decltype(&T::MatchFieldsInX2);

    template <typename T>
    using match_fields_in_x3_t = decltype(&T::MatchFieldsInX3);

    template <typename T>
    using match_fields_const_t = decltype(&T::MatchFieldsConst);

    template <typename T>
    using fix_fields_t = decltype(&T::FixFields);

    template <typename T>
    using fix_fields_const_t = decltype(&T::FixFieldsConst);

    template <typename T>
    using perfect_conductor_fields_t = decltype(&T::PerfectConductorFields);

    template <typename T>
    using perfect_conductor_fields_const_t = decltype(&T::PerfectConductorFieldsConst);

    template <typename T>
    using perfect_conductor_currents_t = decltype(&T::PerfectConductorCurrents);

    template <typename T>
    using perfect_conductor_currents_const_t = decltype(&T::PerfectConductorCurrentsConst);

    template <typename T>
    using custom_fields_t = decltype(&T::CustomFields);

    template <typename T>
    using custom_poststep_t = decltype(&T::CustomPostStep);

    template <typename T>
    using custom_field_output_t = decltype(&T::CustomFieldOutput);

    template <typename T>
    using custom_stat_t = decltype(&T::CustomStat);
  } // namespace pgen

  // for pgen extforce
  template <typename T>
  using species_t = decltype(&T::species);

  // checking compat for the problem generator + engine
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

  template <typename>
  struct always_false : std::false_type {};

  // generic

  template <typename T>
  struct is_pair : std::false_type {};

  template <typename T, typename U>
  struct is_pair<std::pair<T, U>> : std::true_type {};

  // c++20
  namespace metric {

    template <class M>
    concept HasD = requires {
      { M::Dim } -> std::convertible_to<Dimension>;
    };

    template <class M>
    concept HasH_ij = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.template h_<1, 1>(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasHij = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.template h<1, 1>(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasSqrtDetH = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.sqrt_det_h(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasSqrtDetHTilde = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.sqrt_det_h_tilde(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasPolarArea = requires(const M& m, real_t xi_2) {
      { m.polar_area(xi_2) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasTransform_i = requires(const M&               m,
                                      const coord_t<M::Dim>& xi,
                                      real_t                 v_in) {
      {
        m.template transform<1, Idx::U, Idx::D>(xi, v_in)
      } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasTransform = requires(const M&               m,
                                    const coord_t<M::Dim>& xi,
                                    const vec_t<Dim::_3D>& v_in,
                                    vec_t<Dim::_3D>&       v_out) {
      {
        m.template transform<Idx::U, Idx::D>(xi, v_in, v_out)
      } -> std::same_as<void>;
    };

    template <class M>
    concept HasTransformXYZ = requires(const M&                   m,
                                       const coord_t<M::PrtlDim>& xi,
                                       const vec_t<Dim::_3D>&     v_in,
                                       vec_t<Dim::_3D>&           v_out) {
      {
        m.template transform_xyz<Idx::XYZ, Idx::D>(xi, v_in, v_out)
      } -> std::same_as<void>;
    };

    template <class M>
    concept HasConvert_i = requires(const M& m, real_t x) {
      {
        m.template convert<1, Crd::Cd, Crd::Ph>(x)
      } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasConvert = requires(const M&               m,
                                  const coord_t<M::Dim>& x_in,
                                  coord_t<M::Dim>&       x_out) {
      {
        m.template convert<Crd::Cd, Crd::Ph>(x_in, x_out)
      } -> std::same_as<void>;
    };

    template <class M>
    concept HasConvertXYZ = requires(const M&                   m,
                                     const coord_t<M::PrtlDim>& x_in,
                                     coord_t<M::PrtlDim>&       x_out) {
      {
        m.template convert_xyz<Crd::Cd, Crd::XYZ>(x_in, x_out)
      } -> std::same_as<void>;
    };

    template <class M>
    concept HasAlpha = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.alpha(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasBeta1 = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.beta1(xi) } -> std::convertible_to<real_t>;
    };

    template <class M>
    concept HasMetricDerivatives = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.dr_alpha(xi) } -> std::convertible_to<real_t>;
      { m.dr_beta1(xi) } -> std::convertible_to<real_t>;
      { m.dr_h11(xi) } -> std::convertible_to<real_t>;
      { m.dr_h22(xi) } -> std::convertible_to<real_t>;
      { m.dr_h33(xi) } -> std::convertible_to<real_t>;
      { m.dr_h13(xi) } -> std::convertible_to<real_t>;
      { m.dt_alpha(xi) } -> std::convertible_to<real_t>;
      { m.dt_beta1(xi) } -> std::convertible_to<real_t>;
      { m.dt_h11(xi) } -> std::convertible_to<real_t>;
      { m.dt_h22(xi) } -> std::convertible_to<real_t>;
      { m.dt_h33(xi) } -> std::convertible_to<real_t>;
      { m.dt_h13(xi) } -> std::convertible_to<real_t>;
    };

  } // namespace metric

  namespace energydist {

    template <class ED>
    concept IsValid = requires(const ED&             edist,
                               const coord_t<ED::D>& x_Ph,
                               vec_t<Dim::_3D>&      v) {
      { edist(x_Ph, v) } -> std::same_as<void>;
    };

  } // namespace energydist

  namespace spatialdist {

    template <class SD>
    concept IsValid = requires(const SD& sdist, const coord_t<SD::D>& x_Ph) {
      { sdist(x_Ph) } -> std::convertible_to<real_t>;
    };

  } // namespace spatialdist

} // namespace traits

#endif // GLOBAL_ARCH_TRAITS_H
