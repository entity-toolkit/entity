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

#include "arch/kokkos_aliases.h"

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
  namespace engine {
    template <class E>
    concept HasRun = requires(E& engine) {
      { engine.run() } -> std::same_as<void>;
    };
  } // namespace engine

  namespace params {
    template <class P>
    concept HasToString = requires(const P& params) {
      { params.to_string() } -> std::convertible_to<std::string>;
    };
  } // namespace params

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
    concept HasSqrtH_ij = requires(const M& m, const coord_t<M::Dim>& xi) {
      { m.template sqrt_h_<1, 1>(xi) } -> std::convertible_to<real_t>;
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
    concept HasTotVolume = requires(const M& m) {
      { m.totVolume() } -> std::convertible_to<real_t>;
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

  namespace pgen {

    template <class PG>
    concept HasInitFlds = requires(const PG& pgen) { pgen.init_flds; };

    template <class PG, class D>
    concept HasInitPrtls = requires(const PG& pgen, D& domain) {
      pgen.InitPrtls(domain);
    };

    template <class PG>
    concept HasExtForce = requires(const PG& pgen) { pgen.ext_force; };

    template <class PG>
    concept HasExtCurrent = requires(const PG& pgen) { pgen.ext_current; };

    template <class PG>
    concept HasAtmFields = requires(const PG& pgen, simtime_t time) {
      pgen.AtmFields(time);
    };

    template <class PG>
    concept HasMatchFields = requires(const PG& pgen, simtime_t time) {
      pgen.MatchFields(time);
    };

    template <class PG>
    concept HasMatchFieldsInX1 = requires(const PG& pgen, simtime_t time) {
      pgen.MatchFieldsInX1(time);
    };

    template <class PG>
    concept HasMatchFieldsInX2 = requires(const PG& pgen, simtime_t time) {
      pgen.MatchFieldsInX2(time);
    };

    template <class PG>
    concept HasMatchFieldsInX3 = requires(const PG& pgen, simtime_t time) {
      pgen.MatchFieldsInX3(time);
    };

    template <class PG>
    concept HasFixFieldsConst = requires(const PG&      pgen,
                                         const bc_in&   bc,
                                         const ntt::em& comp) {
      pgen.FixFieldsConst(bc, comp);
    };

    template <class PG, class D>
    concept HasCustomPostStep = requires(const PG&  pgen,
                                         timestep_t s,
                                         simtime_t  t,
                                         D&         domain) {
      pgen.CustomPostStep(s, t, domain);
    };

    template <class PG, class D, Dimension Dim>
    concept HasCustomFieldOutput = requires(const PG&          pgen,
                                            const std::string& name,
                                            ndfield_t<Dim, 6>& buff,
                                            index_t            idx,
                                            timestep_t         step,
                                            simtime_t          time,
                                            const D&           dom) {
      pgen.CustomFieldOutput(name, buff, idx, step, time, dom);
    };

    template <class PG, class D>
    concept HasCustomStatOutput = requires(const PG&          pgen,
                                           const std::string& name,
                                           timestep_t         s,
                                           simtime_t          t,
                                           const D&           dom) {
      pgen.CustomStat(name, s, t, dom);
    };

  } // namespace pgen

} // namespace traits

#endif // GLOBAL_ARCH_TRAITS_H
