/**
 * @file traits/policies.h
 * @brief Concepts and traits for pusher policies
 * @implements
 *   - emission::NoPolicy_t - A placeholder policy that does nothing, used when no emission is desired
 *   - emission::IsNoPolicy - Concept to check if a given policy is NoPolicy_t
 *   - emission::HasPayload - Checks if the emission policy defines a Payload type
 *   - emission::HasNumbersInjected - Checks if the pusher policy has a
 *     - numbers_injected() method that returns a vector of npart_t
 *   - emission::HasEmittedSpeciesIndices - Checks if the pusher policy has an
 *     - emitted_species_indices() method that returns a vector of spidx_t
 *   - emission::HasShouldEmit - Checks if the pusher policy has a shouldEmit()
 *     - method with the correct signature that returns a Kokkos::pair<bool, bool>
 *   - emission::HasEmit - Checks if the pusher policy has an emit() method with
 *     - the correct signature that returns void
 *   - EmissionPolicyClass - Checks if a class satisfies all the requirements to be an emission policy
 * @namespaces:
 *   - traits::
 */

#ifndef TRAITS_POLICIES_H
#define TRAITS_POLICIES_H

#include "global.h"

#include "traits/archetypes.h"

#include <Kokkos_Pair.hpp>

#include <vector>

namespace traits::emission {

  struct NoPolicy_t {};

  template <class E>
  concept IsNoPolicy = std::is_same<std::remove_cvref_t<E>, NoPolicy_t>::value;

  template <class E>
  concept HasPayload = requires { typename E::Payload; };

  template <class E>
  concept HasNumbersInjected = requires(E& emission_policy) {
    {
      emission_policy.numbers_injected()
    } -> std::convertible_to<std::vector<npart_t>>;
  };

  template <class E>
  concept HasEmittedSpeciesIndices = requires(const E& emission_policy) {
    {
      emission_policy.emitted_species_indices()
    } -> std::convertible_to<std::vector<spidx_t>>;
  };

  template <class E, class M>
  concept HasShouldEmit = requires(const E&                   emission_policy,
                                   const coord_t<M::PrtlDim>& x_Cd,
                                   const coord_t<M::PrtlDim>& x_Ph,
                                   const vec_t<Dim::_3D>&     u_Ph,
                                   const vec_t<Dim::_3D>&     ep_Ph,
                                   const vec_t<Dim::_3D>&     bp_Ph,
                                   vec_t<Dim::_3D>&           delta_u_Ph,
                                   typename E::Payload&       payload) {
    {
      emission_policy.shouldEmit(x_Cd, x_Ph, u_Ph, ep_Ph, bp_Ph, delta_u_Ph, payload)
    } -> std::convertible_to<Kokkos::pair<bool, bool>>;
  };

  template <class E, class M>
  concept HasEmit = requires(const E&                         emission_policy,
                             const tuple_t<int, M::Dim>&      xi_Cd,
                             const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                             const vec_t<Dim::_3D>&           direction,
                             real_t                           weight,
                             real_t                           phi,
                             const typename E::Payload&       payload) {
    {
      emission_policy.emit(xi_Cd, dxi_Cd, direction, weight, phi, payload)
    } -> std::same_as<void>;
  };

} // namespace traits::emission

template <class E, class M>
concept EmissionPolicyClass = (::traits::emission::HasPayload<E> and
                               ::traits::emission::HasNumbersInjected<E> and
                               ::traits::emission::HasEmittedSpeciesIndices<E> and
                               ::traits::emission::HasShouldEmit<E, M> and
                               ::traits::emission::HasEmit<E, M>) or
                              ::traits::emission::IsNoPolicy<E>;

namespace traits::extfields {

  struct NoPolicy_t {};

  template <class F>
  concept IsNoPolicy = std::is_same<std::remove_cvref_t<F>, NoPolicy_t>::value;

} // namespace traits::extfields

template <class F, Dimension D>
concept ExtFieldsPolicyClass =
  (::traits::fieldsetter::HasFx1<F, D> or ::traits::fieldsetter::HasFx2<F, D> or
   ::traits::fieldsetter::HasFx3<F, D> or ::traits::fieldsetter::HasEx1<F, D> or
   ::traits::fieldsetter::HasEx2<F, D> or ::traits::fieldsetter::HasEx3<F, D> or
   ::traits::fieldsetter::HasBx1<F, D> or ::traits::fieldsetter::HasBx2<F, D> or
   ::traits::fieldsetter::HasBx3<F, D>) or
  ::traits::extfields::IsNoPolicy<F>;

#endif // TRAITS_POLICIES_H