/**
 * @file traits/policies.h
 * @brief Concepts and traits for pusher policies (emission, external fields, custom particle update)
 * @implements
 *   - traits::emission::NoPolicy_t
 *   - traits::emission::IsNoPolicy<>
 *   - traits::emission::HasPayload<>
 *   - traits::emission::HasNumbersInjected<>
 *   - traits::emission::HasEmittedSpeciesIndices<>
 *   - traits::emission::HasShouldEmit<>
 *   - traits::emission::HasEmit<>
 *   - EmissionPolicyClass<>
 *   - traits::extfields::NoPolicy_t
 *   - traits::extfields::IsNoPolicy<>
 *   - ExtFieldsPolicyClass<>
 *   - traits::custom_prtl_update::NoPolicy_t
 *   - traits::custom_prtl_update::IsNoPolicy<>
 *   - CustomParticleUpdatePolicyClass<>
 * @namespaces:
 *   - traits::emission::
 *   - traits::extfields::
 *   - traits::custom_prtl_update::
 */

#ifndef TRAITS_POLICIES_H
#define TRAITS_POLICIES_H

#include "global.h"

#include "traits/archetypes.h"

#include "framework/containers/particles.h"
#include "kernels/pushers/context.h"

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

namespace traits::custom_prtl_update {

  struct NoPolicy_t {};

  template <class CPU>
  concept IsNoPolicy = std::is_same<std::remove_cvref_t<CPU>, NoPolicy_t>::value;

} // namespace traits::custom_prtl_update

template <class CPU, class M>
concept CustomParticleUpdatePolicyClass =
  requires(const CPU&                                  cpu,
           prtlidx_t                                   p,
           const kernel::sr::PusherContext&            pusher_ctx,
           const kernel::sr::PusherBoundaries<M::Dim>& pusher_boundaries,
           const ntt::ParticleArrays&                  particles,
           const M&                                    metric) {
    {
      cpu(p, pusher_ctx, pusher_boundaries, particles, metric)
    } -> std::same_as<void>;
  } or
  traits::custom_prtl_update::IsNoPolicy<CPU>;

namespace traits::twobodyinteractions {

  template <class I>
  concept IsValid = requires(const I& interaction_policy,
                             spidx_t  sp1,
                             npart_t  p1,
                             spidx_t  sp2,
                             npart_t  p2,
                             real_t   tile_vol) {
    { interaction_policy(sp1, p1, sp2, p2, tile_vol) } -> std::same_as<void>;
  };

} // namespace traits::twobodyinteractions

template <class I>
concept TwoBodyInteractionPolicyClass = traits::twobodyinteractions::IsValid<I>;

#endif // TRAITS_POLICIES_H
