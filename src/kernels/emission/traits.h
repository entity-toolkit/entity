/**
 * @file kernels/emission/traits.h
 * @brief Concepts and traits for emission policies
 * @implements
 *   - kernel::traits::emission::HasPayload
 * @namespaces:
 *   - kernel::traits::emission::
 */

#ifndef KERNELS_EMISSION_TRAITS_H
#define KERNELS_EMISSION_TRAITS_H

#include "global.h"

#include <Kokkos_Pair.hpp>

#include <vector>

namespace kernel {
  namespace traits {
    namespace emission {
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
      concept HasShouldEmit = requires(const E& emission_policy,
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
      concept HasEmit = requires(const E&                    emission_policy,
                                 const tuple_t<int, M::Dim>& xi_Cd,
                                 const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                                 const vec_t<Dim::_3D>&           direction,
                                 real_t                           weight,
                                 real_t                           phi,
                                 const typename E::Payload&       payload) {
        {
          emission_policy.emit(xi_Cd, dxi_Cd, direction, weight, phi, payload)
        } -> std::same_as<void>;
      };

      template <class E, class M>
      concept IsValid = HasPayload<E> && HasNumbersInjected<E> &&
                        HasEmittedSpeciesIndices<E> && HasShouldEmit<E, M> &&
                        HasEmit<E, M>;

    } // namespace emission
  } // namespace traits
} // namespace kernel

#endif // KERNELS_EMISSION_TRAITS_H
