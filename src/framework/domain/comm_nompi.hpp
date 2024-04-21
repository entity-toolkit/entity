/**
 * @file framework/domain/comm_nompi.hpp
 * @brief Communication routines without mpi
 * @implements
 *   - comm::CommunicateField<> -> void
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/kokkos_aliases.h
 *   - arch/mpi_aliases.h
 *   - utils/error.h
 *   - framework/domain/domain.h
 * @namespaces:
 *   - comm::
 * @note This should only be included if the MPI_ENABLED flag is not set
 */

#ifndef FRAMEWORK_DOMAIN_COMM_NOMPI_HPP
#define FRAMEWORK_DOMAIN_COMM_NOMPI_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "framework/domain/domain.h"

#include <Kokkos_Core.hpp>

namespace comm {
  using namespace ntt;

  template <SimEngine::type S, class M, int N>
  inline void CommunicateField(unsigned int                      idx,
                               ndfield_t<M::Dim, N>&             fld,
                               const Domain<S, M>*               send_to,
                               const Domain<S, M>*               recv_from,
                               const std::vector<range_tuple_t>& send_slice,
                               const std::vector<range_tuple_t>& recv_slice,
                               const range_tuple_t&              comps,
                               bool                              additive) {
    constexpr auto D = M::Dim;
    raise::ErrorIf(send_to == nullptr && recv_from == nullptr,
                   "CommunicateField called with nullptrs",
                   HERE);

    //  trivial copy if sending to self and receiving from self
    if ((send_to->index() == idx) || (recv_from->index() == idx)) {
      raise::ErrorIf((recv_from->index() != idx) || (send_to->index() != idx),
                     "Cannot send to self and receive from another domain",
                     HERE);
      // sending/recv to/from self
      if (not additive) {
        if constexpr (D == Dim::_1D) {
          Kokkos::deep_copy(Kokkos::subview(fld, recv_slice[0], comps),
                            Kokkos::subview(fld, send_slice[0], comps));
        } else if constexpr (D == Dim::_2D) {
          Kokkos::deep_copy(
            Kokkos::subview(fld, recv_slice[0], recv_slice[1], comps),
            Kokkos::subview(fld, send_slice[0], send_slice[1], comps));
        } else if constexpr (D == Dim::_3D) {
          Kokkos::deep_copy(
            Kokkos::subview(fld, recv_slice[0], recv_slice[1], recv_slice[2], comps),
            Kokkos::subview(fld, send_slice[0], send_slice[1], send_slice[2], comps));
        }
      } else {
        if constexpr (D == Dim::_1D) {
          const auto offset_x1 = (long int)recv_slice[0].first -
                                 (long int)send_slice[0].first;
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
              { recv_slice[0].first, comps.first },
              { recv_slice[0].second, comps.second }),
            Lambda(index_t i1, index_t ci) {
              fld(i1, ci) += fld(i1 - offset_x1, ci);
            });
        } else if constexpr (D == Dim::_2D) {
          const auto offset_x1 = (long int)recv_slice[0].first -
                                 (long int)send_slice[0].first;
          const auto offset_x2 = (long int)recv_slice[1].first -
                                 (long int)send_slice[1].first;
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
              { recv_slice[0].first, recv_slice[1].first, comps.first },
              { recv_slice[0].second, recv_slice[1].second, comps.second }),
            Lambda(index_t i1, index_t i2, index_t ci) {
              fld(i1, i2, ci) += fld(i1 - offset_x1, i2 - offset_x2, ci);
            });
        } else if constexpr (D == Dim::_3D) {
          const auto offset_x1 = (long int)recv_slice[0].first -
                                 (long int)send_slice[0].first;
          const auto offset_x2 = (long int)recv_slice[1].first -
                                 (long int)send_slice[1].first;
          const auto offset_x3 = (long int)recv_slice[2].first -
                                 (long int)send_slice[2].first;
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>, AccelExeSpace>(
              { recv_slice[0].first,
                recv_slice[1].first,
                recv_slice[2].first,
                comps.first },
              { recv_slice[0].second,
                recv_slice[1].second,
                recv_slice[2].second,
                comps.second }),
            Lambda(index_t i1, index_t i2, index_t i3, index_t ci) {
              fld(i1, i2, i3, ci) += fld(i1 - offset_x1,
                                         i2 - offset_x2,
                                         i3 - offset_x3,
                                         ci);
            });
        }
      }
    }

    else {
      raise::Error("Multi domain without MPI is not supported yet", HERE);
    }
  }

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_NOMPI_HPP