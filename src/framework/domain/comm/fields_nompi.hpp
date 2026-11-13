/**
 * @file framework/domain/comm/fields_nompi.hpp
 * @brief Communication routines for fields without mpi
 * @implements
 *   - comm::PendingComm
 *   - comm::CommunicateField_Post<> -> comm::PendingComm
 *   - comm::CommunicateField<> -> void
 * @namespaces:
 *   - comm::
 * @note This should only be included if the MPI_ENABLED flag is not set
 * @note Mirrors fields_mpi.hpp's interface so callers (fields.cpp, fields_sync.cpp)
 *       don't need to branch: with no MPI there is only ever one domain, so
 *       every call lands on the self-case below and PendingComm always comes
 *       back empty (no requests, no deferred finalize) -- there's nothing to
 *       wait on.
 */

#ifndef FRAMEWORK_DOMAIN_COMM_FIELDS_NOMPI_HPP
#define FRAMEWORK_DOMAIN_COMM_FIELDS_NOMPI_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include <Kokkos_Core.hpp>

#include <functional>
#include <vector>

namespace comm {
  using namespace ntt;

  /**
   * @note With MPI_ENABLED off, `requests` never gets populated -- there's
   *       only ever one domain, so CommunicateField_Post always takes the
   *       self-case branch and does its work eagerly. The type only needs to
   *       be a placeholder here; nothing ever constructs one.
   */
  struct PendingComm {
    std::vector<int>      requests;
    std::function<void()> finalize;
  };

  /**
   * @note: Send `fld`, recv to `fld_buff`
   * @note: `fld` and `fld_buff` may be the same
   */
  template <Dimension D, int N>
  inline auto CommunicateField_Post(unsigned int                     idx,
                                    ndfield_t<D, N>&                 fld,
                                    ndfield_t<D, N>&                 fld_buff,
                                    unsigned int                     send_idx,
                                    unsigned int                     recv_idx,
                                    int                              send_rank,
                                    int                              recv_rank,
                                    const std::vector<cell_range_t>& send_slice,
                                    const std::vector<cell_range_t>& recv_slice,
                                    const cell_range_t&              comps,
                                    bool additive) -> PendingComm {
    raise::ErrorIf(send_rank < 0 && recv_rank < 0,
                   "CommunicateField_Post called with negative ranks",
                   HERE);

    //  trivial copy if sending to self and receiving from self
    if ((send_idx == idx) || (recv_idx == idx)) {
      raise::ErrorIf((recv_idx != idx) || (send_idx != idx),
                     "Cannot send to self and receive from another domain",
                     HERE);
      // sending/recv to/from self
      if (not additive) {
        // simply filling the ghost cells
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
        // adding received fields to ghosts + active
        if constexpr (D == Dim::_1D) {
          const auto offset_x1 = (long int)(recv_slice[0].first) -
                                 (long int)(send_slice[0].first);
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
              { recv_slice[0].first, comps.first },
              { recv_slice[0].second, comps.second }),
            Lambda(cellidx_t i1, cellidx_t ci) {
              fld_buff(i1, ci) += fld(i1 - offset_x1, ci);
            });
        } else if constexpr (D == Dim::_2D) {
          const auto offset_x1 = (long int)(recv_slice[0].first) -
                                 (long int)(send_slice[0].first);
          const auto offset_x2 = (long int)(recv_slice[1].first) -
                                 (long int)(send_slice[1].first);
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>(
              { recv_slice[0].first, recv_slice[1].first, comps.first },
              { recv_slice[0].second, recv_slice[1].second, comps.second }),
            Lambda(cellidx_t i1, cellidx_t i2, cellidx_t ci) {
              fld_buff(i1, i2, ci) += fld(i1 - offset_x1, i2 - offset_x2, ci);
            });
        } else if constexpr (D == Dim::_3D) {
          const auto offset_x1 = (long int)(recv_slice[0].first) -
                                 (long int)(send_slice[0].first);
          const auto offset_x2 = (long int)(recv_slice[1].first) -
                                 (long int)(send_slice[1].first);
          const auto offset_x3 = (long int)(recv_slice[2].first) -
                                 (long int)(send_slice[2].first);
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>, Kokkos::DefaultExecutionSpace>(
              { recv_slice[0].first,
                recv_slice[1].first,
                recv_slice[2].first,
                comps.first },
              { recv_slice[0].second,
                recv_slice[1].second,
                recv_slice[2].second,
                comps.second }),
            Lambda(cellidx_t i1, cellidx_t i2, cellidx_t i3, cellidx_t ci) {
              fld_buff(i1, i2, i3, ci) += fld(i1 - offset_x1,
                                              i2 - offset_x2,
                                              i3 - offset_x3,
                                              ci);
            });
        }
      }
    } else {
      raise::Error("Multi domain without MPI is not supported yet", HERE);
    }
    return {};
  }

  /**
   * @brief Blocking convenience wrapper around CommunicateField_Post():
   *        posts the exchange, and runs its finalize before
   *        returning.
   * @note For one-off exchanges (and the comm tests); batched halo exchanges
   *       should collect CommunicateField_Post() results and wait once.
   */
  template <Dimension D, int N>
  inline void CommunicateField(unsigned int                     idx,
                               ndfield_t<D, N>&                 fld,
                               ndfield_t<D, N>&                 fld_buff,
                               unsigned int                     send_idx,
                               unsigned int                     recv_idx,
                               int                              send_rank,
                               int                              recv_rank,
                               const std::vector<cell_range_t>& send_slice,
                               const std::vector<cell_range_t>& recv_slice,
                               const cell_range_t&              comps,
                               bool                             additive) {
    auto pending = CommunicateField_Post<D, N>(idx,
                                               fld,
                                               fld_buff,
                                               send_idx,
                                               recv_idx,
                                               send_rank,
                                               recv_rank,
                                               send_slice,
                                               recv_slice,
                                               comps,
                                               additive);
    if (pending.finalize) {
      pending.finalize();
    }
  }

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_FIELDS_NOMPI_HPP
