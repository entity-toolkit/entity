/**
 * @file framework/domain/comm_mpi.hpp
 * @brief MPI communication routines
 * @implements
 *   - comm::CommunicateField<> -> void
 * @depends:
 *   - global.h
 *   - arch/kokkos_aliases.h
 *   - arch/mpi_aliases.h
 *   - utils/error.h
 *   - framework/domain/domain.h
 * @namespaces:
 *   - comm::
 * @note This should only be included if the MPI_ENABLED flag is set
 */

#ifndef FRAMEWORK_DOMAIN_COMM_MPI_HPP
#define FRAMEWORK_DOMAIN_COMM_MPI_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "utils/error.h"

#include "framework/domain/domain.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace comm {
  using namespace ntt;

  template <Dimension D, int N>
  inline void CommunicateField(unsigned int                      idx,
                               ndfield_t<D, N>&                  fld,
                               unsigned int                      send_idx,
                               unsigned int                      recv_idx,
                               int                               send_rank,
                               int                               recv_rank,
                               const std::vector<range_tuple_t>& send_slice,
                               const std::vector<range_tuple_t>& recv_slice,
                               const range_tuple_t&              comps,
                               bool                              additive) {
    raise::ErrorIf(send_rank < 0 && recv_rank < 0,
                   "CommunicateField called with negative ranks",
                   HERE);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    raise::ErrorIf(
      (send_rank == rank && send_idx != idx) ||
        (recv_rank == rank && recv_idx != idx),
      "Multiple-domain single-rank communication not yet implemented",
      HERE);

    //  trivial copy if sending to self and receiving from self
    if ((send_idx == idx) || (recv_idx == idx)) {
      raise::ErrorIf((recv_idx != idx) || (send_idx != idx),
                     "Cannot send to self and receive from another domain",
                     HERE);
      // sending/recv to/from self
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
      return;
    }

    std::size_t nsend { comps.second - comps.first },
      nrecv { comps.second - comps.first };
    ndarray_t<static_cast<unsigned short>(D) + 1> send_fld, recv_fld;
    for (short d { 0 }; d < (short)D; ++d) {
      if (send_rank > 0) {
        nsend *= (send_slice[d].second - send_slice[d].first);
      }
      if (recv_rank > 0) {
        nrecv *= (recv_slice[d].second - recv_slice[d].first);
      }
    }

    if (send_rank > 0) {
      if constexpr (D == Dim::_1D) {
        send_fld = ndarray_t<2>("send_fld",
                                send_slice[0].second - send_slice[0].first,
                                comps.second - comps.first);
        Kokkos::deep_copy(send_fld, Kokkos::subview(fld, send_slice[0], comps));
      } else if constexpr (D == Dim::_2D) {
        send_fld = ndarray_t<3>("send_fld",
                                send_slice[0].second - send_slice[0].first,
                                send_slice[1].second - send_slice[1].first,
                                comps.second - comps.first);
        Kokkos::deep_copy(
          send_fld,
          Kokkos::subview(fld, send_slice[0], send_slice[1], comps));
      } else if constexpr (D == Dim::_3D) {
        send_fld = ndarray_t<4>("send_fld",
                                send_slice[0].second - send_slice[0].first,
                                send_slice[1].second - send_slice[1].first,
                                send_slice[2].second - send_slice[2].first,
                                comps.second - comps.first);
        Kokkos::deep_copy(
          send_fld,
          Kokkos::subview(fld, send_slice[0], send_slice[1], send_slice[2], comps));
      }
    }
    if (recv_rank > 0) {
      if constexpr (D == Dim::_1D) {
        recv_fld = ndarray_t<2>("recv_fld",
                                recv_slice[0].second - recv_slice[0].first,
                                comps.second - comps.first);
      } else if constexpr (D == Dim::_2D) {
        recv_fld = ndarray_t<3>("recv_fld",
                                recv_slice[0].second - recv_slice[0].first,
                                recv_slice[1].second - recv_slice[1].first,
                                comps.second - comps.first);
      } else if constexpr (D == Dim::_3D) {
        recv_fld = ndarray_t<4>("recv_fld",
                                recv_slice[0].second - recv_slice[0].first,
                                recv_slice[1].second - recv_slice[1].first,
                                recv_slice[2].second - recv_slice[2].first,
                                comps.second - comps.first);
      }
    }
    if (send_rank > 0 && recv_rank > 0) {
      MPI_Sendrecv(send_fld.data(),
                   nsend,
                   mpi::get_type<real_t>(),
                   send_rank,
                   0,
                   recv_fld.data(),
                   nrecv,
                   mpi::get_type<real_t>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if (send_rank > 0) {
      MPI_Send(send_fld.data(), nsend, mpi::get_type<real_t>(), send_rank, 0, MPI_COMM_WORLD);
    } else if (recv_rank > 0) {
      MPI_Recv(recv_fld.data(),
               nrecv,
               mpi::get_type<real_t>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else {
      raise::Error("CommunicateField called with negative ranks", HERE);
    }
    if (recv_rank > 0) {
      // !TODO: perhaps directly recv to the fld?
      if (not additive) {
        if constexpr (D == Dim::_1D) {
          Kokkos::deep_copy(Kokkos::subview(fld, recv_slice[0], comps), recv_fld);
        } else if constexpr (D == Dim::_2D) {
          Kokkos::deep_copy(
            Kokkos::subview(fld, recv_slice[0], recv_slice[1], comps),
            recv_fld);
        } else if constexpr (D == Dim::_3D) {
          Kokkos::deep_copy(
            Kokkos::subview(fld, recv_slice[0], recv_slice[1], recv_slice[2], comps),
            recv_fld);
        }
      } else {
        if constexpr (D == Dim::_1D) {
          const auto offset_x1 = recv_slice[0].first;
          const auto offset_c  = comps.first;
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
              { recv_slice[0].first, comps.first },
              { recv_slice[0].second, comps.second }),
            Lambda(index_t i1, index_t ci) {
              fld(i1, ci) += recv_fld(i1 - offset_x1, ci - offset_c);
            });
        } else if constexpr (D == Dim::_2D) {
          const auto offset_x1 = recv_slice[0].first;
          const auto offset_x2 = recv_slice[1].first;
          const auto offset_c  = comps.first;
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
              { recv_slice[0].first, recv_slice[1].first, comps.first },
              { recv_slice[0].second, recv_slice[1].second, comps.second }),
            Lambda(index_t i1, index_t i2, index_t ci) {
              fld(i1, i2, ci) += recv_fld(i1 - offset_x1,
                                          i2 - offset_x2,
                                          ci - offset_c);
            });
        } else if constexpr (D == Dim::_3D) {
          const auto offset_x1 = recv_slice[0].first;
          const auto offset_x2 = recv_slice[1].first;
          const auto offset_x3 = recv_slice[2].first;
          const auto offset_c  = comps.first;
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
              fld(i1, i2, i3, ci) += recv_fld(i1 - offset_x1,
                                              i2 - offset_x2,
                                              i3 - offset_x3,
                                              ci - offset_c);
            });
        }
      }
    }
  }
} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_MPI_HPP