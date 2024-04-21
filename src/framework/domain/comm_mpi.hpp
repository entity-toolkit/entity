/**
 * @file framework/domain/comm_mpi.hpp
 * @brief MPI communication routines
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
 * @note This should only be included if the MPI_ENABLED flag is set
 */

#ifndef FRAMEWORK_DOMAIN_COMM_MPI_HPP
#define FRAMEWORK_DOMAIN_COMM_MPI_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "utils/error.h"

#include "framework/domain/domain.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if ((send_to->mpi_rank() == rank && send_to->index() != idx) ||
        (recv_from->mpi_rank() == rank && recv_from->index() != idx)) {
      std::cout << "rank " << rank << " with index " << idx << std::endl;
      std::cout << "sending to " << send_to->index() << " with rank "
                << send_to->mpi_rank() << std::endl;
      std::cout << "receiving from " << recv_from->index() << " with rank "
                << recv_from->mpi_rank() << std::endl;
    }
    raise::ErrorIf(
      (send_to->mpi_rank() == rank && send_to->index() != idx) ||
        (recv_from->mpi_rank() == rank && recv_from->index() != idx),
      "Multiple-domain single-rank communication not yet implemented",
      HERE);

    //  trivial copy if sending to self and receiving from self
    if ((send_to->index() == idx) || (recv_from->index() == idx)) {
      raise::ErrorIf((recv_from->index() != idx) || (send_to->index() != idx),
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
      if (send_to != nullptr) {
        nsend *= (send_slice[d].second - send_slice[d].first);
      }
      if (recv_from != nullptr) {
        nrecv *= (recv_slice[d].second - recv_slice[d].first);
      }
    }

    if (send_to != nullptr) {
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
    if (recv_from != nullptr) {
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
    if (send_to != nullptr && recv_from != nullptr) {
      MPI_Sendrecv(send_fld.data(),
                   nsend,
                   mpi::get_type<real_t>(),
                   send_to->mpi_rank(),
                   0,
                   recv_fld.data(),
                   nrecv,
                   mpi::get_type<real_t>(),
                   recv_from->mpi_rank(),
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if (send_to != nullptr) {
      MPI_Send(send_fld.data(),
               nsend,
               mpi::get_type<real_t>(),
               send_to->mpi_rank(),
               0,
               MPI_COMM_WORLD);
    } else if (recv_from != nullptr) {
      MPI_Recv(recv_fld.data(),
               nrecv,
               mpi::get_type<real_t>(),
               recv_from->mpi_rank(),
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else {
      raise::Error("CommunicateField called with nullptrs", HERE);
    }
    if (recv_from != nullptr) {
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