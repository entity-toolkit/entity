/**
 * @file framework/domain/comm_mpi.hpp
 * @brief MPI communication routines
 * @implements
 *   - comm::CommunicateField<> -> void
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

#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <vector>

namespace comm {
  using namespace ntt;

  template <Dimension D, int N>
  inline void CommunicateField(unsigned int                      idx,
                               ndfield_t<D, N>&                  fld,
                               ndfield_t<D, N>&                  fld_buff,
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

    if ((send_idx == idx) and (recv_idx == idx)) {
      //  trivial copy if sending to self and receiving from self
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
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
              { recv_slice[0].first, comps.first },
              { recv_slice[0].second, comps.second }),
            Lambda(index_t i1, index_t ci) {
              fld_buff(i1, ci) += fld(i1 - offset_x1, ci);
            });
        } else if constexpr (D == Dim::_2D) {
          const auto offset_x1 = (long int)(recv_slice[0].first) -
                                 (long int)(send_slice[0].first);
          const auto offset_x2 = (long int)(recv_slice[1].first) -
                                 (long int)(send_slice[1].first);
          Kokkos::parallel_for(
            "CommunicateField-extract",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
              { recv_slice[0].first, recv_slice[1].first, comps.first },
              { recv_slice[0].second, recv_slice[1].second, comps.second }),
            Lambda(index_t i1, index_t i2, index_t ci) {
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
              fld_buff(i1, i2, i3, ci) += fld(i1 - offset_x1,
                                              i2 - offset_x2,
                                              i3 - offset_x3,
                                              ci);
            });
        }
      }
    } else {
      std::size_t nsend { comps.second - comps.first },
        nrecv { comps.second - comps.first };
      ndarray_t<static_cast<unsigned short>(D) + 1> send_fld, recv_fld;

      for (short d { 0 }; d < (short)D; ++d) {
        if (send_rank >= 0) {
          nsend *= (send_slice[d].second - send_slice[d].first);
        }
        if (recv_rank >= 0) {
          nrecv *= (recv_slice[d].second - recv_slice[d].first);
        }
      }
      if (send_rank >= 0) {
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
      if (recv_rank >= 0) {
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
      if (send_rank >= 0 && recv_rank >= 0) {
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
      } else if (send_rank >= 0) {
        MPI_Send(send_fld.data(),
                 nsend,
                 mpi::get_type<real_t>(),
                 send_rank,
                 0,
                 MPI_COMM_WORLD);
      } else if (recv_rank >= 0) {
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
      if (recv_rank >= 0) {
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
                fld_buff(i1, ci) += recv_fld(i1 - offset_x1, ci - offset_c);
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
                fld_buff(i1, i2, ci) += recv_fld(i1 - offset_x1,
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
                fld_buff(i1, i2, i3, ci) += recv_fld(i1 - offset_x1,
                                                     i2 - offset_x2,
                                                     i3 - offset_x3,
                                                     ci - offset_c);
              });
          }
        }
      }
    }
  }

  template <typename T>
  void CommunicateParticleQuantity(array_t<T*>&         arr,
                                   int                  send_rank,
                                   int                  recv_rank,
                                   const range_tuple_t& send_slice,
                                   const range_tuple_t& recv_slice) {
    const std::size_t send_count = send_slice.second - send_slice.first;
    const std::size_t recv_count = recv_slice.second - recv_slice.first;
    if ((send_rank >= 0) and (recv_rank >= 0) and (send_count > 0) and
        (recv_count > 0)) {
      MPI_Sendrecv(arr.data() + send_slice.first,
                   send_count,
                   mpi::get_type<T>(),
                   send_rank,
                   0,
                   arr.data() + recv_slice.first,
                   recv_count,
                   mpi::get_type<T>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if ((send_rank >= 0) and (send_count > 0)) {
      MPI_Send(arr.data() + send_slice.first,
               send_count,
               mpi::get_type<T>(),
               send_rank,
               0,
               MPI_COMM_WORLD);
    } else if ((recv_rank >= 0) and (recv_count > 0)) {
      MPI_Recv(arr.data() + recv_slice.first,
               recv_count,
               mpi::get_type<T>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }

  void ParticleSendRecvCount(int                send_rank,
                             int                recv_rank,
                             const std::size_t& send_count,
                             std::size_t&       recv_count) {
    if ((send_rank >= 0) && (recv_rank >= 0)) {
      MPI_Sendrecv(&send_count,
                   1,
                   mpi::get_type<std::size_t>(),
                   send_rank,
                   0,
                   &recv_count,
                   1,
                   mpi::get_type<std::size_t>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if (send_rank >= 0) {
      MPI_Send(&send_count, 1, mpi::get_type<std::size_t>(), send_rank, 0, MPI_COMM_WORLD);
    } else if (recv_rank >= 0) {
      MPI_Recv(&recv_count,
               1,
               mpi::get_type<std::size_t>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else {
      raise::Error("ParticleSendRecvCount called with negative ranks", HERE);
    }
  }

  template <Dimension D, Coord::type C>
  auto CommunicateParticles(Particles<D, C>&     species,
                            int                  send_rank,
                            int                  recv_rank,
                            const range_tuple_t& send_slice,
                            std::size_t&         index_last) -> std::size_t {
    if ((send_rank < 0) && (recv_rank < 0)) {
      raise::Error("No send or recv in CommunicateParticles", HERE);
    }
    std::size_t recv_count { 0 };
    ParticleSendRecvCount(send_rank,
                          recv_rank,
                          send_slice.second - send_slice.first,
                          recv_count);

    raise::FatalIf((index_last + recv_count) >= species.maxnpart(),
                   "Too many particles to receive (cannot fit into maxptl)",
                   HERE);
    const auto recv_slice = range_tuple_t({ index_last, index_last + recv_count });

    CommunicateParticleQuantity(species.i1, send_rank, recv_rank, send_slice, recv_slice);
    CommunicateParticleQuantity(species.dx1, send_rank, recv_rank, send_slice, recv_slice);
    CommunicateParticleQuantity(species.i1_prev,
                                send_rank,
                                recv_rank,
                                send_slice,
                                recv_slice);
    CommunicateParticleQuantity(species.dx1_prev,
                                send_rank,
                                recv_rank,
                                send_slice,
                                recv_slice);
    if constexpr (D == Dim::_2D || D == Dim::_3D) {
      CommunicateParticleQuantity(species.i2, send_rank, recv_rank, send_slice, recv_slice);
      CommunicateParticleQuantity(species.dx2,
                                  send_rank,
                                  recv_rank,
                                  send_slice,
                                  recv_slice);
      CommunicateParticleQuantity(species.i2_prev,
                                  send_rank,
                                  recv_rank,
                                  send_slice,
                                  recv_slice);
      CommunicateParticleQuantity(species.dx2_prev,
                                  send_rank,
                                  recv_rank,
                                  send_slice,
                                  recv_slice);
    }
    if constexpr (D == Dim::_3D) {
      CommunicateParticleQuantity(species.i3, send_rank, recv_rank, send_slice, recv_slice);
      CommunicateParticleQuantity(species.dx3,
                                  send_rank,
                                  recv_rank,
                                  send_slice,
                                  recv_slice);
      CommunicateParticleQuantity(species.i3_prev,
                                  send_rank,
                                  recv_rank,
                                  send_slice,
                                  recv_slice);
      CommunicateParticleQuantity(species.dx3_prev,
                                  send_rank,
                                  recv_rank,
                                  send_slice,
                                  recv_slice);
    }
    CommunicateParticleQuantity(species.ux1, send_rank, recv_rank, send_slice, recv_slice);
    CommunicateParticleQuantity(species.ux2, send_rank, recv_rank, send_slice, recv_slice);
    CommunicateParticleQuantity(species.ux3, send_rank, recv_rank, send_slice, recv_slice);
    CommunicateParticleQuantity(species.weight,
                                send_rank,
                                recv_rank,
                                send_slice,
                                recv_slice);
    if constexpr (D == Dim::_2D and C != Coord::Cart) {
      CommunicateParticleQuantity(species.phi,
                                  send_rank,
                                  recv_rank,
                                  send_slice,
                                  recv_slice);
    }
    for (auto p { 0 }; p < species.npld(); ++p) {
      CommunicateParticleQuantity(species.pld[p],
                                  send_rank,
                                  recv_rank,
                                  send_slice,
                                  recv_slice);
    }
    return recv_count;
  }

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_MPI_HPP
