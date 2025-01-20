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

#include "arch/directions.h"
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


  template <Dimension D, Coord::type C>
  void CommunicateParticlesBuffer(Particles<D, C>&     species,
                                  Kokkos::View<std::size_t*>  permute_vector,
                                  Kokkos::View<std::size_t*>  allocation_vector,
                                  Kokkos::View<std::size_t*>  tag_offset,
                                  std::vector<std::size_t>    npart_per_tag_arr,
                                  std::vector<std::size_t>    npart_per_tag_arr_recv,
                                  std::vector<int>            send_ranks,
                                  std::vector<int>            recv_ranks,
				  const dir::dirs_t<D>&    legal_directions) {
    // Pointers to the particle data arrays
    auto &this_ux1 = species.ux1;
    auto &this_ux2 = species.ux2;
    auto &this_ux3 = species.ux3;
    auto &this_weight = species.weight;
    auto &this_phi = species.phi;
    auto &this_i1 = species.i1;
    auto &this_i1_prev = species.i1_prev;
    auto &this_i2 = species.i2;
    auto &this_i3 = species.i3;
    auto &this_i2_prev = species.i2_prev;
    auto &this_i3_prev = species.i3_prev;
    auto &this_dx1 = species.dx1;
    auto &this_dx1_prev = species.dx1_prev;
    auto &this_dx2 = species.dx2;
    auto &this_dx3 = species.dx3;
    auto &this_dx2_prev = species.dx2_prev;
    auto &this_dx3_prev = species.dx3_prev;
    auto &this_tag = species.tag;
    auto &this_particleID = species.particleID;

    // Number of arrays of each type to send/recv
    auto NREALS   = 4;
    auto NINTS    = 2;
    auto NFLOATS  = 2;
    auto NLONGS   = 2;
    if constexpr (D == Dim::_2D) {
      if (C != Coord::Cart) {
        NREALS  = 5;
        NINTS   = 4;
        NFLOATS = 4;
        this_phi = species.phi;
      } else {
        NREALS  = 4;
        NINTS   = 4;
        NFLOATS = 4;
      }
    }
    if constexpr (D == Dim::_3D) {
      NREALS  = 4;
      NINTS   = 6;
      NFLOATS = 6;
    }

    // Now make buffers to store recevied data (don't need global send buffers)
    const auto total_send = permute_vector.extent(0) - npart_per_tag_arr[ParticleTag::dead];
    const auto total_recv = allocation_vector.extent(0);
    const auto n_alive    = npart_per_tag_arr[ParticleTag::alive];
    const auto n_dead     = npart_per_tag_arr[ParticleTag::dead];

    // Debug test: print send and recv count
    {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      //printf("MPI rank: %d, Total send: %d, Total recv: %d \n", rank, total_send, total_recv);
    }
    /*
        Brief on recv buffers: Each recv buffer contains all the received arrays of 
        a given type. The different physical quantities are stored next to each other
        to avoid cache misses. The array is structured as follows:
        E.g.,
        recv_buffer_int: | qty1  | qty2 | ... | qtyNINTS | qty1 | qty2 | ... | qtyNINTS | ...
                         <-------particle to recv1------> <-------particle to recv2-------->
                         <----------------------------------total_recv---------------------------->
    */
    Kokkos::View<int*>        recv_buffer_int("recv_buffer_int",      total_recv * NINTS);
    Kokkos::View<real_t*>     recv_buffer_real("recv_buffer_real",    total_recv * NREALS);
    Kokkos::View<prtldx_t*>   recv_buffer_prtldx("recv_buffer_prtldx",total_recv * NFLOATS);
    Kokkos::View<long*>       recv_buffer_long("recv_buffer_long",    total_recv * NLONGS);
    auto recv_buffer_int_h    = Kokkos::create_mirror_view(recv_buffer_int);
    auto recv_buffer_real_h   = Kokkos::create_mirror_view(recv_buffer_real);
    auto recv_buffer_prtldx_h = Kokkos::create_mirror_view(recv_buffer_prtldx);
    auto recv_buffer_long_h   = Kokkos::create_mirror_view(recv_buffer_long);

    auto iteration = 0;
    auto current_received = 0;
    for (const auto& direction : legal_directions) {
      const auto send_rank    = send_ranks[iteration];
      const auto recv_rank    = recv_ranks[iteration];
      const auto tag_send     = mpi::PrtlSendTag<D>::dir2tag(direction);
      const auto tag_recv     = mpi::PrtlSendTag<D>::dir2tag(-direction);
      const auto send_count   = npart_per_tag_arr[tag_send];
      const auto recv_count   = npart_per_tag_arr_recv[tag_recv];
      {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      }
      if (send_rank < 0 and recv_rank < 0) {
        continue;
      }
      Kokkos::View<int*>      send_buffer_int("send_buffer_int",      send_count * NINTS);
      Kokkos::View<real_t*>   send_buffer_real("send_buffer_real",    send_count * NREALS);
      Kokkos::View<prtldx_t*> send_buffer_prtldx("send_buffer_prtldx",send_count * NFLOATS);
      Kokkos::View<long*>     send_buffer_long("send_buffer_long",    send_count * NLONGS);
      auto send_buffer_int_h = Kokkos::create_mirror_view(send_buffer_int);
      auto send_buffer_real_h = Kokkos::create_mirror_view(send_buffer_real);
      auto send_buffer_prtldx_h = Kokkos::create_mirror_view(send_buffer_prtldx);
      auto send_buffer_long_h = Kokkos::create_mirror_view(send_buffer_long);

      // Need different constexpr parallel fors for different dims
      if constexpr(D == Dim::_1D) {
        Kokkos::parallel_for(
          "PopulateSendBuffer",
          send_count,
          Lambda(const std::size_t p){
            const auto idx = permute_vector(tag_offset(tag_send) - n_alive + p);
            send_buffer_int(NINTS * p + 0) = this_i1(idx);
            send_buffer_int(NINTS * p + 1) = this_i1_prev(idx);
            send_buffer_real(NREALS * p + 0) = this_ux1(idx);
            send_buffer_real(NREALS * p + 1) = this_ux2(idx);
            send_buffer_real(NREALS * p + 2) = this_ux3(idx);
            send_buffer_real(NREALS * p + 3) = this_weight(idx);
            send_buffer_prtldx(NFLOATS * p + 0) = this_dx1(idx);
            send_buffer_prtldx(NFLOATS * p + 1) = this_dx1_prev(idx);
            send_buffer_long(NLONGS * p + 0) = this_particleID(idx);
            send_buffer_long(NLONGS * p + 1) = this_tag(idx);
            this_tag(idx)  = ParticleTag::dead;
          });
      }
      if constexpr(D == Dim::_2D && C == Coord::Cart) {
        Kokkos::parallel_for(
          "PopulateSendBuffer",
          send_count,
          Lambda(const std::size_t p){
            const auto idx = permute_vector(tag_offset(tag_send) - n_alive + p);
            send_buffer_int(NINTS * p + 0) = this_i1(idx);
            send_buffer_int(NINTS * p + 1) = this_i1_prev(idx);
            send_buffer_int(NINTS * p + 2) = this_i2(idx);
            send_buffer_int(NINTS * p + 3) = this_i2_prev(idx);     
            send_buffer_real(NREALS * p + 0) = this_ux1(idx);
            send_buffer_real(NREALS * p + 1) = this_ux2(idx);
            send_buffer_real(NREALS * p + 2) = this_ux3(idx);
            send_buffer_real(NREALS * p + 3) = this_weight(idx);
            send_buffer_prtldx(NFLOATS * p + 0) = this_dx1(idx);
            send_buffer_prtldx(NFLOATS * p + 1) = this_dx1_prev(idx);
            send_buffer_prtldx(NFLOATS * p + 2) = this_dx2(idx);
            send_buffer_prtldx(NFLOATS * p + 3) = this_dx2_prev(idx);
            send_buffer_long(NLONGS * p + 0) = this_particleID(idx);
            send_buffer_long(NLONGS * p + 1) = this_tag(idx);
            this_tag(idx)  = ParticleTag::dead;
          });
      }
      if constexpr(D == Dim::_2D && C != Coord::Cart) {
        Kokkos::parallel_for(
          "PopulateSendBuffer",
          send_count,
          Lambda(const std::size_t p){
            const auto idx = permute_vector(tag_offset(tag_send) - n_alive + p);
            send_buffer_int(NINTS * p + 0) = this_i1(idx);
            send_buffer_int(NINTS * p + 1) = this_i1_prev(idx);
            send_buffer_int(NINTS * p + 2) = this_i2(idx);
            send_buffer_int(NINTS * p + 3) = this_i2_prev(idx);     
            send_buffer_real(NREALS * p + 0) = this_ux1(idx);
            send_buffer_real(NREALS * p + 1) = this_ux2(idx);
            send_buffer_real(NREALS * p + 2) = this_ux3(idx);
            send_buffer_real(NREALS * p + 3) = this_weight(idx);
            send_buffer_real(NREALS * p + 4) = this_phi(idx);
            send_buffer_prtldx(NFLOATS * p + 0) = this_dx1(idx);
            send_buffer_prtldx(NFLOATS * p + 1) = this_dx1_prev(idx);
            send_buffer_prtldx(NFLOATS * p + 2) = this_dx2(idx);
            send_buffer_prtldx(NFLOATS * p + 3) = this_dx2_prev(idx);
            send_buffer_long(NLONGS * p + 0) = this_particleID(idx);
            send_buffer_long(NLONGS * p + 1) = this_tag(idx);
            this_tag(idx)  = ParticleTag::dead;
          });
      }
      if constexpr(D == Dim::_3D) {
        Kokkos::parallel_for(
          "PopulateSendBuffer",
          send_count,
          Lambda(const std::size_t p){
            const auto idx = permute_vector(tag_offset(tag_send) - n_alive + p);
            send_buffer_int(NINTS * p + 0) = this_i1(idx);
            send_buffer_int(NINTS * p + 1) = this_i1_prev(idx);
            send_buffer_int(NINTS * p + 2) = this_i2(idx);
            send_buffer_int(NINTS * p + 3) = this_i2_prev(idx);
            send_buffer_int(NINTS * p + 4) = this_i3(idx);
            send_buffer_int(NINTS * p + 5) = this_i3_prev(idx);
            send_buffer_real(NREALS * p + 0) = this_ux1(idx);
            send_buffer_real(NREALS * p + 1) = this_ux2(idx);
            send_buffer_real(NREALS * p + 2) = this_ux3(idx);
            send_buffer_real(NREALS * p + 3) = this_weight(idx);
            send_buffer_prtldx(NFLOATS * p + 0) = this_dx1(idx);
            send_buffer_prtldx(NFLOATS * p + 1) = this_dx1_prev(idx);
            send_buffer_prtldx(NFLOATS * p + 2) = this_dx2(idx);
            send_buffer_prtldx(NFLOATS * p + 3) = this_dx2_prev(idx);
            send_buffer_prtldx(NFLOATS * p + 4) = this_dx3(idx);
            send_buffer_prtldx(NFLOATS * p + 5) = this_dx3_prev(idx);
            send_buffer_long(NLONGS * p + 0) = this_particleID(idx);
            send_buffer_long(NLONGS * p + 1) = this_tag(idx);
            this_tag(idx)  = ParticleTag::dead;
          });
      }

      auto tag_offset_h = Kokkos::create_mirror_view(tag_offset);
      Kokkos::deep_copy(tag_offset_h, tag_offset);
      /*
          Brief on receive offset:
          The receive buffer looks like this
          <----------------------------------->
          |NINT|NINT|NINT|NINT|NINT|NINT|NINT|NINT|...xnrecv
          <--------><--------><--------><-------->
            recv1      recv2      recv3     recv4
                    |________|
                    ^        ^
                  offset   offset + nrecv
      */
      const auto receive_offset_int     = current_received * NINTS;
      const auto receive_offset_real    = current_received * NREALS;
      const auto receive_offset_prtldx  = current_received * NFLOATS;
      const auto receive_offset_long    = current_received * NLONGS;
      // Comms
      // Make host arrays for send and recv buffers
      Kokkos::deep_copy(send_buffer_int_h, send_buffer_int);
      Kokkos::deep_copy(send_buffer_real_h, send_buffer_real);
      Kokkos::deep_copy(send_buffer_prtldx_h, send_buffer_prtldx);
      Kokkos::deep_copy(send_buffer_long_h, send_buffer_long);

      if ((send_rank >= 0) and (recv_rank >= 0) and (send_count > 0) and
      (recv_count > 0)) {
      // Debug: Print the rank and type of mpi operation performed
      {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      //printf("MPI rank: %d, Performing sendrecv operation, direction %d \n", rank, direction);
      }
      MPI_Sendrecv(send_buffer_int.data(),
                   send_count * NINTS,
                   mpi::get_type<int>(),
                   send_rank,
                   0,
                   recv_buffer_int.data() + receive_offset_int,
                   recv_count*NINTS,
                   mpi::get_type<int>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
      MPI_Sendrecv(send_buffer_real.data(),
                    send_count * NREALS,
                    mpi::get_type<real_t>(),
                    send_rank,
                    0,
                    recv_buffer_real.data() + receive_offset_real,
                    recv_count*NREALS,
                    mpi::get_type<real_t>(),
                    recv_rank,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
      MPI_Sendrecv(send_buffer_prtldx.data(),
                    send_count * NFLOATS,
                    mpi::get_type<prtldx_t>(),
                    send_rank,
                    0,
                    recv_buffer_prtldx.data() + receive_offset_prtldx,
                    recv_count*NFLOATS,
                    mpi::get_type<prtldx_t>(),
                    recv_rank,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
      MPI_Sendrecv(send_buffer_long.data(),
                    send_count * NLONGS,
                    mpi::get_type<long>(),
                    send_rank,
                    0,
                    recv_buffer_long.data() + receive_offset_long,
                    recv_count*NLONGS,
                    mpi::get_type<long>(),
                    recv_rank,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
    } else if ((send_rank >= 0) and (send_count > 0)) {
      // Debug: Print the rank and type of mpi operation performed
      {      
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //printf("MPI rank: %d, Performing send operation, direction %d \n", rank, direction);
      }
      MPI_Send(send_buffer_int.data(),
               send_count * NINTS,
               mpi::get_type<int>(),
               send_rank,
               0,
               MPI_COMM_WORLD);
      MPI_Send(send_buffer_real.data(),
                send_count * NREALS,
                mpi::get_type<real_t>(),
                send_rank,
                0,
                MPI_COMM_WORLD);
      MPI_Send(send_buffer_prtldx.data(),
                send_count * NFLOATS,
                mpi::get_type<prtldx_t>(),
                send_rank,
                0,
                MPI_COMM_WORLD);
      MPI_Send(send_buffer_long.data(),
                send_count * NLONGS,
                mpi::get_type<long>(),
                send_rank,
                0,
                MPI_COMM_WORLD);
      } else if ((recv_rank >= 0) and (recv_count > 0)) {
      // Debug: Print the rank and type of mpi operation performed
      {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //printf("MPI rank: %d, Performing recv operation, direction %d \n", rank, direction);
      }
      MPI_Recv(recv_buffer_int.data() + receive_offset_int,
               recv_count * NINTS,
               mpi::get_type<int>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(recv_buffer_real.data() + receive_offset_real,
                recv_count * NREALS,
                mpi::get_type<real_t>(),
                recv_rank,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
      MPI_Recv(recv_buffer_prtldx.data() + receive_offset_prtldx,
                recv_count * NFLOATS,
                mpi::get_type<prtldx_t>(),
                recv_rank,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
      MPI_Recv(recv_buffer_long.data() + receive_offset_long,
                recv_count * NLONGS,
                mpi::get_type<long>(),
                recv_rank,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
    }
    current_received += recv_count;
    iteration++;

    // Debug test: Print recv buffer before and after
    /*
    {
      int total_ranks;
      MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);
      for (int allranks=0; allranks<total_ranks; allranks++)
      {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == allranks && species.label() == "e+_b" && total_send != total_recv) 
        {
          // Print sent and received particles
          std::cout << "Species: " << species.label() << std::endl;
          //std::cout << "send rank: " << send_rank << "  recv rank: " << recv_rank << std::endl;
          // Print the tags of the dead particles

          
          for (int i=0; i<send_count; i++)
          {
            auto position_y = send_buffer_int_h(i*NINTS + 2);
            auto dx_y = send_buffer_prtldx_h(i*NFLOATS + 2);
            auto velocity_y = send_buffer_real_h(i*NREALS + 1);
            auto particle_ID = send_buffer_long_h(i*NLONGS + 0);
            auto particle_tag = send_buffer_long_h(i*NLONGS + 1);
            printf("MPI rank: %d,  Send ID %ld, tag %ld, pos i2 %d, dx: %f and u2 %f \n", rank,
            particle_ID, particle_tag, position_y, dx_y, velocity_y);
          }
          auto allocation_vector_h = Kokkos::create_mirror_view(allocation_vector);
          Kokkos::deep_copy(allocation_vector_h, allocation_vector);

          for (int i=0; i<recv_count; i++)
          {
            auto position_y = recv_buffer_int_h(receive_offset_int + i*NINTS + 2);
            auto velocity_y = recv_buffer_real_h(receive_offset_real + i*NREALS + 1);
            auto dx_y = recv_buffer_prtldx_h(receive_offset_prtldx + i*NFLOATS + 2);
            auto particle_ID = recv_buffer_long_h(receive_offset_long + i*NLONGS + 0);
            auto particle_tag = recv_buffer_long_h(receive_offset_long + i*NLONGS + 1);
            printf("MPI rank: %d,  Recv ID %ld, tag %ld, pos i2 %d, dx: %f  and u2 %f ", rank,
            particle_ID, particle_tag, position_y, dx_y, velocity_y);
            // Print where this particle gets allocated to
            auto idx = current_received + i - recv_count;
            std::cout << "\n current recieved " << current_received;
            std::cout << "\n current index " << idx;
            std::cout << "\n  allocated to position " << allocation_vector_h(idx) << std::endl;
          }
          printf("***** \n");
        }
      }
    }
    */
   
    } // end over direction loop
    /*Kokkos::deep_copy(recv_buffer_int, recv_buffer_int_h);
    Kokkos::deep_copy(recv_buffer_real, recv_buffer_real_h);
    Kokkos::deep_copy(recv_buffer_prtldx, recv_buffer_prtldx_h);*/
    if constexpr (D == Dim::_1D)
    {
      Kokkos::parallel_for(
      "PopulateFromRecvBuffer",
      total_recv,
      Lambda(const std::size_t p){
        auto idx = allocation_vector(p);
        this_tag(idx)       = ParticleTag::alive;
        this_i1(idx)        = recv_buffer_int(NINTS * p + 0);
        this_i1_prev(idx)   = recv_buffer_int(NINTS * p + 1);
        this_ux1(idx)       = recv_buffer_real(NREALS * p + 0);
        this_ux2(idx)       = recv_buffer_real(NREALS * p + 1);
        this_ux3(idx)       = recv_buffer_real(NREALS * p + 2);
        this_weight(idx)    = recv_buffer_real(NREALS * p + 3);
        this_dx1(idx)       = recv_buffer_prtldx(NFLOATS * p + 0);
        this_dx1_prev(idx)  = recv_buffer_prtldx(NFLOATS * p + 1);
        this_particleID(idx) = recv_buffer_long(NLONGS * p + 0);
    });
    }

    if constexpr (D == Dim::_2D && C == Coord::Cart)
    {
    Kokkos::parallel_for(
    "PopulateFromRecvBuffer",
    total_recv,
    Lambda(const std::size_t p){
      auto idx = allocation_vector(p);
      this_tag(idx)       = ParticleTag::alive;
      this_i1(idx)        = recv_buffer_int(NINTS * p + 0);
      this_i1_prev(idx)   = recv_buffer_int(NINTS * p + 1);
      this_i2(idx)        = recv_buffer_int(NINTS * p + 2);
      this_i2_prev(idx)   = recv_buffer_int(NINTS * p + 3);
      this_ux1(idx)       = recv_buffer_real(NREALS * p + 0);
      this_ux2(idx)       = recv_buffer_real(NREALS * p + 1);
      this_ux3(idx)       = recv_buffer_real(NREALS * p + 2);
      this_weight(idx)    = recv_buffer_real(NREALS * p + 3);
      this_dx1(idx)       = recv_buffer_prtldx(NFLOATS * p + 0);
      this_dx1_prev(idx)  = recv_buffer_prtldx(NFLOATS * p + 1);
      this_dx2(idx)       = recv_buffer_prtldx(NFLOATS * p + 2);
      this_dx2_prev(idx)  = recv_buffer_prtldx(NFLOATS * p + 3);
      this_particleID(idx) = recv_buffer_long(NLONGS * p + 0);
    });
    }

    if constexpr (D == Dim::_2D && C != Coord::Cart)
    {
      Kokkos::parallel_for(
      "PopulateFromRecvBuffer",
      total_recv,
      Lambda(const std::size_t p){
        auto idx = allocation_vector(p);
        this_tag(idx)       = ParticleTag::alive;
        this_i1(idx)        = recv_buffer_int(NINTS * p + 0);
        this_i1_prev(idx)   = recv_buffer_int(NINTS * p + 1);
        this_i2(idx)        = recv_buffer_int(NINTS * p + 2);
        this_i2_prev(idx)   = recv_buffer_int(NINTS * p + 3);
        this_ux1(idx)       = recv_buffer_real(NREALS * p + 0);
        this_ux2(idx)       = recv_buffer_real(NREALS * p + 1);
        this_ux3(idx)       = recv_buffer_real(NREALS * p + 2);
        this_weight(idx)    = recv_buffer_real(NREALS * p + 3);
        this_phi(idx)       = recv_buffer_real(NREALS * p + 4);
        this_dx1(idx)       = recv_buffer_prtldx(NFLOATS * p + 0);
        this_dx1_prev(idx)  = recv_buffer_prtldx(NFLOATS * p + 1);
        this_dx2(idx)       = recv_buffer_prtldx(NFLOATS * p + 2);
        this_dx2_prev(idx)  = recv_buffer_prtldx(NFLOATS * p + 3);
        this_particleID(idx)     = recv_buffer_long(NLONGS * p + 0);
    });
    }

    if constexpr (D == Dim::_3D)
    {
      Kokkos::parallel_for(
      "PopulateFromRecvBuffer",
      total_recv,
      Lambda(const std::size_t p){
        auto idx = allocation_vector(p);
        this_tag(idx)       = ParticleTag::alive;
        this_i1(idx)        = recv_buffer_int(NINTS * p + 0);
        this_i1_prev(idx)   = recv_buffer_int(NINTS * p + 1);
        this_i2(idx)        = recv_buffer_int(NINTS * p + 2);
        this_i2_prev(idx)   = recv_buffer_int(NINTS * p + 3);
        this_i3(idx)        = recv_buffer_int(NINTS * p + 4);
        this_i3_prev(idx)   = recv_buffer_int(NINTS * p + 5);
        this_ux1(idx)       = recv_buffer_real(NREALS * p + 0);
        this_ux2(idx)       = recv_buffer_real(NREALS * p + 1);
        this_ux3(idx)       = recv_buffer_real(NREALS * p + 2);
        this_weight(idx)    = recv_buffer_real(NREALS * p + 3);
        this_dx1(idx)       = recv_buffer_prtldx(NFLOATS * p + 0);
        this_dx1_prev(idx)  = recv_buffer_prtldx(NFLOATS * p + 1);
        this_dx2(idx)       = recv_buffer_prtldx(NFLOATS * p + 2);
        this_dx2_prev(idx)  = recv_buffer_prtldx(NFLOATS * p + 3);
        this_dx3(idx)       = recv_buffer_prtldx(NFLOATS * p + 4);
        this_dx3_prev(idx)  = recv_buffer_prtldx(NFLOATS * p + 5);
        this_particleID(idx)     = recv_buffer_long(NLONGS * p + 0);
    });
    }
    species.set_npart(species.npart() + std::max(permute_vector.extent(0), 
                      allocation_vector.extent(0)) - permute_vector.extent(0));
    /*
    {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      // Print the total number of particles after each pass
      int species_npart = species.npart();
      int global_species_npart = 0;
      // Reduce all local sums into global_sum on rank 0
      MPI_Reduce(&species_npart, &global_species_npart, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      int total_ranks;
      MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);
      for (int allranks=0; allranks<total_ranks; allranks++)
      {
        MPI_Barrier(MPI_COMM_WORLD);
        // Only the root rank (rank 0) will print the global sum

        if (rank == allranks && species.label() == "e+_b") {
          printf("rank %d count: %d \n", rank, species_npart);
          std::cout << "Total" << species.label() << "count : " << global_species_npart << std::endl;
          auto [npart_per_tag_arr_new,
          tag_offset_new]       = species.npart_per_tag();
          for (int i=0; i<species.ntags(); i++)
          {
            std::cout << "Tag: " << i << " count after recv: " << npart_per_tag_arr_new[i] << std::endl;
          }
          // Copy the tag array to host
          auto tag_h = Kokkos::create_mirror_view(species.tag);
          Kokkos::deep_copy(tag_h, species.tag);
          std::cout << "Tag locs after send" << std::endl;
          for (std::size_t i { 0 }; i < species.npart(); i++) {
            if (tag_h(i) != ParticleTag::alive)
              std::cout <<" Tag: " << tag_h(i) << " loc: "<< i << std::endl;
          }
        }
      }
    }
    */
    return;
}

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_MPI_HPP
