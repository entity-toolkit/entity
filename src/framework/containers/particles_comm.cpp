#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "arch/mpi_tags.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "framework/containers/particles.h"

#include "kernels/comm.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

namespace ntt {

  namespace prtls {
    template <typename T>
    void send_recv(array_t<T*>& send_arr,
                   array_t<T*>& recv_arr,
                   int          send_rank,
                   int          recv_rank,
                   npart_t      nsend,
                   npart_t      nrecv,
                   npart_t      offset) {
#if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
      MPI_Sendrecv(send_arr.data(),
                   nsend,
                   mpi::get_type<T>(),
                   send_rank,
                   0,
                   recv_arr.data() + offset,
                   nrecv,
                   mpi::get_type<T>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
#else
      const auto slice = std::make_pair(offset, offset + nrecv);

      auto send_arr_h = Kokkos::create_mirror_view(send_arr);
      auto recv_arr_h = Kokkos::create_mirror_view(
        Kokkos::subview(recv_arr, slice));
      Kokkos::deep_copy(send_arr_h, send_arr);
      MPI_Sendrecv(send_arr_h.data(),
                   nsend,
                   mpi::get_type<T>(),
                   send_rank,
                   0,
                   recv_arr_h.data(),
                   nrecv,
                   mpi::get_type<T>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
      Kokkos::deep_copy(Kokkos::subview(recv_arr, slice), recv_arr_h);
#endif
    }

    void send_recv_count(int      send_rank,
                         int      recv_rank,
                         npart_t  send_count,
                         npart_t& recv_count) {
      if ((send_rank >= 0) && (recv_rank >= 0)) {
        MPI_Sendrecv(&send_count,
                     1,
                     mpi::get_type<npart_t>(),
                     send_rank,
                     0,
                     &recv_count,
                     1,
                     mpi::get_type<npart_t>(),
                     recv_rank,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
      } else if (send_rank >= 0) {
        MPI_Send(&send_count, 1, mpi::get_type<npart_t>(), send_rank, 0, MPI_COMM_WORLD);
      } else if (recv_rank >= 0) {
        MPI_Recv(&recv_count,
                 1,
                 mpi::get_type<npart_t>(),
                 recv_rank,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        raise::Error("ParticleSendRecvCount called with negative ranks", HERE);
      }
    }

    template <typename T>
    void send(array_t<T*>& send_arr, int send_rank, npart_t nsend) {
#if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
      MPI_Send(send_arr.data(), nsend, mpi::get_type<T>(), send_rank, 0, MPI_COMM_WORLD);
#else
      auto send_arr_h = Kokkos::create_mirror_view(send_arr);
      Kokkos::deep_copy(send_arr_h, send_arr);
      MPI_Send(send_arr_h.data(), nsend, mpi::get_type<T>(), send_rank, 0, MPI_COMM_WORLD);
#endif
    }

    template <typename T>
    void recv(array_t<T*>& recv_arr, int recv_rank, npart_t nrecv, npart_t offset) {
#if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
      MPI_Recv(recv_arr.data() + offset,
               nrecv,
               mpi::get_type<T>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
#else
      const auto slice = std::make_pair(offset, offset + nrecv);

      auto recv_arr_h = Kokkos::create_mirror_view(
        Kokkos::subview(recv_arr, slice));
      MPI_Recv(recv_arr_h.data(),
               nrecv,
               mpi::get_type<T>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      Kokkos::deep_copy(Kokkos::subview(recv_arr, slice), recv_arr_h);
#endif
    }

    template <typename T>
    void communicate(array_t<T*>& send_arr,
                     array_t<T*>& recv_arr,
                     int          send_rank,
                     int          recv_rank,
                     npart_t      nsend,
                     npart_t      nrecv,
                     npart_t      offset) {
      if (send_rank >= 0 && recv_rank >= 0) {
        raise::ErrorIf(
          nrecv + offset > recv_arr.extent(0),
          "recv_arr is not large enough to hold the received particles",
          HERE);
        send_recv<T>(send_arr, recv_arr, send_rank, recv_rank, nsend, nrecv, offset);
      } else if (send_rank >= 0) {
        send<T>(send_arr, send_rank, nsend);
      } else if (recv_rank >= 0) {
        raise::ErrorIf(
          nrecv + offset > recv_arr.extent(0),
          "recv_arr is not large enough to hold the received particles",
          HERE);
        recv<T>(recv_arr, recv_rank, nrecv, offset);
      } else {
        raise::Error("CommunicateParticles called with negative ranks", HERE);
      }
    }
  } // namespace prtls

  template <Dimension D, Coord::type C>
  void Particles<D, C>::Communicate(const dir::dirs_t<D>&     dirs_to_comm,
                                    const array_t<int*>&      shifts_in_x1,
                                    const array_t<int*>&      shifts_in_x2,
                                    const array_t<int*>&      shifts_in_x3,
                                    const dir::map_t<D, int>& send_ranks,
                                    const dir::map_t<D, int>& recv_ranks) {
    logger::Checkpoint(fmt::format("Communicating species #%d\n", index()), HERE);

    // at this point particles should already be tagged in the pusher
    auto [npptag_vec, tag_offsets] = NpartsPerTagAndOffsets();
    const auto npart_dead          = npptag_vec[ParticleTag::dead];
    const auto npart_alive         = npptag_vec[ParticleTag::alive];

    // # of particles to receive per each tag (direction)
    std::vector<npart_t> npptag_recv_vec(ntags() - 2, 0);

    // total # of received particles from all directions
    npart_t npart_recv_tot = 0u;

    // loop dir
    for (const auto& direction : dirs_to_comm) {
      // tags corresponding to the direction (both send & recv)
      const auto tag_recv = mpi::PrtlSendTag<D>::dir2tag(-direction);
      const auto tag_send = mpi::PrtlSendTag<D>::dir2tag(direction);

      // get ranks of send/recv meshblocks
      const auto send_rank = send_ranks.at(direction);
      const auto recv_rank = recv_ranks.at(direction);

      // record the # of particles to-be-sent
      const auto nsend = npptag_vec[tag_send];

      // request the # of particles to-be-received ...
      // ... and send the # of particles to-be-sent
      npart_t nrecv = 0;
      prtls::send_recv_count(send_rank, recv_rank, nsend, nrecv);
      npart_recv_tot                += nrecv;
      npptag_recv_vec[tag_recv - 2]  = nrecv;

      raise::ErrorIf((npart() + npart_recv_tot) >= maxnpart(),
                     "Too many particles to receive (cannot fit into maxptl)",
                     HERE);
    }

    array_t<npart_t*> outgoing_indices { "outgoing_indices", npart() - npart_alive };
    // clang-format off
    Kokkos::parallel_for(
      "PrepareOutgoingPrtls",
      rangeActiveParticles(),
      kernel::comm::PrepareOutgoingPrtls_kernel<D>(
          shifts_in_x1, shifts_in_x2, shifts_in_x3,
          outgoing_indices,
          npart(), npart_alive, npart_dead, ntags(),
          i1, i1_prev, 
          i2, i2_prev,
          i3, i3_prev,
          tag, tag_offsets)
    );
    // clang-format on

    // number of arrays of each type to send/recv
    const unsigned short NREALS = 4 + static_cast<unsigned short>(
                                        D == Dim::_2D and C != Coord::Cart);
    const unsigned short NINTS   = 2 * static_cast<unsigned short>(D);
    const unsigned short NPRTLDX = 2 * static_cast<unsigned short>(D);
    const unsigned short NPLDS_R = npld_r();
    const unsigned short NPLDS_I = npld_i();

    // buffers to store recv data
    const auto       npart_recv = std::accumulate(npptag_recv_vec.begin(),
                                            npptag_recv_vec.end(),
                                            static_cast<npart_t>(0));
    array_t<int*>    recv_buff_int { "recv_buff_int", npart_recv * NINTS };
    array_t<real_t*> recv_buff_real { "recv_buff_real", npart_recv * NREALS };
    array_t<prtldx_t*> recv_buff_prtldx { "recv_buff_prtldx", npart_recv * NPRTLDX };
    array_t<real_t*>  recv_buff_pld_r;
    array_t<npart_t*> recv_buff_pld_i;

    if (NPLDS_R > 0) {
      recv_buff_pld_r = array_t<real_t*> { "recv_buff_pld_r", npart_recv * NPLDS_R };
    }
    if (NPLDS_I > 0) {
      recv_buff_pld_i = array_t<npart_t*> { "recv_buff_pld_i",
                                            npart_recv * NPLDS_I };
    }

    auto iteration        = 0;
    auto current_received = 0;

    for (const auto& direction : dirs_to_comm) {
      const auto send_rank     = send_ranks.at(direction);
      const auto recv_rank     = recv_ranks.at(direction);
      const auto tag_send      = mpi::PrtlSendTag<D>::dir2tag(direction);
      const auto tag_recv      = mpi::PrtlSendTag<D>::dir2tag(-direction);
      const auto npart_send_in = npptag_vec[tag_send];
      const auto npart_recv_in = npptag_recv_vec[tag_recv - 2];
      if (send_rank < 0 and recv_rank < 0) {
        continue;
      }
      array_t<int*> send_buff_int { "send_buff_int", npart_send_in * NINTS };
      array_t<real_t*> send_buff_real { "send_buff_real", npart_send_in * NREALS };
      array_t<prtldx_t*> send_buff_prtldx { "send_buff_prtldx",
                                            npart_send_in * NPRTLDX };
      array_t<real_t*>   send_buff_pld_r;
      array_t<npart_t*>  send_buff_pld_i;
      if (NPLDS_R > 0) {
        send_buff_pld_r = array_t<real_t*> { "send_buff_pld_r",
                                             npart_send_in * NPLDS_R };
      }
      if (NPLDS_I > 0) {
        send_buff_pld_i = array_t<npart_t*> { "send_buff_pld_i",
                                              npart_send_in * NPLDS_I };
      }

      auto tag_offsets_h = Kokkos::create_mirror_view(tag_offsets);
      Kokkos::deep_copy(tag_offsets_h, tag_offsets);

      npart_t idx_offset = npart_dead;
      if (tag_send > 2) {
        idx_offset += tag_offsets_h(tag_send - 3);
      }
      // clang-format off
      Kokkos::parallel_for(
        "PopulatePrtlSendBuffer",
        npart_send_in,
        kernel::comm::PopulatePrtlSendBuffer_kernel<D, C>(
          send_buff_int, send_buff_real, send_buff_prtldx, send_buff_pld_r, send_buff_pld_i,
          NINTS, NREALS, NPRTLDX, NPLDS_R, NPLDS_I, idx_offset,
          i1, i1_prev, dx1, dx1_prev,
          i2, i2_prev, dx2, dx2_prev,
          i3, i3_prev, dx3, dx3_prev,
          ux1, ux2, ux3, 
          weight, phi, pld_r, pld_i, tag,
          outgoing_indices)
      );
      // clang-format on

      const auto recv_offset_int    = current_received * NINTS;
      const auto recv_offset_real   = current_received * NREALS;
      const auto recv_offset_prtldx = current_received * NPRTLDX;
      const auto recv_offset_pld_r  = current_received * NPLDS_R;
      const auto recv_offset_pld_i  = current_received * NPLDS_I;

      prtls::communicate<int>(send_buff_int,
                              recv_buff_int,
                              send_rank,
                              recv_rank,
                              npart_send_in * NINTS,
                              npart_recv_in * NINTS,
                              recv_offset_int);
      prtls::communicate<real_t>(send_buff_real,
                                 recv_buff_real,
                                 send_rank,
                                 recv_rank,
                                 npart_send_in * NREALS,
                                 npart_recv_in * NREALS,
                                 recv_offset_real);
      prtls::communicate<prtldx_t>(send_buff_prtldx,
                                   recv_buff_prtldx,
                                   send_rank,
                                   recv_rank,
                                   npart_send_in * NPRTLDX,
                                   npart_recv_in * NPRTLDX,
                                   recv_offset_prtldx);
      if (NPLDS_R > 0) {
        prtls::communicate<real_t>(send_buff_pld_r,
                                   recv_buff_pld_r,
                                   send_rank,
                                   recv_rank,
                                   npart_send_in * NPLDS_R,
                                   npart_recv_in * NPLDS_R,
                                   recv_offset_pld_r);
      }
      if (NPLDS_I > 0) {
        prtls::communicate<npart_t>(send_buff_pld_i,
                                    recv_buff_pld_i,
                                    send_rank,
                                    recv_rank,
                                    npart_send_in * NPLDS_I,
                                    npart_recv_in * NPLDS_I,
                                    recv_offset_pld_i);
      }
      current_received += npart_recv_in;
      iteration++;

    } // end direction loop

    // clang-format off
    Kokkos::parallel_for(
      "PopulateFromRecvBuffer",
      npart_recv,
      kernel::comm::ExtractReceivedPrtls_kernel<D, C>(
            recv_buff_int, recv_buff_real, recv_buff_prtldx, recv_buff_pld_r, recv_buff_pld_i,
            NINTS, NREALS, NPRTLDX, NPLDS_R, NPLDS_I,
            npart(),
            i1, i1_prev, dx1, dx1_prev,
            i2, i2_prev, dx2, dx2_prev,
            i3, i3_prev, dx3, dx3_prev,
            ux1, ux2, ux3,
            weight, phi, pld_r, pld_i, tag,
            outgoing_indices)
    );
    // clang-format on

    const auto npart_holes = outgoing_indices.extent(0);
    if (npart_recv > npart_holes) {
      set_npart(npart() + npart_recv - npart_holes);
    }
    set_unsorted();
  }

#define PARTICLES_COMM(D, C)                                                   \
  template void Particles<D, C>::Communicate(const dir::dirs_t<D>&,            \
                                             const array_t<int*>&,             \
                                             const array_t<int*>&,             \
                                             const array_t<int*>&,             \
                                             const dir::map_t<D, int>&,        \
                                             const dir::map_t<D, int>&);

  PARTICLES_COMM(Dim::_1D, Coord::Cart)
  PARTICLES_COMM(Dim::_2D, Coord::Cart)
  PARTICLES_COMM(Dim::_3D, Coord::Cart)
  PARTICLES_COMM(Dim::_2D, Coord::Sph)
  PARTICLES_COMM(Dim::_2D, Coord::Qsph)
  PARTICLES_COMM(Dim::_3D, Coord::Sph)
  PARTICLES_COMM(Dim::_3D, Coord::Qsph)
#undef PARTICLES_COMM

} // namespace ntt
