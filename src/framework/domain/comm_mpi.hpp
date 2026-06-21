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

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "utils/error.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <vector>

namespace comm {
  using namespace ntt;

  namespace flds {
    template <unsigned short D>
    void send_recv(ndarray_t<D>& send_arr,
                   ndarray_t<D>& recv_arr,
                   int           send_rank,
                   int           recv_rank,
                   ncells_t      nsend,
                   ncells_t      nrecv) {
#if defined(DEVICE_ENABLED)
      // guard for Intel GPUs.
      // Should be a null-operation for other architectures.
      Kokkos::fence();
#endif
#if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
      MPI_Sendrecv(send_arr.data(),
                   nsend,
                   mpi::get_type<real_t>(),
                   send_rank,
                   0,
                   recv_arr.data(),
                   nrecv,
                   mpi::get_type<real_t>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
#else
      auto send_arr_h = Kokkos::create_mirror_view(send_arr);
      auto recv_arr_h = Kokkos::create_mirror_view(recv_arr);
      Kokkos::deep_copy(send_arr_h, send_arr);
      MPI_Sendrecv(send_arr_h.data(),
                   nsend,
                   mpi::get_type<real_t>(),
                   send_rank,
                   0,
                   recv_arr_h.data(),
                   nrecv,
                   mpi::get_type<real_t>(),
                   recv_rank,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
      Kokkos::deep_copy(recv_arr, recv_arr_h);
#endif
    }

    template <unsigned short D>
    void send(ndarray_t<D>& send_arr, int send_rank, ncells_t nsend) {
#if defined(DEVICE_ENABLED)
      // guard for Intel GPUs.
      // Should be a null-operation for other architectures.
      Kokkos::fence();
#endif
#if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
      MPI_Send(send_arr.data(), nsend, mpi::get_type<real_t>(), send_rank, 0, MPI_COMM_WORLD);
#else
      auto send_arr_h = Kokkos::create_mirror_view(send_arr);
      Kokkos::deep_copy(send_arr_h, send_arr);
      MPI_Send(send_arr_h.data(),
               nsend,
               mpi::get_type<real_t>(),
               send_rank,
               0,
               MPI_COMM_WORLD);
#endif
    }

    template <unsigned short D>
    void recv(ndarray_t<D>& recv_arr, int recv_rank, ncells_t nrecv) {
#if defined(DEVICE_ENABLED)
      // guard for Intel GPUs.
      // Should be a null-operation for other architectures.
      Kokkos::fence();
#endif
#if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
      MPI_Recv(recv_arr.data(),
               nrecv,
               mpi::get_type<real_t>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
#else
      auto recv_arr_h = Kokkos::create_mirror_view(recv_arr);
      MPI_Recv(recv_arr_h.data(),
               nrecv,
               mpi::get_type<real_t>(),
               recv_rank,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      Kokkos::deep_copy(recv_arr, recv_arr_h);
#endif
    }

    template <unsigned short D>
    void communicate(ndarray_t<D>& send_arr,
                     ndarray_t<D>& recv_arr,
                     int           send_rank,
                     int           recv_rank,
                     ncells_t      nsend,
                     ncells_t      nrecv) {
      if (send_rank >= 0 and recv_rank >= 0 and nsend > 0 and nrecv > 0) {
        send_recv<D>(send_arr, recv_arr, send_rank, recv_rank, nsend, nrecv);
      } else if (send_rank >= 0 and nsend > 0) {
        send<D>(send_arr, send_rank, nsend);
      } else if (recv_rank >= 0 and nrecv > 0) {
        recv<D>(recv_arr, recv_rank, nrecv);
      }
    }

  } // namespace flds

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
      ncells_t nsend { comps.second - comps.first },
        nrecv { comps.second - comps.first };
      ndarray_t<static_cast<dim_t>(D) + 1> send_fld, recv_fld;

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

      flds::communicate<static_cast<unsigned short>(D) + 1>(send_fld,
                                                            recv_fld,
                                                            send_rank,
                                                            recv_rank,
                                                            nsend,
                                                            nrecv);

      if (recv_rank >= 0) {

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
              Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
                { recv_slice[0].first, comps.first },
                { recv_slice[0].second, comps.second }),
              Lambda(cellidx_t i1, cellidx_t ci) {
                fld_buff(i1, ci) += recv_fld(i1 - offset_x1, ci - offset_c);
              });
          } else if constexpr (D == Dim::_2D) {
            const auto offset_x1 = recv_slice[0].first;
            const auto offset_x2 = recv_slice[1].first;
            const auto offset_c  = comps.first;
            Kokkos::parallel_for(
              "CommunicateField-extract",
              Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>(
                { recv_slice[0].first, recv_slice[1].first, comps.first },
                { recv_slice[0].second, recv_slice[1].second, comps.second }),
              Lambda(cellidx_t i1, cellidx_t i2, cellidx_t ci) {
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

  // ---- batched, non-blocking field halo exchange ------------------------ //

  // Per-direction parameters, collected once by the caller (GetSendRecvParams)
  // and reused for every field communicated in the same call.
  template <Dimension D>
  struct FieldCommDir {
    int                       send_rank { -1 }, recv_rank { -1 };
    unsigned int              send_ind { 0 }, recv_ind { 0 };
    std::vector<cell_range_t> send_slice {}, recv_slice {};
    int                       tag { 0 };
  };

  /**
   * @brief Non-additive (ghost-overwrite) halo exchange of one field across
   *        all directions at once.
   * @details Packs every send, posts all `MPI_Irecv` then all `MPI_Isend`
   *          (unique per-direction tags), a single `MPI_Waitall`, then unpacks
   *          — overlapping the round-trips instead of serializing one blocking
   *          `MPI_Sendrecv` per direction. Self-communication (periodic single
   *          domain) is a local copy. Slicing/packing mirrors
   *          `comm::CommunicateField`. The tag pairs A's send in a direction
   *          with B's receive in the same direction (the matching neighbor),
   *          so it is unique per (rank-pair, direction).
   */
  template <Dimension D, int N>
  inline void CommunicateFieldBatched(unsigned int                        my_idx,
                                      ndfield_t<D, N>&                    fld,
                                      const std::vector<FieldCommDir<D>>& dirs,
                                      const cell_range_t&                 comps) {
    static constexpr unsigned short Dp1 = static_cast<unsigned short>(D) + 1;
    using buf_t                         = ndarray_t<Dp1>;
    const ncells_t ncomp       = comps.second - comps.first;
    const auto     ndirs       = dirs.size();

    const auto is_self = [&](const FieldCommDir<D>& dd) {
      return (dd.send_ind == my_idx) && (dd.recv_ind == my_idx);
    };
    const auto ext = [](const std::vector<cell_range_t>& sl, int d) -> ncells_t {
      return sl[d].second - sl[d].first;
    };
    const auto make_buf = [&](const char*                      lbl,
                              const std::vector<cell_range_t>& sl) -> buf_t {
      if constexpr (D == Dim::_1D) {
        return buf_t(lbl, ext(sl, 0), ncomp);
      } else if constexpr (D == Dim::_2D) {
        return buf_t(lbl, ext(sl, 0), ext(sl, 1), ncomp);
      } else {
        return buf_t(lbl, ext(sl, 0), ext(sl, 1), ext(sl, 2), ncomp);
      }
    };
    const auto count = [](const buf_t& b) -> int {
      ncells_t n = 1;
      for (auto d { 0 }; d < static_cast<int>(Dp1); ++d) {
        n *= static_cast<ncells_t>(b.extent(d));
      }
      return static_cast<int>(n);
    };
    const auto pack = [&](buf_t& b, const std::vector<cell_range_t>& sl) {
      if constexpr (D == Dim::_1D) {
        Kokkos::deep_copy(b, Kokkos::subview(fld, sl[0], comps));
      } else if constexpr (D == Dim::_2D) {
        Kokkos::deep_copy(b, Kokkos::subview(fld, sl[0], sl[1], comps));
      } else {
        Kokkos::deep_copy(b, Kokkos::subview(fld, sl[0], sl[1], sl[2], comps));
      }
    };
    const auto unpack = [&](const buf_t& b, const std::vector<cell_range_t>& sl) {
      if constexpr (D == Dim::_1D) {
        Kokkos::deep_copy(Kokkos::subview(fld, sl[0], comps), b);
      } else if constexpr (D == Dim::_2D) {
        Kokkos::deep_copy(Kokkos::subview(fld, sl[0], sl[1], comps), b);
      } else {
        Kokkos::deep_copy(Kokkos::subview(fld, sl[0], sl[1], sl[2], comps), b);
      }
    };
    const auto self_copy = [&](const std::vector<cell_range_t>& ssl,
                               const std::vector<cell_range_t>& rsl) {
      if constexpr (D == Dim::_1D) {
        Kokkos::deep_copy(Kokkos::subview(fld, rsl[0], comps),
                          Kokkos::subview(fld, ssl[0], comps));
      } else if constexpr (D == Dim::_2D) {
        Kokkos::deep_copy(Kokkos::subview(fld, rsl[0], rsl[1], comps),
                          Kokkos::subview(fld, ssl[0], ssl[1], comps));
      } else {
        Kokkos::deep_copy(Kokkos::subview(fld, rsl[0], rsl[1], rsl[2], comps),
                          Kokkos::subview(fld, ssl[0], ssl[1], ssl[2], comps));
      }
    };

    std::vector<buf_t> send_buf(ndirs), recv_buf(ndirs);
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
    using hbuf_t = typename buf_t::host_mirror_type;
    std::vector<hbuf_t> send_h(ndirs), recv_h(ndirs);
#endif

    // phase 1: self copies, pack sends, allocate recvs
    for (auto i { 0u }; i < ndirs; ++i) {
      const auto& dd = dirs[i];
      if (is_self(dd)) {
        self_copy(dd.send_slice, dd.recv_slice);
        continue;
      }
      if (dd.recv_rank >= 0) {
        recv_buf[i] = make_buf("recv_fld", dd.recv_slice);
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
        recv_h[i] = Kokkos::create_mirror_view(recv_buf[i]);
#endif
      }
      if (dd.send_rank >= 0) {
        send_buf[i] = make_buf("send_fld", dd.send_slice);
        pack(send_buf[i], dd.send_slice);
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
        send_h[i] = Kokkos::create_mirror_view(send_buf[i]);
        Kokkos::deep_copy(send_h[i], send_buf[i]);
#endif
      }
    }
    // drain packs (and device->host staging) before MPI reads the buffers
    Kokkos::fence("CommunicateFieldBatched: pre-MPI");

    // phase 2: post all recvs, then all sends
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * ndirs);
    for (auto i { 0u }; i < ndirs; ++i) {
      const auto& dd = dirs[i];
      if (is_self(dd) || dd.recv_rank < 0) {
        continue;
      }
      reqs.emplace_back();
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
      MPI_Irecv(recv_h[i].data(), count(recv_buf[i]), mpi::get_type<real_t>(),
                dd.recv_rank, dd.tag, MPI_COMM_WORLD, &reqs.back());
#else
      MPI_Irecv(recv_buf[i].data(), count(recv_buf[i]), mpi::get_type<real_t>(),
                dd.recv_rank, dd.tag, MPI_COMM_WORLD, &reqs.back());
#endif
    }
    for (auto i { 0u }; i < ndirs; ++i) {
      const auto& dd = dirs[i];
      if (is_self(dd) || dd.send_rank < 0) {
        continue;
      }
      reqs.emplace_back();
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
      MPI_Isend(send_h[i].data(), count(send_buf[i]), mpi::get_type<real_t>(),
                dd.send_rank, dd.tag, MPI_COMM_WORLD, &reqs.back());
#else
      MPI_Isend(send_buf[i].data(), count(send_buf[i]), mpi::get_type<real_t>(),
                dd.send_rank, dd.tag, MPI_COMM_WORLD, &reqs.back());
#endif
    }

    // phase 3: complete all transfers
    if (not reqs.empty()) {
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }

    // phase 4: unpack received ghosts
    for (auto i { 0u }; i < ndirs; ++i) {
      const auto& dd = dirs[i];
      if (is_self(dd) || dd.recv_rank < 0) {
        continue;
      }
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
      Kokkos::deep_copy(recv_buf[i], recv_h[i]);
#endif
      unpack(recv_buf[i], dd.recv_slice);
    }
  }

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_MPI_HPP
