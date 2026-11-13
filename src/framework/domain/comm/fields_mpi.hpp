/**
 * @file framework/domain/comm/fields_mpi.hpp
 * @brief MPI communication routines for fields
 * @implements
 *   - comm::PendingComm
 *   - comm::CommunicateField_Post<> -> comm::PendingComm
 *   - comm::CommunicateField<> -> void
 * @namespaces:
 *   - comm::
 * @note This should only be included if the MPI_ENABLED flag is set
 * @note CommunicateField_Post() POSTS a non-blocking send/recv and returns
 *       immediately; it does not touch the destination field. The caller
 *       must MPI_Waitall() every PendingComm::requests collected across a
 *       batch of these calls (e.g. one call per direction x per field array
 *       within one halo-exchange round), THEN invoke each returned
 *       PendingComm::finalize(), before relying on any of the destination
 *       fields.
 */

#ifndef FRAMEWORK_DOMAIN_COMM_FIELDS_MPI_HPP
#define FRAMEWORK_DOMAIN_COMM_FIELDS_MPI_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "utils/error.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <functional>
#include <vector>

namespace comm {
  using namespace ntt;

  /**
   * @brief A single CommunicateField_Post() call's outstanding MPI
   *        request(s) (0, 1, or 2 of them) plus the deferred host-side work
   *        (copying/accumulating a completed receive into the destination
   *        field) that must run only once those requests are known to have
   *        completed.
   * @note `finalize` also keeps the temporary send/recv device (or host
   *       mirror) buffers alive by capturing them by value, since
   *       MPI_Isend/Irecv reference that memory asynchronously until the
   *       matching request completes.
   */
  struct PendingComm {
    std::vector<MPI_Request> requests;
    std::function<void()>    finalize;
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

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    raise::ErrorIf(
      (send_rank == rank && send_idx != idx) ||
        (recv_rank == rank && recv_idx != idx),
      "Multiple-domain single-rank communication not yet implemented",
      HERE);

    PendingComm pending;

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
      return pending; // no MPI requests, nothing to finalize
    }

    // cross-rank case: extract into temporaries, POST non-blocking
    // send/recv, and defer the copy/accumulate-back into `finalize`
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

#if defined(DEVICE_ENABLED)
    // guard for Intel GPUs. Should be a null-operation for other
    // architectures. Ensures the deep_copy(s) above have actually landed in
    // send_fld before MPI (potentially GPU-aware) touches it.
    Kokkos::fence();
#endif

#if !defined(DEVICE_ENABLED) || defined(GPU_AWARE_MPI)
    if (send_rank >= 0 and nsend > 0) {
      MPI_Request req;
      MPI_Isend(send_fld.data(),
                nsend,
                mpi::get_type<real_t>(),
                send_rank,
                0,
                MPI_COMM_WORLD,
                &req);
      pending.requests.push_back(req);
    }
    if (recv_rank >= 0 and nrecv > 0) {
      MPI_Request req;
      MPI_Irecv(recv_fld.data(),
                nrecv,
                mpi::get_type<real_t>(),
                recv_rank,
                0,
                MPI_COMM_WORLD,
                &req);
      pending.requests.push_back(req);
    }

    pending.finalize = [&fld,
                        &fld_buff,
                        send_fld,
                        recv_fld,
                        recv_slice,
                        comps,
                        recv_rank,
                        additive]() {
      if (recv_rank < 0) {
        return; // send-only: nothing to copy back, buffers just needed to
                // stay alive (via this closure's captures) until now
      }
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
    };
#else
    // non-GPU-aware MPI: stage through host mirrors. Isend/Irecv are posted
    // on the host buffers; the device-side recv_fld is only populated (via
    // deep_copy from recv_fld_h) inside finalize(), i.e. after Waitall.
    auto send_fld_h = Kokkos::create_mirror_view(send_fld);
    auto recv_fld_h = Kokkos::create_mirror_view(recv_fld);
    if (send_rank >= 0 and nsend > 0) {
      Kokkos::deep_copy(send_fld_h, send_fld);
      MPI_Request req;
      MPI_Isend(send_fld_h.data(),
                nsend,
                mpi::get_type<real_t>(),
                send_rank,
                0,
                MPI_COMM_WORLD,
                &req);
      pending.requests.push_back(req);
    }
    if (recv_rank >= 0 and nrecv > 0) {
      MPI_Request req;
      MPI_Irecv(recv_fld_h.data(),
                nrecv,
                mpi::get_type<real_t>(),
                recv_rank,
                0,
                MPI_COMM_WORLD,
                &req);
      pending.requests.push_back(req);
    }

    pending.finalize = [&fld,
                        &fld_buff,
                        send_fld_h,
                        recv_fld,
                        recv_fld_h,
                        recv_slice,
                        comps,
                        recv_rank,
                        additive]() mutable {
      if (recv_rank < 0) {
        return;
      }
      Kokkos::deep_copy(recv_fld, recv_fld_h);
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
    };
#endif

    return pending;
  }

  /**
   * @brief Blocking convenience wrapper around CommunicateField_Post():
   *        posts the exchange, waits for it, and runs its finalize before
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
    if (not pending.requests.empty()) {
      MPI_Waitall(static_cast<int>(pending.requests.size()),
                  pending.requests.data(),
                  MPI_STATUSES_IGNORE);
    }
    if (pending.finalize) {
      pending.finalize();
    }
  }

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_FIELDS_MPI_HPP
