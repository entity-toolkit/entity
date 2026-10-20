/**
 * @file framework/domain/comm/utils.hpp
 * @brief Utility functions for inter-domain communication
 * @implements
 *  - ntt::GetSendRecvRanks<> -> std::pair<address_t, address_t>
 *  - ntt::GetSendRecvParams<> -> std::pair<comm_params_t, comm_params_t>
 *  - ntt::HaloDirections<> -> std::vector<comm::HaloDirection> [if MPI_ENABLED]
 * @namespaces:
 *  - ntt::
 * @macros:
 *  - MPI_ENABLED
 *  - OUTPUT_ENABLED
 */

#ifndef FRAMEWORK_DOMAIN_COMM_UTILS_HPP
#define FRAMEWORK_DOMAIN_COMM_UTILS_HPP

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include "framework/domain/comm/halo_exchange.hpp"
#endif

#include <Kokkos_Core.hpp>

#include <utility>
#include <vector>

namespace ntt {

  using address_t     = std::pair<unsigned int, int>;
  using comm_params_t = std::pair<address_t, std::vector<cell_range_t>>;

  template <SimEngine::type S, MetricClass M>
  auto GetSendRecvRanks(const Metadomain<S, M>* const   metadomain,
                        Domain<S, M>&                   domain,
                        const dir::direction_t<M::Dim>& direction)
    -> std::pair<address_t, address_t> {
    const Domain<S, M>* send_to_nghbr_ptr   = nullptr;
    const Domain<S, M>* recv_from_nghbr_ptr = nullptr;
    // set pointers to the correct send/recv domains
    // can coincide with the current domain if periodic
    if (domain.mesh.flds_bc_in(direction) == FldsBC::PERIODIC) {
      // sending / receiving from itself
      raise::ErrorIf(
        domain.neighbor_idx_in(direction) != domain.index(),
        fmt::format(
          "Periodic boundaries in `%s` imply communication within the "
          "same domain, but %u != %u",
          direction.to_string().c_str(),
          domain.neighbor_idx_in(direction),
          domain.index()),
        HERE);
      raise::ErrorIf(
        domain.mesh.flds_bc_in(-direction) != FldsBC::PERIODIC,
        "Periodic boundary conditions must be set in both directions",
        HERE);
      send_to_nghbr_ptr   = &domain;
      recv_from_nghbr_ptr = &domain;
    } else if (domain.mesh.flds_bc_in(direction) == FldsBC::SYNC) {
      // sending to other domain
      raise::ErrorIf(
        domain.neighbor_idx_in(direction) == domain.index(),
        "Sync boundaries imply communication between separate domains",
        HERE);
      send_to_nghbr_ptr = metadomain->subdomain_ptr(
        domain.neighbor_idx_in(direction));
      if (domain.mesh.flds_bc_in(-direction) == FldsBC::SYNC) {
        // receiving from other domain
        raise::ErrorIf(
          domain.neighbor_idx_in(-direction) == domain.index(),
          "Sync boundaries imply communication between separate domains",
          HERE);
        recv_from_nghbr_ptr = metadomain->subdomain_ptr(
          domain.neighbor_idx_in(-direction));
      }
    } else if (domain.mesh.flds_bc_in(-direction) == FldsBC::SYNC) {
      // only receiving from other domain
      raise::ErrorIf(
        domain.neighbor_idx_in(-direction) == domain.index(),
        "Sync boundaries imply communication between separate domains",
        HERE);
      recv_from_nghbr_ptr = metadomain->subdomain_ptr(
        domain.neighbor_idx_in(-direction));
    } else {
      // no communication necessary
      return {
        { 0, -1 },
        { 0, -1 }
      };
    }
#if defined(MPI_ENABLED)
    const auto send_rank = (send_to_nghbr_ptr != nullptr)
                             ? send_to_nghbr_ptr->mpi_rank()
                             : -1;
    const auto recv_rank = (recv_from_nghbr_ptr != nullptr)
                             ? recv_from_nghbr_ptr->mpi_rank()
                             : -1;
#else
    const auto send_rank = (send_to_nghbr_ptr != nullptr) ? 0 : -1;
    const auto recv_rank = (recv_from_nghbr_ptr != nullptr) ? 0 : -1;
#endif
    const auto send_ind = (send_to_nghbr_ptr != nullptr)
                            ? send_to_nghbr_ptr->index()
                            : 0;
    const auto recv_ind = (recv_from_nghbr_ptr != nullptr)
                            ? recv_from_nghbr_ptr->index()
                            : 0;
    (void)send_rank;
    (void)recv_rank;
    return {
      { send_ind, send_rank },
      { recv_ind, recv_rank }
    };
  }

  template <SimEngine::type S, MetricClass M>
  auto GetSendRecvParams(const Metadomain<S, M>* const metadomain,
                         Domain<S, M>&                 domain,
                         dir::direction_t<M::Dim>      direction,
                         bool                          synchronize)
    -> std::pair<comm_params_t, comm_params_t> {
    const auto [send_indrank,
                recv_indrank] = GetSendRecvRanks(metadomain, domain, direction);
    const auto [send_ind, send_rank] = send_indrank;
    const auto [recv_ind, recv_rank] = recv_indrank;
    const auto is_sending            = (send_rank >= 0);
    const auto is_receiving          = (recv_rank >= 0);
    if (not(is_sending or is_receiving)) {
      return {
        { { 0, -1 }, {} },
        { { 0, -1 }, {} }
      };
    }
    auto     send_slice   = std::vector<cell_range_t> {};
    auto     recv_slice   = std::vector<cell_range_t> {};
    const in components[] = { in::x1, in::x2, in::x3 };
    // find the field components and indices to be sent/received
    for (auto d { 0u }; d < direction.size(); ++d) {
      const auto c   = components[d];
      const auto dir = direction[d];
      if (not synchronize) {
        // recv to: ghost zones
        // send from: active zones
        if (is_sending) {
          if (dir == 0) {
            send_slice.emplace_back(domain.mesh.i_min(c), domain.mesh.i_max(c));
          } else if (dir == 1) {
            send_slice.emplace_back(domain.mesh.i_max(c) - N_GHOSTS,
                                    domain.mesh.i_max(c));
          } else {
            send_slice.emplace_back(domain.mesh.i_min(c),
                                    domain.mesh.i_min(c) + N_GHOSTS);
          }
        }
        if (is_receiving) {
          if (-dir == 0) {
            recv_slice.emplace_back(domain.mesh.i_min(c), domain.mesh.i_max(c));
          } else if (-dir == 1) {
            recv_slice.emplace_back(domain.mesh.i_max(c),
                                    domain.mesh.i_max(c) + N_GHOSTS);
          } else {
            recv_slice.emplace_back(domain.mesh.i_min(c) - N_GHOSTS,
                                    domain.mesh.i_min(c));
          }
        }
      } else {
        // recv to: active + ghost zones
        // send from: active + ghost zones
        if (is_sending) {
          if (dir == 0) {
            send_slice.emplace_back(domain.mesh.i_min(c) - N_GHOSTS,
                                    domain.mesh.i_max(c) + N_GHOSTS);
          } else if (dir == 1) {
            send_slice.emplace_back(domain.mesh.i_max(c) - N_GHOSTS,
                                    domain.mesh.i_max(c) + N_GHOSTS);
          } else {
            send_slice.emplace_back(domain.mesh.i_min(c) - N_GHOSTS,
                                    domain.mesh.i_min(c) + N_GHOSTS);
          }
        }
        if (is_receiving) {
          if (-dir == 0) {
            recv_slice.emplace_back(domain.mesh.i_min(c) - N_GHOSTS,
                                    domain.mesh.i_max(c) + N_GHOSTS);
          } else if (-dir == 1) {
            recv_slice.emplace_back(domain.mesh.i_max(c) - N_GHOSTS,
                                    domain.mesh.i_max(c) + N_GHOSTS);
          } else {
            recv_slice.emplace_back(domain.mesh.i_min(c) - N_GHOSTS,
                                    domain.mesh.i_min(c) + N_GHOSTS);
          }
        }
      }
    }

    return {
      { { send_ind, send_rank }, send_slice },
      { { recv_ind, recv_rank }, recv_slice },
    };
  }

#if defined(MPI_ENABLED)
  /**
   * @brief Neighbor directions of `domain` that take part in a halo exchange,
   *        in the order of dir::Directions<D>::all, with the direction index
   *        as the message tag. A direction whose send and receive partner is
   *        the domain itself (periodic self-wrap) is marked local.
   * @param synchronize false: active -> ghost slices (fill);
   *        true: active + ghost slices (additive deposit remap)
   */
  template <SimEngine::type S, MetricClass M>
  auto HaloDirections(const Metadomain<S, M>* const metadomain,
                      Domain<S, M>&                 domain,
                      int                           mpi_rank,
                      bool synchronize) -> std::vector<comm::HaloDirection> {
    std::vector<comm::HaloDirection> dirs;
    int                              tag = 0;
    for (const auto& direction : dir::Directions<M::Dim>::all) {
      const auto [send_params,
                  recv_params] = GetSendRecvParams(metadomain,
                                                   domain,
                                                   direction,
                                                   synchronize);
      const auto [send_indrank, send_slice] = send_params;
      const auto [recv_indrank, recv_slice] = recv_params;
      const auto [send_ind, send_rank]      = send_indrank;
      const auto [recv_ind, recv_rank]      = recv_indrank;
      if (send_rank >= 0 or recv_rank >= 0) {
        raise::ErrorIf(
          (send_rank == mpi_rank and send_ind != domain.index()) or
            (recv_rank == mpi_rank and recv_ind != domain.index()),
          "Multiple-domain single-rank communication not yet implemented",
          HERE);
        comm::HaloDirection dr;
        dr.tag       = tag;
        dr.local     = (send_ind == domain.index()) and
                   (recv_ind == domain.index());
        dr.send_rank = send_rank;
        dr.recv_rank = recv_rank;
        raise::ErrorIf(dr.local and (send_rank < 0 or recv_rank < 0),
                       "HaloDirections: one-sided self communication",
                       HERE);
        dr.send_slice = send_slice;
        dr.recv_slice = recv_slice;
        dirs.push_back(std::move(dr));
      }
      ++tag;
    }
    return dirs;
  }
#endif

} // namespace ntt

#endif // FRAMEWORK_DOMAIN_COMM_UTILS_HPP
