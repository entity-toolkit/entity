#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/log.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/specialization_registry.h"

#include "framework/domain/comm/utils.hpp"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

#include <Kokkos_Core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::CommunicateParticles(Domain<S, M>& domain) const {
#if defined(MPI_ENABLED)
    logger::Checkpoint("Communicating particles\n", HERE);
    for (auto& species : domain.species) {
      const auto ntags = species.ntags();

      // coordinate shifts per each direction
      array_t<int*> shifts_in_x1 { "shifts_in_x1", ntags - 2 };
      array_t<int*> shifts_in_x2 { "shifts_in_x2", ntags - 2 };
      array_t<int*> shifts_in_x3 { "shifts_in_x3", ntags - 2 };
      auto          shifts_in_x1_h = Kokkos::create_mirror_view(shifts_in_x1);
      auto          shifts_in_x2_h = Kokkos::create_mirror_view(shifts_in_x2);
      auto          shifts_in_x3_h = Kokkos::create_mirror_view(shifts_in_x3);

      // all directions requiring communication
      dir::dirs_t<D> dirs_to_comm;

      // ranks & indices of meshblock to send/recv from
      dir::map_t<D, int> send_ranks;
      dir::map_t<D, int> recv_ranks;

      for (const auto& direction : dir::Directions<D>::all) {
        // tags corresponding to the direction (both send & recv)
        const auto tag_send = mpi::PrtlSendTag<D>::dir2tag(direction);

        // get indices & ranks of send/recv meshblocks
        const auto [send_params,
                    recv_params] = GetSendRecvRanks(this, domain, direction);
        const auto [send_ind, send_rank] = send_params;
        const auto [recv_ind, recv_rank] = recv_params;

        // skip if no communication is necessary
        const auto is_sending   = (send_rank >= 0);
        const auto is_receiving = (recv_rank >= 0);
        if (not is_sending and not is_receiving) {
          continue;
        }
        dirs_to_comm.push_back(direction);
        send_ranks[direction] = send_rank;
        recv_ranks[direction] = recv_rank;

        // if sending, record displacements to apply before
        // ... tag_send - 2: because we only shift tags > 2 (i.e. no dead/alive)
        if (is_sending) {
          if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
            if (direction[0] == -1) {
              // sending backwards in x1 (add sx1 of target meshblock)
              shifts_in_x1_h(tag_send - 2) = subdomain(send_ind).mesh.n_active(
                in::x1);
            } else if (direction[0] == 1) {
              // sending forward in x1 (subtract sx1 of source meshblock)
              shifts_in_x1_h(tag_send - 2) = -domain.mesh.n_active(in::x1);
            }
          }
          if constexpr (D == Dim::_2D || D == Dim::_3D) {
            if (direction[1] == -1) {
              shifts_in_x2_h(tag_send - 2) = subdomain(send_ind).mesh.n_active(
                in::x2);
            } else if (direction[1] == 1) {
              shifts_in_x2_h(tag_send - 2) = -domain.mesh.n_active(in::x2);
            }
          }
          if constexpr (D == Dim::_3D) {
            if (direction[2] == -1) {
              shifts_in_x3_h(tag_send - 2) = subdomain(send_ind).mesh.n_active(
                in::x3);
            } else if (direction[2] == 1) {
              shifts_in_x3_h(tag_send - 2) = -domain.mesh.n_active(in::x3);
            }
          }
        }
      } // end directions loop

      Kokkos::deep_copy(shifts_in_x1, shifts_in_x1_h);
      Kokkos::deep_copy(shifts_in_x2, shifts_in_x2_h);
      Kokkos::deep_copy(shifts_in_x3, shifts_in_x3_h);

      species.Communicate(dirs_to_comm,
                          shifts_in_x1,
                          shifts_in_x2,
                          shifts_in_x3,
                          send_ranks,
                          recv_ranks);

    } // end species loop
#else
    (void)domain;
#endif
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_COMM(S, M, D)                                               \
  template void Metadomain<S, M<D>>::CommunicateParticles(Domain<S, M<D>>&) const;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_COMM)
#undef METADOMAIN_COMM
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
