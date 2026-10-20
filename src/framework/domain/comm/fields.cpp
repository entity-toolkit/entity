#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "traits/engine.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/specialization_registry.h"

#include "framework/domain/comm/utils.hpp"

#if defined(MPI_ENABLED)
  #include "framework/domain/comm/fields_mpi.hpp"
#else
  #include "framework/domain/comm/fields_nompi.hpp"
#endif

#include <Kokkos_Core.hpp>

#include <string>
#include <utility>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::CommunicateFields(Domain<S, M>& domain,
                                           CommTags      tags) const {
    const auto comm_em   = (tags & Comm::EM_012) or (tags & Comm::EM_345);
    const auto comm_em0  = (tags & Comm::EM0_012) or (tags & Comm::EM0_345);
    const auto comm_cur  = (tags & Comm::CUR);
    const auto comm_cur0 = (tags & Comm::CUR0);
    const auto comm_aux  = (tags & Comm::AUX_012) or (tags & Comm::AUX_345);
    const auto comm_bckp = (tags & Comm::Bckp);

    raise::ErrorIf(not(comm_em or comm_em0 or comm_cur or comm_aux or comm_cur0 or
                       comm_bckp),
                   "CommunicateFields called with no task",
                   HERE);
    if constexpr (not ::traits::engine::DefinesEM0Fields<S>) {
      raise::ErrorIf(comm_em0,
                     "CommunicateFields called with EM0 communication "
                     "for an engine that does not define EM0 fields",
                     HERE);
    }
    if constexpr (not ::traits::engine::DefinesAuxFields<S>) {
      raise::ErrorIf(comm_aux,
                     "CommunicateFields called with AUX communication "
                     "for an engine that does not define AUX fields",
                     HERE);
    }
    if constexpr (not ::traits::engine::DefinesCur0Fields<S>) {
      raise::ErrorIf(comm_cur0,
                     "CommunicateFields called with CUR0 communication "
                     "for an engine that does not define CUR0 fields",
                     HERE);
    }

    std::string comms;
    if (tags & Comm::EM_012) {
      comms += "EM[0-2] ";
    }
    if (tags & Comm::EM_345) {
      comms += "EM[3-5] ";
    }
    if (comm_cur) {
      comms += "CUR ";
    }
    if (tags & Comm::AUX_012) {
      comms += "AUX[0-2] ";
    }
    if (tags & Comm::AUX_345) {
      comms += "AUX[3-5] ";
    }
    if (tags & Comm::EM0_012) {
      comms += "EM0[0-2] ";
    }
    if (tags & Comm::EM0_345) {
      comms += "EM0[3-5] ";
    }
    if (comm_cur0) {
      comms += "CUR0 ";
    }
    if (comm_bckp) {
      comms += "Bckp ";
    }
    logger::Checkpoint(fmt::format("Communicating %s\n", comms.c_str()), HERE);

#if defined(MPI_ENABLED)
    // all fields of the call in one message per direction, all directions
    // in flight at once (field order fixes the message layout)
    const auto split_range = [](bool lo, bool hi) -> cell_range_t {
      if (lo and hi) {
        return { 0, 6 };
      } else if (lo) {
        return { 0, 3 };
      } else {
        return { 3, 6 };
      }
    };
    std::vector<comm::HaloField> flds;
    if (comm_em) {
      flds.push_back(comm::MakeHaloField<M::Dim, 6>(
        domain.fields.em,
        domain.fields.em,
        split_range(tags & Comm::EM_012, tags & Comm::EM_345)));
    }
    if (comm_aux) {
      flds.push_back(comm::MakeHaloField<M::Dim, 6>(
        domain.fields.aux,
        domain.fields.aux,
        split_range(tags & Comm::AUX_012, tags & Comm::AUX_345)));
    }
    if (comm_em0) {
      flds.push_back(comm::MakeHaloField<M::Dim, 6>(
        domain.fields.em0,
        domain.fields.em0,
        split_range(tags & Comm::EM0_012, tags & Comm::EM0_345)));
    }
    if (comm_cur0) {
      flds.push_back(comm::MakeHaloField<M::Dim, 3>(domain.fields.cur0,
                                                    domain.fields.cur0,
                                                    { 0, 3 }));
    }
    if (comm_cur) {
      flds.push_back(comm::MakeHaloField<M::Dim, 3>(domain.fields.cur,
                                                    domain.fields.cur,
                                                    { 0, 3 }));
    }
    if (comm_bckp) {
      flds.push_back(comm::MakeHaloField<M::Dim, 6>(domain.fields.bckp,
                                                    domain.fields.bckp,
                                                    { 0, 6 }));
    }
    g_halo.Exchange(static_cast<int>(tags),
                    HaloDirections(this, domain, g_mpi_rank, false),
                    flds,
                    false);
#else
    // traverse in all directions and send/recv the fields
    for (auto& direction : dir::Directions<M::Dim>::all) {
      const auto [send_params,
                  recv_params] = GetSendRecvParams(this, domain, direction, false);
      const auto [send_indrank, send_slice] = send_params;
      const auto [recv_indrank, recv_slice] = recv_params;
      const auto [send_ind, send_rank]      = send_indrank;
      const auto [recv_ind, recv_rank]      = recv_indrank;
      if (send_rank < 0 and recv_rank < 0) {
        continue;
      }
      if (comm_em) {
        auto comp_range = cell_range_t {};
        if ((tags & Comm::EM_012) and (tags & Comm::EM_345)) {
          comp_range = cell_range_t(0, 6);
        } else if (tags & Comm::EM_012) {
          comp_range = cell_range_t(0, 3);
        } else if (tags & Comm::EM_345) {
          comp_range = cell_range_t(3, 6);
        } else {
          raise::Error("Incorrect logic", HERE);
        }
        comm::CommunicateField<M::Dim, 6>(domain.index(),
                                          domain.fields.em,
                                          domain.fields.em,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          comp_range,
                                          false);
      }
      if (comm_aux) {
        auto comp_range = cell_range_t {};
        if ((tags & Comm::AUX_012) and (tags & Comm::AUX_345)) {
          comp_range = cell_range_t(0, 6);
        } else if (tags & Comm::AUX_012) {
          comp_range = cell_range_t(0, 3);
        } else if (tags & Comm::AUX_345) {
          comp_range = cell_range_t(3, 6);
        } else {
          raise::Error("Incorrect logic", HERE);
        }
        comm::CommunicateField<M::Dim, 6>(domain.index(),
                                          domain.fields.aux,
                                          domain.fields.aux,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          comp_range,
                                          false);
      }
      if (comm_em0) {
        auto comp_range = cell_range_t {};
        if ((tags & Comm::EM0_012) and (tags & Comm::EM0_345)) {
          comp_range = cell_range_t(0, 6);
        } else if (tags & Comm::EM0_012) {
          comp_range = cell_range_t(0, 3);
        } else if (tags & Comm::EM0_345) {
          comp_range = cell_range_t(3, 6);
        } else {
          raise::Error("Incorrect logic", HERE);
        }
        comm::CommunicateField<M::Dim, 6>(domain.index(),
                                          domain.fields.em0,
                                          domain.fields.em0,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          comp_range,
                                          false);
      }
      if (comm_cur0) {
        comm::CommunicateField<M::Dim, 3>(domain.index(),
                                          domain.fields.cur0,
                                          domain.fields.cur0,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          { 0, 3 },
                                          false);
      }
      if (comm_cur) {
        auto comp_range = cell_range_t(0, 3);
        comm::CommunicateField<M::Dim, 3>(domain.index(),
                                          domain.fields.cur,
                                          domain.fields.cur,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          { 0, 3 },
                                          false);
      }
      if (comm_bckp) {
        // copy active -> ghost of bckp (Ec/Bc); read by the hybrid pusher gather
        comm::CommunicateField<M::Dim, 6>(domain.index(),
                                          domain.fields.bckp,
                                          domain.fields.bckp,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          { 0, 6 },
                                          false);
      }
    }
#endif
  }

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::CommunicateBckp(Domain<S, M>&       domain,
                                         const cell_range_t& components) const {
    // Halo FILL of the bckp buffer: copy each neighbor's active boundary cells
    // into this domain's ghost zones (additive=false). This is distinct from
    // SynchronizeFields, which sums ghost-deposited values back into active
    // cells. The renderer needs the ghost halo populated so trilinear sampling
    // near a domain face reads valid neighbor values (C0 across the face).
    for (auto& direction : dir::Directions<M::Dim>::all) {
      const auto [send_params,
                  recv_params] = GetSendRecvParams(this, domain, direction, false);
      const auto [send_indrank, send_slice] = send_params;
      const auto [recv_indrank, recv_slice] = recv_params;
      const auto [send_ind, send_rank]      = send_indrank;
      const auto [recv_ind, recv_rank]      = recv_indrank;
      if (send_rank < 0 and recv_rank < 0) {
        continue;
      }
      comm::CommunicateField<M::Dim, 6>(domain.index(),
                                        domain.fields.bckp,
                                        domain.fields.bckp,
                                        send_ind,
                                        recv_ind,
                                        send_rank,
                                        recv_rank,
                                        send_slice,
                                        recv_slice,
                                        components,
                                        false);
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_COMM(S, M, D)                                               \
  template void Metadomain<S, M<D>>::CommunicateFields(Domain<S, M<D>>&,       \
                                                       CommTags) const;        \
  template void Metadomain<S, M<D>>::CommunicateBckp(Domain<S, M<D>>&,         \
                                                     const cell_range_t&) const;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_COMM)
#undef METADOMAIN_COMM
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
