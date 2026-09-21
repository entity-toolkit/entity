#include "enums.h"
#include "global.h"

#include "arch/directions.h"
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
    const auto comm_em = ((S == SimEngine::SRPIC) and
                          ((tags & Comm::E) or (tags & Comm::B))) or
                         ((S == SimEngine::GRPIC) and
                          ((tags & Comm::D) or (tags & Comm::B)));
    const bool comm_em0 = (S == SimEngine::GRPIC) and
                          ((tags & Comm::B0) or (tags & Comm::D0));
    const bool comm_j   = (tags & Comm::J);
    const bool comm_aux = (S == SimEngine::GRPIC) and
                          ((tags & Comm::E) or (tags & Comm::H));
    raise::ErrorIf(not(comm_em or comm_em0 or comm_j or comm_aux),
                   "CommunicateFields called with no task",
                   HERE);

    std::string comms;
    if (tags & Comm::E) {
      comms += "E ";
    }
    if (tags & Comm::B) {
      comms += "B ";
    }
    if (tags & Comm::J) {
      comms += "J ";
    }
    if (tags & Comm::D) {
      comms += "D ";
    }
    if (tags & Comm::H) {
      comms += "H ";
    }
    if (tags & Comm::D0) {
      comms += "D0 ";
    }
    if (tags & Comm::B0) {
      comms += "B0 ";
    }
    logger::Checkpoint(fmt::format("Communicating %s\n", comms.c_str()), HERE);

    /**
     * @note this block is designed to support in the future multiple domains
     * on a single rank, however that is not yet implemented
     */
    // establish the last index ranges for fields (i.e., components)
    auto comp_range_fld = cell_range_t {};
    auto comp_range_cur = cell_range_t {};
    if constexpr (S == SimEngine::GRPIC) {
      if (((tags & Comm::D) and (tags & Comm::B)) or
          ((tags & Comm::D0) and (tags & Comm::B0)) or
          ((tags & Comm::E) and (tags & Comm::H))) {
        comp_range_fld = cell_range_t(em::dx1, em::bx3 + 1);
      } else if ((tags & Comm::D) or (tags & Comm::D0) or (tags & Comm::E)) {
        comp_range_fld = cell_range_t(em::dx1, em::dx3 + 1);
      } else if ((tags & Comm::B) or (tags & Comm::B0) or (tags & Comm::H)) {
        comp_range_fld = cell_range_t(em::bx1, em::bx3 + 1);
      }
    } else if constexpr (S == SimEngine::SRPIC) {
      if ((tags & Comm::E) and (tags & Comm::B)) {
        comp_range_fld = cell_range_t(em::ex1, em::bx3 + 1);
      } else if (tags & Comm::E) {
        comp_range_fld = cell_range_t(em::ex1, em::ex3 + 1);
      } else if (tags & Comm::B) {
        comp_range_fld = cell_range_t(em::bx1, em::bx3 + 1);
      }
    } else {
      raise::Error("Unknown simulation engine", HERE);
    }
    if (comm_j) {
      comp_range_cur = cell_range_t(cur::jx1, cur::jx3 + 1);
    }
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
        comm::CommunicateField<M::Dim, 6>(domain.index(),
                                          domain.fields.em,
                                          domain.fields.em,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          comp_range_fld,
                                          false);
      }
      if constexpr (S == SimEngine::GRPIC) {
        if (comm_aux) {
          comm::CommunicateField<M::Dim, 6>(domain.index(),
                                            domain.fields.aux,
                                            domain.fields.aux,
                                            send_ind,
                                            recv_ind,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_fld,
                                            false);
        }
        if (comm_em0) {
          comm::CommunicateField<M::Dim, 6>(domain.index(),
                                            domain.fields.em0,
                                            domain.fields.em0,
                                            send_ind,
                                            recv_ind,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_fld,
                                            false);
          // @HACK_GR_1.2.0 -- this has to be done carefully
          // comm::CommunicateField<M::Dim, 6>(domain.index(),
          //                                   domain.fields.aux,
          //                                   domain.fields.aux,
          //                                   send_ind,
          //                                   recv_ind,
          //                                   send_rank,
          //                                   recv_rank,
          //                                   send_slice,
          //                                   recv_slice,
          //                                   comp_range_fld,
          //                                   false);
        }
        if (comm_j) {
          comm::CommunicateField<M::Dim, 3>(domain.index(),
                                            domain.fields.cur0,
                                            domain.fields.cur0,
                                            send_ind,
                                            recv_ind,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_cur,
                                            false);
        }
      } else {
        if (comm_j) {
          comm::CommunicateField<M::Dim, 3>(domain.index(),
                                            domain.fields.cur,
                                            domain.fields.cur,
                                            send_ind,
                                            recv_ind,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_cur,
                                            false);
        }
      }
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_COMM(S, M, D)                                               \
  template void Metadomain<S, M<D>>::CommunicateFields(Domain<S, M<D>>&,       \
                                                       CommTags) const;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_COMM)
#undef METADOMAIN_COMM
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
