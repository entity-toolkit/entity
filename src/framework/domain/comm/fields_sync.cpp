#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"

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

  template <Dimension D, int N>
  void AddBufferedFields(ndfield_t<D, N>&    field,
                         ndfield_t<D, N>&    buffer,
                         const range_t<D>&   range_policy,
                         const cell_range_t& components) {
    const auto cmin = components.first;
    const auto cmax = components.second;
    if constexpr (D == Dim::_1D) {
      Kokkos::parallel_for(
        "AddBufferedFields",
        range_policy,
        Lambda(cellidx_t i1) {
          for (auto c { cmin }; c < cmax; ++c) {
            field(i1, c) += buffer(i1, c);
          }
        });
    } else if constexpr (D == Dim::_2D) {
      Kokkos::parallel_for(
        "AddBufferedFields",
        range_policy,
        Lambda(cellidx_t i1, cellidx_t i2) {
          for (auto c { cmin }; c < cmax; ++c) {
            field(i1, i2, c) += buffer(i1, i2, c);
          }
        });
    } else if constexpr (D == Dim::_3D) {
      Kokkos::parallel_for(
        "AddBuffers",
        range_policy,
        Lambda(cellidx_t i1, cellidx_t i2, cellidx_t i3) {
          for (auto c { cmin }; c < cmax; ++c) {
            field(i1, i2, i3, c) += buffer(i1, i2, i3, c);
          }
        });
    } else {
      raise::Error("Wrong Dimension", HERE);
    }
  }

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::SynchronizeFields(Domain<S, M>& domain,
                                           CommTags      tags,
                                           const cell_range_t& components) const {
    const bool comm_j    = (tags & Comm::J);
    const bool comm_bckp = (tags & Comm::Bckp);
    const bool comm_buff = (tags & Comm::Buff);
    raise::ErrorIf(not(comm_j || comm_bckp || comm_buff),
                   "SynchronizeFields called with no task or incorrect task",
                   HERE);
    raise::ErrorIf(comm_j and comm_buff,
                   "SynchronizeFields cannot sync J and Buff at the same time",
                   HERE);
    const auto synchronize = true;

    std::string comms;
    if (comm_j) {
      comms += "J ";
    }
    if (comm_bckp) {
      comms += "Bckp ";
    }
    if (comm_buff) {
      comms += "Buff ";
    }
    logger::Checkpoint(fmt::format("Synchronizing %s\n", comms.c_str()), HERE);

    auto comp_range_cur = cell_range_t {};
    if (comm_j) {
      comp_range_cur = cell_range_t(cur::jx1, cur::jx3 + 1);
      Kokkos::deep_copy(domain.fields.buff, ZERO);
    }
    ndfield_t<M::Dim, 6> bckp_recv;
    ndfield_t<M::Dim, 3> buff_recv;
    if (comm_bckp) {
      if constexpr (M::Dim == Dim::_1D) {
        bckp_recv = ndfield_t<M::Dim, 6> { "bckp_recv",
                                           domain.fields.bckp.extent(0) };
      } else if constexpr (M::Dim == Dim::_2D) {
        bckp_recv = ndfield_t<M::Dim, 6> { "bckp_recv",
                                           domain.fields.bckp.extent(0),
                                           domain.fields.bckp.extent(1) };
      } else if constexpr (M::Dim == Dim::_3D) {
        bckp_recv = ndfield_t<M::Dim, 6> { "bckp_recv",
                                           domain.fields.bckp.extent(0),
                                           domain.fields.bckp.extent(1),
                                           domain.fields.bckp.extent(2) };
      }
    }
    if (comm_buff) {
      if constexpr (M::Dim == Dim::_1D) {
        buff_recv = ndfield_t<M::Dim, 3> { "buff_recv",
                                           domain.fields.buff.extent(0) };
      } else if constexpr (M::Dim == Dim::_2D) {
        buff_recv = ndfield_t<M::Dim, 3> { "buff_recv",
                                           domain.fields.buff.extent(0),
                                           domain.fields.buff.extent(1) };
      } else if constexpr (M::Dim == Dim::_3D) {
        buff_recv = ndfield_t<M::Dim, 3> { "buff_recv",
                                           domain.fields.buff.extent(0),
                                           domain.fields.buff.extent(1),
                                           domain.fields.buff.extent(2) };
      }
    }
    // traverse in all directions and sync the fields
    for (auto& direction : dir::Directions<M::Dim>::all) {
      const auto [send_params,
                  recv_params] = GetSendRecvParams(this, domain, direction, true);
      const auto [send_indrank, send_slice] = send_params;
      const auto [recv_indrank, recv_slice] = recv_params;
      const auto [send_ind, send_rank]      = send_indrank;
      const auto [recv_ind, recv_rank]      = recv_indrank;
      if (send_rank < 0 and recv_rank < 0) {
        continue;
      }
      if (comm_j) {
        if constexpr (S == SimEngine::GRPIC) {
          comm::CommunicateField<M::Dim, 3>(domain.index(),
                                            domain.fields.cur0,
                                            domain.fields.buff,
                                            send_ind,
                                            recv_ind,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_cur,
                                            synchronize);
        } else {
          comm::CommunicateField<M::Dim, 3>(domain.index(),
                                            domain.fields.cur,
                                            domain.fields.buff,
                                            send_ind,
                                            recv_ind,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_cur,
                                            synchronize);
        }
      }
      if (comm_bckp) {
        comm::CommunicateField<M::Dim, 6>(domain.index(),
                                          domain.fields.bckp,
                                          bckp_recv,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          components,
                                          synchronize);
      }
      if (comm_buff) {
        comm::CommunicateField<M::Dim, 3>(domain.index(),
                                          domain.fields.buff,
                                          buff_recv,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          components,
                                          synchronize);
      }
    }
    if (comm_j) {
      if constexpr (S == SimEngine::GRPIC) {
        AddBufferedFields<M::Dim, 3>(domain.fields.cur0,
                                     domain.fields.buff,
                                     domain.mesh.rangeActiveCells(),
                                     comp_range_cur);
      } else {
        AddBufferedFields<M::Dim, 3>(domain.fields.cur,
                                     domain.fields.buff,
                                     domain.mesh.rangeActiveCells(),
                                     comp_range_cur);
      }
    }
    if (comm_bckp) {
      AddBufferedFields<M::Dim, 6>(domain.fields.bckp,
                                   bckp_recv,
                                   domain.mesh.rangeActiveCells(),
                                   components);
    }
    if (comm_buff) {
      AddBufferedFields<M::Dim, 3>(domain.fields.buff,
                                   buff_recv,
                                   domain.mesh.rangeActiveCells(),
                                   components);
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_COMM(S, M, D)                                               \
  template void Metadomain<S, M<D>>::SynchronizeFields(Domain<S, M<D>>&,       \
                                                       CommTags,               \
                                                       const cell_range_t&) const;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_COMM)
#undef METADOMAIN_COMM
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
