#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "traits/engine.h"
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
    const bool comm_cur  = (tags & Comm::CUR);
    const bool comm_cur0 = (tags & Comm::CUR0);
    const bool comm_bckp = (tags & Comm::Bckp);
    const bool comm_buff = (tags & Comm::Buff);
    const bool comm_aux  = (tags & Comm::AUX_012) || (tags & Comm::AUX_345);
    raise::ErrorIf(
      not(comm_cur0 || comm_cur || comm_bckp || comm_buff || comm_aux),
      "SynchronizeFields called with no task or incorrect task",
      HERE);
    raise::ErrorIf(
      (comm_cur0 and comm_buff) or (comm_cur and comm_buff),
      "SynchronizeFields cannot sync CUR/CUR0 and Buff at the same time",
      HERE);
    raise::ErrorIf((comm_cur0 and comm_cur),
                   "SynchronizeFields cannot sync CUR and CUR0 at the same "
                   "time (both use Buff as buffer)",
                   HERE);

    if constexpr (not ::traits::engine::DefinesCur0Fields<S>) {
      raise::ErrorIf(comm_cur0,
                     "CommunicateFields called with CUR0 communication "
                     "for an engine that does not define CUR0 fields",
                     HERE);
    }
    if constexpr (not ::traits::engine::DefinesAuxFields<S>) {
      raise::ErrorIf(comm_aux,
                     "SynchronizeFields called with AUX synchronization "
                     "for an engine that does not define AUX fields",
                     HERE);
    }

    const auto SYNCHRONIZE = true;

    std::string comms;
    if (comm_cur) {
      comms += "CUR ";
    }
    if (comm_cur0) {
      comms += "CUR0 ";
    }
    if (comm_bckp) {
      comms += "Bckp ";
    }
    if (comm_buff) {
      comms += "Buff ";
    }
    if (comm_aux) {
      comms += "AUX ";
    }
    logger::Checkpoint(fmt::format("Synchronizing %s\n", comms.c_str()), HERE);

#if defined(MPI_ENABLED)
    // deposit tails of all fields of the call in one message per direction,
    // all directions in flight at once; contributions are summed into the
    // accumulators one direction at a time, in direction order
    ndfield_t<M::Dim, 6>         bckp_recv;
    ndfield_t<M::Dim, 6>         aux_recv;
    ndfield_t<M::Dim, 3>         buff_recv;
    std::vector<comm::HaloField> flds;
    // buff accumulates the received deposit tails of cur/cur0 and is added
    // into cur/cur0 once after the exchange, so it is zeroed exactly once here
    if (comm_cur or comm_cur0) {
      Kokkos::deep_copy(Kokkos::DefaultExecutionSpace {}, domain.fields.buff, ZERO);
    }
    if (comm_cur) {
      flds.push_back(comm::MakeHaloField<M::Dim, 3>(domain.fields.cur,
                                                    domain.fields.buff,
                                                    { 0, 3 }));
    } else if (comm_cur0) {
      flds.push_back(comm::MakeHaloField<M::Dim, 3>(domain.fields.cur0,
                                                    domain.fields.buff,
                                                    { 0, 3 }));
    }
    if (comm_bckp) {
      bckp_recv = g_halo.template Accumulator<6>(0, domain.fields.bckp);
      flds.push_back(
        comm::MakeHaloField<M::Dim, 6>(domain.fields.bckp, bckp_recv, components));
    }
    if (comm_aux) {
      aux_recv = g_halo.template Accumulator<6>(1, domain.fields.aux);
      flds.push_back(
        comm::MakeHaloField<M::Dim, 6>(domain.fields.aux, aux_recv, { 0, 6 }));
    }
    if (comm_buff) {
      buff_recv = g_halo.template Accumulator<3>(0, domain.fields.buff);
      flds.push_back(
        comm::MakeHaloField<M::Dim, 3>(domain.fields.buff, buff_recv, components));
    }
    g_halo.Exchange(static_cast<int>(tags) | (1 << 16),
                    HaloDirections(this, domain, g_mpi_rank, SYNCHRONIZE),
                    flds,
                    true);
#else
    ndfield_t<M::Dim, 6> bckp_recv;
    ndfield_t<M::Dim, 6> aux_recv;
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
    if (comm_aux) {
      if constexpr (M::Dim == Dim::_1D) {
        aux_recv = ndfield_t<M::Dim, 6> { "aux_recv",
                                          domain.fields.aux.extent(0) };
      } else if constexpr (M::Dim == Dim::_2D) {
        aux_recv = ndfield_t<M::Dim, 6> { "aux_recv",
                                          domain.fields.aux.extent(0),
                                          domain.fields.aux.extent(1) };
      } else if constexpr (M::Dim == Dim::_3D) {
        aux_recv = ndfield_t<M::Dim, 6> { "aux_recv",
                                          domain.fields.aux.extent(0),
                                          domain.fields.aux.extent(1),
                                          domain.fields.aux.extent(2) };
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
    // buff accumulates the received deposit tails from every direction and is
    // added into cur/cur0 once after the loop, so it must be zeroed exactly
    // once, here, before the loop. Zeroing it per direction would keep only the
    // last direction's contribution.
    if (comm_cur or comm_cur0) {
      Kokkos::deep_copy(domain.fields.buff, ZERO);
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
      if (comm_cur) {
        comm::CommunicateField<M::Dim, 3>(domain.index(),
                                          domain.fields.cur,
                                          domain.fields.buff,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          { 0, 3 },
                                          SYNCHRONIZE);
      } else if (comm_cur0) {
        comm::CommunicateField<M::Dim, 3>(domain.index(),
                                          domain.fields.cur0,
                                          domain.fields.buff,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          { 0, 3 },
                                          SYNCHRONIZE);
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
                                          SYNCHRONIZE);
      }
      if (comm_aux) {
        // additive remap of moment deposit tails (Pegasus §3.6): accumulate the
        // ghost-cell contributions of aux (V in 0..2, N in 3) into aux_recv
        comm::CommunicateField<M::Dim, 6>(domain.index(),
                                          domain.fields.aux,
                                          aux_recv,
                                          send_ind,
                                          recv_ind,
                                          send_rank,
                                          recv_rank,
                                          send_slice,
                                          recv_slice,
                                          { 0, 6 },
                                          SYNCHRONIZE);
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
                                          SYNCHRONIZE);
      }
    }
#endif
    if (comm_cur) {
      AddBufferedFields<M::Dim, 3>(domain.fields.cur,
                                   domain.fields.buff,
                                   domain.mesh.rangeActiveCells(),
                                   { 0, 3 });
    } else if (comm_cur0) {
      AddBufferedFields<M::Dim, 3>(domain.fields.cur0,
                                   domain.fields.buff,
                                   domain.mesh.rangeActiveCells(),
                                   { 0, 3 });
    }
    if (comm_bckp) {
      AddBufferedFields<M::Dim, 6>(domain.fields.bckp,
                                   bckp_recv,
                                   domain.mesh.rangeActiveCells(),
                                   components);
    }
    if (comm_aux) {
      AddBufferedFields<M::Dim, 6>(domain.fields.aux,
                                   aux_recv,
                                   domain.mesh.rangeActiveCells(),
                                   { 0, 6 });
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
