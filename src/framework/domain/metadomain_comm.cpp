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

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"

  #include "framework/domain/comm_mpi.hpp"
#else
  #include "framework/domain/comm_nompi.hpp"
#endif

#include <Kokkos_Core.hpp>

#include <string>
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
  }

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
        Kokkos::deep_copy(domain.fields.buff, ZERO);
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
        Kokkos::deep_copy(domain.fields.buff, ZERO);
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
#define METADOMAIN_COMM(S, M, D)                                                   \
  template void Metadomain<S, M<D>>::CommunicateFields(Domain<S, M<D>>&,           \
                                                       CommTags) const;            \
  template void Metadomain<S, M<D>>::SynchronizeFields(Domain<S, M<D>>&,           \
                                                       CommTags,                   \
                                                       const cell_range_t&) const; \
  template void Metadomain<S, M<D>>::CommunicateParticles(Domain<S, M<D>>&) const;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_COMM)
#undef METADOMAIN_COMM
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
