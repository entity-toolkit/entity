#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/timer.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"

  #include "framework/domain/comm_mpi.hpp"
#else
  #include "framework/domain/comm_nompi.hpp"
#endif

#include <utility>
#include <vector>

namespace ntt {

  using address_t     = std::pair<unsigned int, int>;
  using comm_params_t = std::pair<address_t, std::vector<range_tuple_t>>;

  template <SimEngine::type S, class M>
  auto GetSendRecvRanks(Metadomain<S, M>*        metadomain,
                        Domain<S, M>&            domain,
                        dir::direction_t<M::Dim> direction)
    -> std::pair<address_t, address_t> {
    Domain<S, M>* send_to_nghbr_ptr   = nullptr;
    Domain<S, M>* recv_from_nghbr_ptr = nullptr;
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
        {0, -1},
        {0, -1}
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
      {send_ind, send_rank},
      {recv_ind, recv_rank}
    };
  }

  template <SimEngine::type S, class M>
  auto GetSendRecvParams(Metadomain<S, M>*        metadomain,
                         Domain<S, M>&            domain,
                         dir::direction_t<M::Dim> direction,
                         bool                     synchronize)
    -> std::pair<comm_params_t, comm_params_t> {
    const auto [send_indrank,
                recv_indrank] = GetSendRecvRanks(metadomain, domain, direction);
    const auto [send_ind, send_rank] = send_indrank;
    const auto [recv_ind, recv_rank] = recv_indrank;
    const auto is_sending            = (send_rank >= 0);
    const auto is_receiving          = (recv_rank >= 0);
    if (not(is_sending or is_receiving)) {
      return {
        {{ 0, -1 }, {}},
        {{ 0, -1 }, {}}
      };
    }
    auto     send_slice   = std::vector<range_tuple_t> {};
    auto     recv_slice   = std::vector<range_tuple_t> {};
    const in components[] = { in::x1, in::x2, in::x3 };
    // find the field components and indices to be sent/received
    for (std::size_t d { 0 }; d < direction.size(); ++d) {
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
      {{ send_ind, send_rank }, send_slice},
      {{ recv_ind, recv_rank }, recv_slice},
    };
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::CommunicateFields(Domain<S, M>& domain, CommTags tags) {
    const auto comm_fields = (tags & Comm::E) || (tags & Comm::B) ||
                             (tags & Comm::J) || (tags & Comm::D) ||
                             (tags & Comm::D0) || (tags & Comm::B0);
    const bool comm_em = (tags & Comm::E) || (tags & Comm::B) || (tags & Comm::D);
    const bool comm_em0 = (tags & Comm::B0) || (tags & Comm::D0);
    const bool comm_j   = (tags & Comm::J);
    raise::ErrorIf(not comm_fields, "CommunicateFields called with no task", HERE);

    std::string comms = "";
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
    auto comp_range_fld = range_tuple_t {};
    auto comp_range_cur = range_tuple_t {};
    if constexpr (S == SimEngine::GRPIC) {
      if (((tags & Comm::D) && (tags & Comm::B)) ||
          ((tags & Comm::D0) && (tags & Comm::B0))) {
        comp_range_fld = range_tuple_t(em::dx1, em::bx3 + 1);
      } else if ((tags & Comm::D) || (tags & Comm::D0)) {
        comp_range_fld = range_tuple_t(em::dx1, em::dx3 + 1);
      } else if ((tags & Comm::B) || (tags & Comm::B0)) {
        comp_range_fld = range_tuple_t(em::bx1, em::bx3 + 1);
      }
    } else if constexpr (S == SimEngine::SRPIC) {
      if ((tags & Comm::E) && (tags & Comm::B)) {
        comp_range_fld = range_tuple_t(em::ex1, em::bx3 + 1);
      } else if (tags & Comm::E) {
        comp_range_fld = range_tuple_t(em::ex1, em::ex3 + 1);
      } else if (tags & Comm::B) {
        comp_range_fld = range_tuple_t(em::bx1, em::bx3 + 1);
      }
    } else {
      raise::Error("Unknown simulation engine", HERE);
    }
    if (comm_j) {
      comp_range_cur = range_tuple_t(cur::jx1, cur::jx3 + 1);
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
        }
      }
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

  template <Dimension D, int N>
  void AddBufferedFields(ndfield_t<D, N>&     field,
                         ndfield_t<D, N>&     buffer,
                         const range_t<D>&    range_policy,
                         const range_tuple_t& components) {
    const auto cmin = components.first;
    const auto cmax = components.second;
    if constexpr (D == Dim::_1D) {
      Kokkos::parallel_for(
        "AddBufferedFields",
        range_policy,
        Lambda(index_t i1) {
          for (auto c { cmin }; c < cmax; ++c) {
            field(i1, c) += buffer(i1, c);
          }
        });
    } else if constexpr (D == Dim::_2D) {
      Kokkos::parallel_for(
        "AddBufferedFields",
        range_policy,
        Lambda(index_t i1, index_t i2) {
          for (auto c { cmin }; c < cmax; ++c) {
            field(i1, i2, c) += buffer(i1, i2, c);
          }
        });
    } else if constexpr (D == Dim::_3D) {
      Kokkos::parallel_for(
        "AddBuffers",
        range_policy,
        Lambda(index_t i1, index_t i2, index_t i3) {
          for (auto c { cmin }; c < cmax; ++c) {
            field(i1, i2, i3, c) += buffer(i1, i2, i3, c);
          }
        });
    } else {
      raise::Error("Wrong Dimension", HERE);
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::SynchronizeFields(Domain<S, M>&        domain,
                                           CommTags             tags,
                                           const range_tuple_t& components) {
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

    std::string comms = "";
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

    auto comp_range_cur = range_tuple_t {};
    if (comm_j) {
      comp_range_cur = range_tuple_t(cur::jx1, cur::jx3 + 1);
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
      AddBufferedFields<M::Dim, 3>(domain.fields.cur,
                                   domain.fields.buff,
                                   domain.mesh.rangeActiveCells(),
                                   comp_range_cur);
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

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::CommunicateParticles(Domain<S, M>&  domain,
                                              timer::Timers* timers) {
    raise::ErrorIf(timers == nullptr,
                   "Timers not passed when Comm::Prtl called",
                   HERE);
    logger::Checkpoint("Communicating particles\n", HERE);
    for (auto& species : domain.species) {
      // at this point particles should already by tagged in the pusher
      timers->start("Sorting");
      const auto npart_per_tag = species.SortByTags();
      timers->stop("Sorting");
#if defined(MPI_ENABLED)
      timers->start("Communications");
      // only necessary when MPI is enabled
      /**
       *                                                        index_last
       *                                                            |
       *    alive      new dead         tag1       tag2             v     dead
       * [ 11111111   000000000    222222222    3333333 .... nnnnnnn  00000000 ... ]
       *                           ^        ^
       *                           |        |
       *     tag_offset[tag1] -----+        +----- tag_offset[tag1] + npart_per_tag[tag1]
       *          "send_pmin"                      "send_pmax" (after last element)
       */
      auto tag_offset { npart_per_tag };
      for (std::size_t i { 1 }; i < tag_offset.size(); ++i) {
        tag_offset[i] += tag_offset[i - 1];
      }
      for (std::size_t i { 0 }; i < tag_offset.size(); ++i) {
        tag_offset[i] -= npart_per_tag[i];
      }
      auto index_last = tag_offset[tag_offset.size() - 1] +
                        npart_per_tag[npart_per_tag.size() - 1];
      for (auto& direction : dir::Directions<D>::all) {
        const auto [send_params,
                    recv_params] = GetSendRecvParams(this, domain, direction, true);
        const auto [send_indrank, send_slice] = send_params;
        const auto [recv_indrank, recv_slice] = recv_params;
        const auto [send_ind, send_rank]      = send_indrank;
        const auto [recv_ind, recv_rank]      = recv_indrank;
        if (send_rank < 0 and recv_rank < 0) {
          continue;
        }
        const auto send_dir_tag = mpi::PrtlSendTag<D>::dir2tag(direction);
        const auto nsend        = npart_per_tag[send_dir_tag];
        const auto send_pmin    = tag_offset[send_dir_tag];
        const auto send_pmax    = tag_offset[send_dir_tag] + nsend;
        const auto recv_count = comm::CommunicateParticles<M::Dim, M::CoordType>(
          species,
          send_rank,
          recv_rank,
          { send_pmin, send_pmax },
          index_last);
        if (recv_count > 0) {
          if constexpr (D == Dim::_1D) {
            int shift_in_x1 { 0 };
            if ((-direction)[0] == -1) {
              shift_in_x1 = -subdomain(recv_ind).mesh.n_active(in::x1);
            } else if ((-direction)[0] == 1) {
              shift_in_x1 = domain.mesh.n_active(in::x1);
            }
            auto& this_tag     = species.tag;
            auto& this_i1      = species.i1;
            auto& this_i1_prev = species.i1_prev;
            Kokkos::parallel_for(
              "CommunicateParticles",
              recv_count,
              Lambda(index_t p) {
                this_tag(index_last + p)      = ParticleTag::alive;
                this_i1(index_last + p)      += shift_in_x1;
                this_i1_prev(index_last + p) += shift_in_x1;
              });
          } else if constexpr (D == Dim::_2D) {
            int shift_in_x1 { 0 }, shift_in_x2 { 0 };
            if ((-direction)[0] == -1) {
              shift_in_x1 = -subdomain(recv_ind).mesh.n_active(in::x1);
            } else if ((-direction)[0] == 1) {
              shift_in_x1 = domain.mesh.n_active()[0];
            }
            if ((-direction)[1] == -1) {
              shift_in_x2 = -subdomain(recv_ind).mesh.n_active(in::x2);
            } else if ((-direction)[1] == 1) {
              shift_in_x2 = domain.mesh.n_active(in::x2);
            }
            auto& this_tag     = species.tag;
            auto& this_i1      = species.i1;
            auto& this_i2      = species.i2;
            auto& this_i1_prev = species.i1_prev;
            auto& this_i2_prev = species.i2_prev;
            Kokkos::parallel_for(
              "CommunicateParticles",
              recv_count,
              Lambda(index_t p) {
                this_tag(index_last + p)      = ParticleTag::alive;
                this_i1(index_last + p)      += shift_in_x1;
                this_i2(index_last + p)      += shift_in_x2;
                this_i1_prev(index_last + p) += shift_in_x1;
                this_i2_prev(index_last + p) += shift_in_x2;
              });
          } else if constexpr (D == Dim::_3D) {
            int shift_in_x1 { 0 }, shift_in_x2 { 0 }, shift_in_x3 { 0 };
            if ((-direction)[0] == -1) {
              shift_in_x1 = -subdomain(recv_ind).mesh.n_active(in::x1);
            } else if ((-direction)[0] == 1) {
              shift_in_x1 = domain.mesh.n_active(in::x1);
            }
            if ((-direction)[1] == -1) {
              shift_in_x2 = -subdomain(recv_ind).mesh.n_active(in::x2);
            } else if ((-direction)[1] == 1) {
              shift_in_x2 = domain.mesh.n_active(in::x2);
            }
            if ((-direction)[2] == -1) {
              shift_in_x3 = -subdomain(recv_ind).mesh.n_active(in::x3);
            } else if ((-direction)[2] == 1) {
              shift_in_x3 = domain.mesh.n_active(in::x3);
            }
            auto& this_tag     = species.tag;
            auto& this_i1      = species.i1;
            auto& this_i2      = species.i2;
            auto& this_i3      = species.i3;
            auto& this_i1_prev = species.i1_prev;
            auto& this_i2_prev = species.i2_prev;
            auto& this_i3_prev = species.i3_prev;
            Kokkos::parallel_for(
              "CommunicateParticles",
              recv_count,
              Lambda(index_t p) {
                this_tag(index_last + p)      = ParticleTag::alive;
                this_i1(index_last + p)      += shift_in_x1;
                this_i2(index_last + p)      += shift_in_x2;
                this_i3(index_last + p)      += shift_in_x3;
                this_i1_prev(index_last + p) += shift_in_x1;
                this_i2_prev(index_last + p) += shift_in_x2;
                this_i3_prev(index_last + p) += shift_in_x3;
              });
          }
          index_last += recv_count;
          species.set_npart(index_last);
        }
        Kokkos::deep_copy(
          Kokkos::subview(species.tag, std::make_pair(send_pmin, send_pmax)),
          ParticleTag::dead);
      }
      timers->stop("Communications");
      // !TODO: maybe there is a way to not sort twice
      timers->start("Sorting");
      species.set_unsorted();
      species.SortByTags();
      timers->stop("Sorting");
#endif
    }
  }

  /*
    New function to communicate particles using a buffer
  */
  template <SimEngine::type S, class M>
  void Metadomain<S, M>::CommunicateParticlesBuffer(Domain<S, M>&  domain,
                                                    timer::Timers* timers) {
    raise::ErrorIf(timers == nullptr,
                   "Timers not passed when Comm::Prtl called",
                   HERE);
    logger::Checkpoint("Communicating particles\n", HERE);
    for (auto& species : domain.species) {
      /*
        Brief on arrays
        npart_per_tag_arr (vector):       | dead count| alive count | tag=1 count | tag=2 count | ...
                                          <--------------------------size = ntags()-------------------------->
        tag_offset (Kokkos::View):        | 0 | dead count | dead + alive count | dead + alive + tag=1 count | ...
                                          <--------------------------size = ntags()-------------------------->
        npart_per_tag_arr_recv (vector):  | 0 | 0 | nrecv1 | nrecv2 | ...
                                          <--------------------------size = ntags()-------------------------->
      */
      auto [npart_per_tag_arr,
            tag_offset]       = species.npart_per_tag();
      auto npart              = static_cast<std::size_t>(species.npart());
      auto total_alive        = static_cast<std::size_t>(
                                npart_per_tag_arr[ParticleTag::alive]);
      auto total_dead         = static_cast<std::size_t>(
                                npart_per_tag_arr[ParticleTag::dead]);
      auto total_holes        = static_cast<std::size_t>(npart - total_alive);
      auto total_recv         = static_cast<std::size_t>(0);

      std::vector<int> send_ranks, send_inds;
      std::vector<int> recv_ranks, recv_inds;
      // at this point particles should already by tagged in the pusher
#if defined(MPI_ENABLED)
      std::vector<std::size_t> npart_per_tag_arr_recv(species.ntags(), 0);
      Kokkos::View<int*> shifts_in_x1("shifts_in_x1", species.ntags());
      Kokkos::View<int*> shifts_in_x2("shifts_in_x2", species.ntags());
      Kokkos::View<int*> shifts_in_x3("shifts_in_x3", species.ntags());
      auto shifts_in_x1_h = Kokkos::create_mirror_view(shifts_in_x1);
      auto shifts_in_x2_h = Kokkos::create_mirror_view(shifts_in_x2);
      auto shifts_in_x3_h = Kokkos::create_mirror_view(shifts_in_x3);
      dir::dirs_t<D> legal_directions;

      // Get receive counts + displacements
      for (auto& direction : dir::Directions<D>::all) {
        const auto tag_recv = mpi::PrtlSendTag<D>::dir2tag(-direction);
        const auto tag_send = mpi::PrtlSendTag<D>::dir2tag(direction);
        const auto [send_params,
                    recv_params] = GetSendRecvParams(this, domain, direction, true);
        const auto [send_indrank, send_slice] = send_params;
        const auto [recv_indrank, recv_slice] = recv_params;
        const auto [send_ind, send_rank]      = send_indrank;
        const auto [recv_ind, recv_rank]      = recv_indrank;
        if (send_rank < 0 and recv_rank < 0) {
          continue;
        }
        const auto  nsend        = npart_per_tag_arr[tag_send];
        std::size_t nrecv        = 0;

	legal_directions.push_back(direction);
        send_ranks.push_back(send_rank);
        recv_ranks.push_back(recv_rank);
        send_inds.push_back(send_ind);
        recv_inds.push_back(recv_ind);
        comm::ParticleSendRecvCount(send_rank, recv_rank, nsend, nrecv);
        total_recv += nrecv;
        npart_per_tag_arr_recv[tag_recv] = nrecv;
        // Perform displacements before sending
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          if ((-direction)[0] == -1) {
            shifts_in_x1_h(tag_recv) = subdomain(recv_ind).mesh.n_active(in::x1);
          } else if ((-direction)[0] == 1) {
            shifts_in_x1_h(tag_recv) = -domain.mesh.n_active(in::x1);
          }
        } 
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          if ((-direction)[1] == -1) {
            shifts_in_x2_h(tag_recv) = subdomain(recv_ind).mesh.n_active(in::x2);
          } else if ((-direction)[1] == 1) {
            shifts_in_x2_h(tag_recv) = -domain.mesh.n_active(in::x2);
          }
        } 
        if constexpr (D == Dim::_3D) {
          if ((-direction)[2] == -1) {
            shifts_in_x3_h(tag_recv) = subdomain(recv_ind).mesh.n_active(in::x3);
          } else if ((-direction)[2] == 1) {
            shifts_in_x3_h(tag_recv) = -domain.mesh.n_active(in::x3);
          }
        }
      } // end directions loop
      Kokkos::deep_copy(shifts_in_x1, shifts_in_x1_h);
      Kokkos::deep_copy(shifts_in_x2, shifts_in_x2_h);
      Kokkos::deep_copy(shifts_in_x3, shifts_in_x3_h);

      raise::FatalIf((npart + total_recv) >= species.maxnpart(),
                     "Too many particles to receive (cannot fit into maxptl)",
                     HERE);

      auto& this_tag        = species.tag;
      auto& this_i1         = species.i1;
      auto& this_i1_prev    = species.i1_prev;
      auto& this_i2         = species.i2;
      auto& this_i2_prev    = species.i2_prev;
      auto& this_i3         = species.i3;
      auto& this_i3_prev    = species.i3_prev;

      /* 
          Brief on permute vector: It contains the sorted indices of tag != alive particles
          E.g., consider the following tag array
          species.tag =     [ 0, 0, 1, 0, 2, 3,...]
          Then, permute vector will look something like
          permute_vector =  [0, 1, 3, ...,  4, ..., ...   5, ...          ]
                            |<--------- >| |<----->|      |<----->| ....
                               tag=dead ct  tag=2 ct       tag=3 ct
      */
      Kokkos::View<std::size_t*> permute_vector("permute_vector", total_holes);
      Kokkos::View<std::size_t*> current_offset("current_offset", species.ntags());
      auto &this_tag_offset = tag_offset;

      auto n_alive = npart_per_tag_arr[ParticleTag::alive];

      if constexpr (D == Dim::_1D){
      Kokkos::parallel_for(
        "PermuteVector and Displace",
        species.npart(),
        Lambda(index_t p) {
          const auto current_tag     = this_tag(p);
          if (current_tag != ParticleTag::alive){
            // dead tags only
            if (current_tag == ParticleTag::dead) {
              const auto idx_permute_vec =  Kokkos::atomic_fetch_add(
                                            &current_offset(current_tag),
                                            1);
              permute_vector(idx_permute_vec) = p;
            }
            // tag = 1->N (excluding dead and alive)
            else{
              const auto idx_permute_vec =  this_tag_offset(current_tag) -
                                            n_alive + 
                                            Kokkos::atomic_fetch_add(
                                            &current_offset(current_tag),
                                            1);
              permute_vector(idx_permute_vec) = p;
              this_i1(p)      += shifts_in_x1(current_tag);
              this_i1_prev(p) += shifts_in_x1(current_tag);
            }
          }
        });
      }

      if constexpr (D == Dim::_2D){
      Kokkos::parallel_for(
        "PermuteVector and Displace",
        species.npart(),
        Lambda(index_t p) {
          const auto current_tag     = this_tag(p);
          if (current_tag != ParticleTag::alive){
            // dead tags only
            if (current_tag == ParticleTag::dead) {
              const auto idx_permute_vec =  Kokkos::atomic_fetch_add(
                                            &current_offset(current_tag),
                                            1);
              permute_vector(idx_permute_vec) = p;
            }
            // tag = 1->N (excluding dead and alive)
            else{
              const auto idx_permute_vec =  this_tag_offset(current_tag) -
                                            n_alive + 
                                            Kokkos::atomic_fetch_add(
                                            &current_offset(current_tag),
                                            1);
              permute_vector(idx_permute_vec) = p;
              this_i1(p)      += shifts_in_x1(current_tag);
              this_i1_prev(p) += shifts_in_x1(current_tag);
              this_i2(p)      += shifts_in_x2(current_tag);
              this_i2_prev(p) += shifts_in_x2(current_tag);
            }
          }
        });
      }

      if constexpr (D == Dim::_3D){
      Kokkos::parallel_for(
        "PermuteVector and Displace",
        species.npart(),
        Lambda(index_t p) {
          const auto current_tag     = this_tag(p);
          if (current_tag != ParticleTag::alive){
            // dead tags only
            if (current_tag == ParticleTag::dead) {
              const auto idx_permute_vec =  Kokkos::atomic_fetch_add(
                                            &current_offset(current_tag),
                                            1);
              permute_vector(idx_permute_vec) = p;
            }
            // tag = 1->N (excluding dead and alive)
            else{
              const auto idx_permute_vec =  this_tag_offset(current_tag) -
                                            n_alive + 
                                            Kokkos::atomic_fetch_add(
                                            &current_offset(current_tag),
                                            1);
              permute_vector(idx_permute_vec) = p;
              this_i1(p)      += shifts_in_x1(current_tag);
              this_i1_prev(p) += shifts_in_x1(current_tag);
              this_i2(p)      += shifts_in_x2(current_tag);
              this_i2_prev(p) += shifts_in_x2(current_tag);
              this_i3(p)      += shifts_in_x3(current_tag);
              this_i3_prev(p) += shifts_in_x3(current_tag);
            }
          }
        });
      }



      // Sanity check: npart_per_tag must be equal to the current offset except tag=alive
      auto current_offset_h = Kokkos::create_mirror_view(current_offset);
      Kokkos::deep_copy(current_offset_h, current_offset);
      for (std::size_t i { 0 }; i < species.ntags(); ++i) {
        if (i != ParticleTag::alive){
        raise::FatalIf(current_offset_h(i) != npart_per_tag_arr[i],
                       "Error in permute vector construction",
                       HERE);
        }
        else{
          raise::FatalIf(current_offset_h(i) != 0,
                       "Error in permute vector construction",
                       HERE);
        }
      }

      /*
          Brief on allocation vector: It contains the indices of holes that are filled
          by the particles received from other domains
          case 1: total_recv > nholes
            allocation_vector = | i1 | i2 | i3 | ....    | npart | npart + 1 | ...
                                <-------total_holes------>   <---total_recv - nholes-->
                               (same as permuute vector)   (extra particles appended at end)
          case 2: total_recv <= nholes
            allocation_vector = | i1 | i2 | i3 | ....
                                <----total_recv----->
                                (same as permuute vector)
      */
      Kokkos::View<std::size_t*> allocation_vector("allocation_vector", total_recv);
      if (total_recv > total_holes)
      {
        // Fill the first bit with the permute vector; these are the holes to be filled
        Kokkos::parallel_for(
          "AllocationVector",
          total_holes,
          Lambda(index_t p) {
            allocation_vector(p) = permute_vector(p);
          });

        // Now allocate the rest to the end of the array
        Kokkos::parallel_for(
          "AllocationVector",
          total_recv - total_holes,
          Lambda(index_t p) {
            allocation_vector(total_holes + p) = static_cast<int>(npart + p);
          });
      }
      else
      {   Kokkos::parallel_for(
          "AllocationVector",
          total_recv,
          Lambda(index_t p) {
            allocation_vector(p) = permute_vector(p);
          });
      }

      /*
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 1 && species.label() == "e+_b")
      {
        // Copy the tag array to host
        auto tag_h = Kokkos::create_mirror_view(species.tag);
        Kokkos::deep_copy(tag_h, species.tag);
        std::cout << "Tag locs before send" << std::endl;
        for (std::size_t i { 0 }; i < species.npart(); i++) {
          if (tag_h(i) != ParticleTag::alive)
            std::cout <<" Tag: " << tag_h(i) << " loc: "<< i << std::endl;
        }

        // Print allocation vector after copying to host
        auto allocation_vector_h = Kokkos::create_mirror_view(allocation_vector);
        std::cout << "Total holes: " << total_holes << " Total recv: " << total_recv << std::endl;
        Kokkos::deep_copy(allocation_vector_h, allocation_vector);
        for (std::size_t i { 0 }; i < total_recv; ++i) {
          std::cout << "Rank: " << rank << " Allocation vector: " << allocation_vector_h(i) << std::endl;
        }
        // Print the permute vector as well
        auto permute_vector_h = Kokkos::create_mirror_view(permute_vector);
        Kokkos::deep_copy(permute_vector_h, permute_vector);
        for (std::size_t i { 0 }; i < total_holes; ++i) {
          std::cout << "Rank: " << rank << " Permuted vector: " << permute_vector_h(i) << 
          " tag: " << tag_h(permute_vector_h(i)) << std::endl;
        }
      }
      */
     
      // Communicate the arrays
      comm::CommunicateParticlesBuffer<M::Dim, M::CoordType>(species, permute_vector, allocation_vector,
                                        this_tag_offset, npart_per_tag_arr, npart_per_tag_arr_recv,
                                        send_ranks, recv_ranks, legal_directions);
#endif
    }
  }

  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;

} // namespace ntt
