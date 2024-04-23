#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/error.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include "framework/domain/comm_mpi.hpp"
#else
  #include "framework/domain/comm_nompi.hpp"
#endif

#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::Communicate(Domain<S, M>& domain, CommTags tags) {
    const auto comm_fields = (tags & Comm::E) || (tags & Comm::B) ||
                             (tags & Comm::J) || (tags & Comm::D) ||
                             (tags & Comm::D0) || (tags & Comm::B0);
    const bool comm_em = (tags & Comm::E) || (tags & Comm::B) || (tags & Comm::D);
    const bool comm_em0 = (tags & Comm::B0) || (tags & Comm::D0);
    const bool comm_j   = (tags & Comm::J);
    const bool sync_j   = (tags & Comm::J_sync);
    raise::ErrorIf(comm_fields && sync_j,
                   "Cannot communicate fields and sync currents simultaneously",
                   HERE);

    /**
     * @note this block is designed to support in the future multiple domains
     * on a single rank, however that is not yet implemented
     */
    if (comm_fields || sync_j) {
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
      if (comm_j || sync_j) {
        comp_range_cur = range_tuple_t(cur::jx1, cur::jx3 + 1);
      }
      // traverse in all directions and send/recv the fields
      for (auto& direction : dir::Directions<M::Dim>::all) {
        Domain<S, M>* send_to_nghbr_ptr   = nullptr;
        Domain<S, M>* recv_from_nghbr_ptr = nullptr;
        // set pointers to the correct send/recv domains
        // can coincide with the current domain if periodic
        if (domain.mesh.flds_bc_in(direction) == FldsBC::PERIODIC) {
          // sending / receiving from itself
          raise::ErrorIf(
            domain.neighbor_idx_in(direction) != domain.index(),
            "Periodic boundaries imply communication within the same domain",
            HERE);
          raise::ErrorIf(
            domain.mesh.flds_bc_in(-direction) != FldsBC::PERIODIC,
            "Periodic boundary conditions must be set in both directions",
            HERE);
          send_to_nghbr_ptr = recv_from_nghbr_ptr = &domain;
        } else if (domain.mesh.flds_bc_in(direction) == FldsBC::SYNC) {
          // sending to other domain
          raise::ErrorIf(
            domain.neighbor_idx_in(direction) == domain.index(),
            "Sync boundaries imply communication between separate domains",
            HERE);
          send_to_nghbr_ptr = subdomain_ptr(domain.neighbor_idx_in(direction));
          if (domain.mesh.flds_bc_in(-direction) == FldsBC::SYNC) {
            // receiving from other domain
            raise::ErrorIf(
              domain.neighbor_idx_in(-direction) == domain.index(),
              "Sync boundaries imply communication between separate domains",
              HERE);
            recv_from_nghbr_ptr = subdomain_ptr(domain.neighbor_idx_in(-direction));
          }
        } else if (domain.mesh.flds_bc_in(-direction) == FldsBC::SYNC) {
          // only receiving from other domain
          raise::ErrorIf(
            domain.neighbor_idx_in(-direction) == domain.index(),
            "Sync boundaries imply communication between separate domains",
            HERE);
          recv_from_nghbr_ptr = subdomain_ptr(domain.neighbor_idx_in(-direction));
        } else {
          // no communication necessary
          continue;
        }
        const auto is_sending   = (send_to_nghbr_ptr != nullptr);
        const auto is_receiving = (recv_from_nghbr_ptr != nullptr);
        auto       send_slice   = std::vector<range_tuple_t> {};
        auto       recv_slice   = std::vector<range_tuple_t> {};
        const in   components[] = { in::x1, in::x2, in::x3 };
        // find the field components and indices to be sent/received
        for (std::size_t d { 0 }; d < direction.size(); ++d) {
          const auto c   = components[d];
          const auto dir = direction[d];
          if (comm_fields) {
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
          } else if (sync_j) {
            if (is_sending) {
              if (dir == 0) {
                send_slice.emplace_back(domain.mesh.i_min(c), domain.mesh.i_max(c));
              } else if (dir == 1) {
                send_slice.emplace_back(domain.mesh.i_max(c),
                                        domain.mesh.i_max(c) + N_GHOSTS);
              } else {
                send_slice.emplace_back(domain.mesh.i_min(c) - N_GHOSTS,
                                        domain.mesh.i_min(c));
              }
            }
            if (is_receiving) {
              if (-dir == 0) {
                recv_slice.emplace_back(domain.mesh.i_min(c), domain.mesh.i_max(c));
              } else if (-dir == 1) {
                recv_slice.emplace_back(domain.mesh.i_max(c) - N_GHOSTS,
                                        domain.mesh.i_max(c));
              } else {
                recv_slice.emplace_back(domain.mesh.i_min(c),
                                        domain.mesh.i_min(c) + N_GHOSTS);
              }
            }
          }
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
        const auto send_idx = (send_to_nghbr_ptr != nullptr)
                                ? send_to_nghbr_ptr->index()
                                : 0;
        const auto recv_idx = (recv_from_nghbr_ptr != nullptr)
                                ? recv_from_nghbr_ptr->index()
                                : 0;
        // perform send/recv
        if (comm_em) {
          comm::CommunicateField<M::Dim, 6>(domain.index(),
                                            domain.fields.em,
                                            send_idx,
                                            recv_idx,
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
                                              send_idx,
                                              recv_idx,
                                              send_rank,
                                              recv_rank,
                                              send_slice,
                                              recv_slice,
                                              comp_range_fld,
                                              false);
          }
        }
        if (comm_j || sync_j) {
          comm::CommunicateField<M::Dim, 3>(domain.index(),
                                            domain.fields.cur,
                                            send_idx,
                                            recv_idx,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_cur,
                                            sync_j);
        }
      }
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