#ifdef MPI_ENABLED

#  include "wrapper.h"

#  include "simulation.h"

#  include "communications/metadomain.h"
#  include "meshblock/fields.h"
#  include "meshblock/meshblock.h"

#  include <mpi.h>

namespace ntt {

  /* -------------------------------------------------------------------------- */
  /*                     Cross-meshblock MPI communications                     */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, int N>
  void CommunicateField(ndfield_t<D, N>&                  fld,
                        const Domain<D>*                  send_to,
                        const Domain<D>*                  recv_from,
                        const std::vector<range_tuple_t>& send_slice,
                        const std::vector<range_tuple_t>& recv_slice,
                        const range_tuple_t&              comps) {
    std::size_t nsend { comps.second - comps.first }, nrecv { comps.second - comps.first };
    ndarray_t<(short)D + 1> send_fld, recv_fld;
    for (short d { 0 }; d < (short)D; ++d) {
      if (send_to != nullptr) {
        nsend *= (send_slice[d].second - send_slice[d].first);
      }
      if (recv_from != nullptr) {
        nrecv *= (recv_slice[d].second - recv_slice[d].first);
      }
    }

    if (send_to != nullptr) {
      if constexpr (D == Dim1) {
        send_fld = ndarray_t<2>(
          "send_fld", send_slice[0].second - send_slice[0].first, comps.second - comps.first);
        Kokkos::deep_copy(send_fld, Kokkos::subview(fld, send_slice[0], comps));
      } else if constexpr (D == Dim2) {
        send_fld = ndarray_t<3>("send_fld",
                                send_slice[0].second - send_slice[0].first,
                                send_slice[1].second - send_slice[1].first,
                                comps.second - comps.first);
        Kokkos::deep_copy(send_fld, Kokkos::subview(fld, send_slice[0], send_slice[1], comps));
      } else if constexpr (D == Dim3) {
        send_fld = ndarray_t<4>("send_fld",
                                send_slice[0].second - send_slice[0].first,
                                send_slice[1].second - send_slice[1].first,
                                send_slice[2].second - send_slice[2].first,
                                comps.second - comps.first);
        Kokkos::deep_copy(
          send_fld, Kokkos::subview(fld, send_slice[0], send_slice[1], send_slice[2], comps));
      }
    }
    if (recv_from != nullptr) {
      if constexpr (D == Dim1) {
        recv_fld = ndarray_t<2>(
          "recv_fld", recv_slice[0].second - recv_slice[0].first, comps.second - comps.first);
      } else if constexpr (D == Dim2) {
        recv_fld = ndarray_t<3>("recv_fld",
                                recv_slice[0].second - recv_slice[0].first,
                                recv_slice[1].second - recv_slice[1].first,
                                comps.second - comps.first);
      } else if constexpr (D == Dim3) {
        recv_fld = ndarray_t<4>("recv_fld",
                                recv_slice[0].second - recv_slice[0].first,
                                recv_slice[1].second - recv_slice[1].first,
                                recv_slice[2].second - recv_slice[2].first,
                                comps.second - comps.first);
      }
    }
    if (send_to != nullptr && recv_from != nullptr) {
      MPI_Sendrecv(send_fld.data(),
                   nsend,
                   mpi_get_type<real_t>(),
                   send_to->mpiRank(),
                   0,
                   recv_fld.data(),
                   nrecv,
                   mpi_get_type<real_t>(),
                   recv_from->mpiRank(),
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if (send_to != nullptr) {
      MPI_Send(
        send_fld.data(), nsend, mpi_get_type<real_t>(), send_to->mpiRank(), 0, MPI_COMM_WORLD);
    } else if (recv_from != nullptr) {
      MPI_Recv(recv_fld.data(),
               nrecv,
               mpi_get_type<real_t>(),
               recv_from->mpiRank(),
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else {
      NTTHostError("CommunicateField called with nullptrs");
    }
    if (recv_from != nullptr) {
      if constexpr (D == Dim1) {
        Kokkos::deep_copy(Kokkos::subview(fld, recv_slice[0], comps), recv_fld);
      } else if constexpr (D == Dim2) {
        Kokkos::deep_copy(Kokkos::subview(fld, recv_slice[0], recv_slice[1], comps), recv_fld);
      } else if constexpr (D == Dim3) {
        Kokkos::deep_copy(
          Kokkos::subview(fld, recv_slice[0], recv_slice[1], recv_slice[2], comps), recv_fld);
      }
    }
  }

  template <Dimension D, typename T>
  void CommunicateParticleQuantity(array_t<T*>&         arr,
                                   const Domain<D>*     send_to,
                                   const Domain<D>*     recv_from,
                                   const range_tuple_t& send_slice,
                                   const range_tuple_t& recv_slice) {
    const auto send_count = send_slice.second - send_slice.first;
    const auto recv_count = recv_slice.second - recv_slice.first;
    if ((send_to != nullptr) && (recv_from != nullptr) && (send_count > 0)
        && (recv_count > 0)) {
      MPI_Sendrecv(arr.data() + send_slice.first,
                   send_count,
                   mpi_get_type<T>(),
                   send_to->mpiRank(),
                   0,
                   arr.data() + recv_slice.first,
                   recv_count,
                   mpi_get_type<T>(),
                   recv_from->mpiRank(),
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if ((send_to != nullptr) && (send_count > 0)) {
      MPI_Send(arr.data() + send_slice.first,
               send_count,
               mpi_get_type<T>(),
               send_to->mpiRank(),
               0,
               MPI_COMM_WORLD);
    } else if ((recv_from != nullptr) && (recv_count > 0)) {
      MPI_Recv(arr.data() + recv_slice.first,
               recv_count,
               mpi_get_type<T>(),
               recv_from->mpiRank(),
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }

  template <Dimension D>
  void ParticleSendRecvCount(const Domain<D>*   send_to,
                             const Domain<D>*   recv_from,
                             const std::size_t& send_count,
                             std::size_t&       recv_count) {
    if ((send_to != nullptr) && (recv_from != nullptr)) {
      MPI_Sendrecv(&send_count,
                   1,
                   mpi_get_type<std::size_t>(),
                   send_to->mpiRank(),
                   0,
                   &recv_count,
                   1,
                   mpi_get_type<std::size_t>(),
                   recv_from->mpiRank(),
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if (send_to != nullptr) {
      MPI_Send(
        &send_count, 1, mpi_get_type<std::size_t>(), send_to->mpiRank(), 0, MPI_COMM_WORLD);
    } else if (recv_from != nullptr) {
      MPI_Recv(&recv_count,
               1,
               mpi_get_type<std::size_t>(),
               recv_from->mpiRank(),
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    } else {
      NTTHostError("ParticleSendRecvCount called with nullptrs");
    }
  }

  template <Dimension D, SimulationEngine S>
  void CommunicateParticles(Particles<D, S>&      particles,
                            const direction_t<D>& dir,
                            const Domain<D>*      local,
                            const Domain<D>*      send_to,
                            const Domain<D>*      recv_from,
                            const range_tuple_t&  send_slice,
                            std::size_t&          index_last) {
    if ((send_to == nullptr) && (recv_from == nullptr)) {
      NTTHostError("No send or recv in CommunicateParticles");
    }
    const auto  send_count { send_slice.second - send_slice.first };
    std::size_t recv_count;
    ParticleSendRecvCount(send_to, recv_from, send_count, recv_count);

    NTTHostErrorIf((index_last + recv_count) >= particles.maxnpart(),
                   "Too many particles to receive (cannot fit into maxptl)");
    const auto recv_slice = range_tuple_t({ index_last, index_last + recv_count });

    CommunicateParticleQuantity(particles.i1, send_to, recv_from, send_slice, recv_slice);
    CommunicateParticleQuantity(particles.dx1, send_to, recv_from, send_slice, recv_slice);
    if constexpr (S == GRPICEngine) {
      CommunicateParticleQuantity(
        particles.i1_prev, send_to, recv_from, send_slice, recv_slice);
      CommunicateParticleQuantity(
        particles.dx1_prev, send_to, recv_from, send_slice, recv_slice);
    }
    if constexpr (D == Dim2 || D == Dim3) {
      CommunicateParticleQuantity(particles.i2, send_to, recv_from, send_slice, recv_slice);
      CommunicateParticleQuantity(particles.dx2, send_to, recv_from, send_slice, recv_slice);
      if constexpr (S == GRPICEngine) {
        CommunicateParticleQuantity(
          particles.i2_prev, send_to, recv_from, send_slice, recv_slice);
        CommunicateParticleQuantity(
          particles.dx2_prev, send_to, recv_from, send_slice, recv_slice);
      }
    }
    if constexpr (D == Dim3) {
      CommunicateParticleQuantity(particles.i3, send_to, recv_from, send_slice, recv_slice);
      CommunicateParticleQuantity(particles.dx3, send_to, recv_from, send_slice, recv_slice);
      if constexpr (S == GRPICEngine) {
        CommunicateParticleQuantity(
          particles.i3_prev, send_to, recv_from, send_slice, recv_slice);
        CommunicateParticleQuantity(
          particles.dx3_prev, send_to, recv_from, send_slice, recv_slice);
      }
    }
    CommunicateParticleQuantity(particles.ux1, send_to, recv_from, send_slice, recv_slice);
    CommunicateParticleQuantity(particles.ux2, send_to, recv_from, send_slice, recv_slice);
    CommunicateParticleQuantity(particles.ux3, send_to, recv_from, send_slice, recv_slice);
    CommunicateParticleQuantity(particles.weight, send_to, recv_from, send_slice, recv_slice);
    if constexpr (D == Dim2) {
#  ifndef MINKOWSKI_METRIC
      CommunicateParticleQuantity(particles.phi, send_to, recv_from, send_slice, recv_slice);
#  endif
    }
    if (recv_count > 0) {
      if constexpr (D == Dim1) {
        int shift_in_x1 { 0 };
        if ((-dir)[0] == -1) {
          shift_in_x1 = -recv_from->ncells()[0];
        } else if ((-dir)[0] == 1) {
          shift_in_x1 = local->ncells()[0];
        }
        Kokkos::parallel_for(
          "CommunicateParticles", recv_count, Lambda(index_t p) {
            particles.tag(index_last + p) = ParticleTag::alive;
            particles.i1(index_last + p) += shift_in_x1;
          });
      } else if constexpr (D == Dim2) {
        int shift_in_x1 { 0 }, shift_in_x2 { 0 };
        if ((-dir)[0] == -1) {
          shift_in_x1 = -recv_from->ncells()[0];
        } else if ((-dir)[0] == 1) {
          shift_in_x1 = local->ncells()[0];
        }
        if ((-dir)[1] == -1) {
          shift_in_x2 = -recv_from->ncells()[1];
        } else if ((-dir)[1] == 1) {
          shift_in_x2 = local->ncells()[1];
        }
        Kokkos::parallel_for(
          "CommunicateParticles", recv_count, Lambda(index_t p) {
            particles.tag(index_last + p) = ParticleTag::alive;
            particles.i1(index_last + p) += shift_in_x1;
            particles.i2(index_last + p) += shift_in_x2;
          });
      } else if constexpr (D == Dim3) {
        int shift_in_x1 { 0 }, shift_in_x2 { 0 }, shift_in_x3 { 0 };
        if ((-dir)[0] == -1) {
          shift_in_x1 = -recv_from->ncells()[0];
        } else if ((-dir)[0] == 1) {
          shift_in_x1 = local->ncells()[0];
        }
        if ((-dir)[1] == -1) {
          shift_in_x2 = -recv_from->ncells()[1];
        } else if ((-dir)[1] == 1) {
          shift_in_x2 = local->ncells()[1];
        }
        if ((-dir)[2] == -1) {
          shift_in_x3 = -recv_from->ncells()[2];
        } else if ((-dir)[2] == 1) {
          shift_in_x3 = local->ncells()[2];
        }
        Kokkos::parallel_for(
          "CommunicateParticles", recv_count, Lambda(index_t p) {
            particles.tag(index_last + p) = ParticleTag::alive;
            particles.i1(index_last + p) += shift_in_x1;
            particles.i2(index_last + p) += shift_in_x2;
            particles.i3(index_last + p) += shift_in_x3;
          });
      }
      index_last += recv_count;
      particles.setNpart(index_last);
    }
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Communicate(CommTags comm) {
    NTTHostErrorIf((comm == Comm_None), "Communicate called with Comm_None");
    if constexpr (S != GRPICEngine) {
      NTTHostErrorIf((comm & Comm_D) || (comm & Comm_H) || (comm & Comm_D0) || (comm & Comm_B0),
                     "SR only supports E, B, J, and particles in Communicate");
    } else {
      NTTHostErrorIf((comm & Comm_E) || (comm & Comm_H),
                     "GR should not need to communicate E & H");
    }

    const auto local_domain = m_metadomain.localDomain();
    auto&      mblock       = this->meshblock;

    const auto comm_fields  = (comm & Comm_E) || (comm & Comm_B) || (comm & Comm_J)
                             || (comm & Comm_D) || (comm & Comm_D0) || (comm & Comm_B0);
    const auto comm_em  = (comm & Comm_E) || (comm & Comm_B) || (comm & Comm_D);
    const auto comm_em0 = (comm & Comm_B0) || (comm & Comm_D0);

    if (comm_fields) {
      auto comp_range_fld = range_tuple_t {};
      auto comp_range_cur = range_tuple_t {};
      if constexpr (S == GRPICEngine) {
        if (((comm & Comm_D) && (comm & Comm_B)) || ((comm & Comm_D0) && (comm & Comm_B0))) {
          comp_range_fld = range_tuple_t(em::dx1, em::bx3 + 1);
        } else if ((comm & Comm_D) || (comm & Comm_D0)) {
          comp_range_fld = range_tuple_t(em::dx1, em::dx3 + 1);
        } else if ((comm & Comm_B) || (comm & Comm_B0)) {
          comp_range_fld = range_tuple_t(em::bx1, em::bx3 + 1);
        }
      } else {
        if ((comm & Comm_E) && (comm & Comm_B)) {
          comp_range_fld = range_tuple_t(em::ex1, em::bx3 + 1);
        } else if (comm & Comm_E) {
          comp_range_fld = range_tuple_t(em::ex1, em::ex3 + 1);
        } else if (comm & Comm_B) {
          comp_range_fld = range_tuple_t(em::bx1, em::bx3 + 1);
        }
      }
      if (comm & Comm_J) {
        comp_range_cur = range_tuple_t(cur::jx1, cur::jx3 + 1);
      }

      for (auto& direction : Directions<D>::all) {
        const auto is_send = (local_domain->neighbors(direction) != nullptr);
        const auto is_recv = (local_domain->neighbors(-direction) != nullptr);
        if (!is_send && !is_recv) {
          continue;
        }
        auto send_slice = std::vector<range_tuple_t> {};
        auto recv_slice = std::vector<range_tuple_t> {};
        for (short d { 0 }; d < (short)(direction.size()); ++d) {
          const auto dir = direction[d];
          if (is_send) {
            if (dir == 0) {
              send_slice.emplace_back(mblock.i_min(d), mblock.i_max(d));
            } else if (dir == 1) {
              send_slice.emplace_back(mblock.i_max(d) - N_GHOSTS, mblock.i_max(d));
            } else {
              send_slice.emplace_back(mblock.i_min(d), mblock.i_min(d) + N_GHOSTS);
            }
          }
          if (is_recv) {
            if (-dir == 0) {
              recv_slice.emplace_back(mblock.i_min(d), mblock.i_max(d));
            } else if (-dir == 1) {
              recv_slice.emplace_back(mblock.i_max(d), mblock.i_max(d) + N_GHOSTS);
            } else {
              recv_slice.emplace_back(mblock.i_min(d) - N_GHOSTS, mblock.i_min(d));
            }
          }
        }
        if (comm_em) {
          CommunicateField<D, 6>(mblock.em,
                                 local_domain->neighbors(direction),
                                 local_domain->neighbors(-direction),
                                 send_slice,
                                 recv_slice,
                                 comp_range_fld);
        }
        if constexpr (S == GRPICEngine) {
          if (comm_em0) {
            CommunicateField<D, 6>(mblock.em0,
                                   local_domain->neighbors(direction),
                                   local_domain->neighbors(-direction),
                                   send_slice,
                                   recv_slice,
                                   comp_range_fld);
          }
        }
        if (comm & Comm_J) {
          CommunicateField<D, 3>(mblock.cur,
                                 local_domain->neighbors(direction),
                                 local_domain->neighbors(-direction),
                                 send_slice,
                                 recv_slice,
                                 comp_range_fld);
        }
      }
    }
    if (comm & Comm_Prtl) {
      for (auto& species : mblock.particles) {
        const auto npart_per_tag = species.ReshuffleByTags(true);
        /**
         *    alive        dead         tag1        tag2
         * [ 11111111   000000000    222222222    3333333 .... ]
         *                           ^        ^
         *                           |        |
         *     tag_offset[tag1] -----+        +----- tag_offset[tag1] + npart_per_tag[tag2]
         *          "send_pmin"                      "send_pmax" (after last element)
         */
        auto       tag_offset { npart_per_tag };
        for (std::size_t i { 1 }; i < tag_offset.size(); ++i) {
          tag_offset[i] += tag_offset[i - 1];
        }
        for (std::size_t i { 0 }; i < tag_offset.size(); ++i) {
          tag_offset[i] -= npart_per_tag[i];
        }
        auto index_last
          = tag_offset[tag_offset.size() - 1] + npart_per_tag[npart_per_tag.size() - 1];
        for (auto& direction : Directions<D>::all) {
          const auto is_send = (local_domain->neighbors(direction) != nullptr);
          const auto is_recv = (local_domain->neighbors(-direction) != nullptr);
          if (!is_send && !is_recv) {
            continue;
          }
          const auto send_dir_tag = PrtlSendTag<D>::dir2tag(direction);
          const auto send_pmin    = tag_offset[send_dir_tag];
          const auto send_pmax    = tag_offset[send_dir_tag] + npart_per_tag[send_dir_tag];

          CommunicateParticles(species,
                               direction,
                               local_domain,
                               local_domain->neighbors(direction),
                               local_domain->neighbors(-direction),
                               { send_pmin, send_pmax },
                               index_last);
        }
      }
    }
  }
}    // namespace ntt

template void ntt::CommunicateField<ntt::Dim1, 6>(ntt::ndfield_t<ntt::Dim1, 6>&,
                                                  const ntt::Domain<ntt::Dim1>*,
                                                  const ntt::Domain<ntt::Dim1>*,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim2, 6>(ntt::ndfield_t<ntt::Dim2, 6>&,
                                                  const ntt::Domain<ntt::Dim2>*,
                                                  const ntt::Domain<ntt::Dim2>*,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim3, 6>(ntt::ndfield_t<ntt::Dim3, 6>&,
                                                  const ntt::Domain<ntt::Dim3>*,
                                                  const ntt::Domain<ntt::Dim3>*,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim1, 3>(ntt::ndfield_t<ntt::Dim1, 3>&,
                                                  const ntt::Domain<ntt::Dim1>*,
                                                  const ntt::Domain<ntt::Dim1>*,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim2, 3>(ntt::ndfield_t<ntt::Dim2, 3>&,
                                                  const ntt::Domain<ntt::Dim2>*,
                                                  const ntt::Domain<ntt::Dim2>*,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim3, 3>(ntt::ndfield_t<ntt::Dim3, 3>&,
                                                  const ntt::Domain<ntt::Dim3>*,
                                                  const ntt::Domain<ntt::Dim3>*,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

#endif