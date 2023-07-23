#include "wrapper.h"

#include "simulation.h"

#include "meshblock/fields.h"
#include "meshblock/meshblock.h"

#ifdef MPI_ENABLED
#  include <mpi.h>
#endif

namespace ntt {

#ifndef MPI_ENABLED
  /* -------------------------------------------------------------------------- */
  /*                    Single meshblock self-communications                    */
  /* -------------------------------------------------------------------------- */

#  ifdef MINKOWSKI_METRIC
  // helper function
  template <Dimension D, int N>
  void CommunicateField(const ndfield_t<D, N>&            fld,
                        const std::vector<range_tuple_t>& range_to,
                        const std::vector<range_tuple_t>& range_from,
                        const range_tuple_t&              comps) {
    if constexpr (D == Dim1) {
      Kokkos::deep_copy(Kokkos::subview(fld, range_to[0], comps),
                        Kokkos::subview(fld, range_from[0], comps));
    } else if constexpr (D == Dim2) {
      Kokkos::deep_copy(Kokkos::subview(fld, range_to[0], range_to[1], comps),
                        Kokkos::subview(fld, range_from[0], range_from[1], comps));
    } else if constexpr (D == Dim3) {
      Kokkos::deep_copy(
        Kokkos::subview(fld, range_to[0], range_to[1], range_to[2], comps),
        Kokkos::subview(fld, range_from[0], range_from[1], range_from[2], comps));
    }
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Communicate(CommTags comm) {
    auto& mblock = this->meshblock;

    if constexpr (S == GRPICEngine) {
      NTTHostError("Wrong communicate call");
    }
    NTTHostErrorIf((comm == Comm_None), "Communicate called with Comm_None");
    for (auto& bcs : mblock.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf((bc != BoundaryCondition::PERIODIC),
                       "Minkowski only supports periodic boundaries");
      }
    }
    NTTHostErrorIf((comm & Comm_D) || (comm & Comm_H) || (comm & Comm_D0) || (comm & Comm_B0),
                   "SR only supports E, B, J, and particles in Communicate");

    if ((comm & Comm_E) || (comm & Comm_B) || (comm & Comm_J)) {
      for (auto& direction : Directions<D>::all) {
        auto range_to       = std::vector<range_tuple_t> {};
        auto range_from     = std::vector<range_tuple_t> {};
        auto comp_range_fld = range_tuple_t {};
        auto comp_range_cur = range_tuple_t {};
        if ((comm & Comm_E) && (comm & Comm_B)) {
          comp_range_fld = range_tuple_t(em::ex1, em::bx3 + 1);
        } else if (comm & Comm_E) {
          comp_range_fld = range_tuple_t(em::ex1, em::ex3 + 1);
        } else if (comm & Comm_B) {
          comp_range_fld = range_tuple_t(em::bx1, em::bx3 + 1);
        }
        if (comm & Comm_J) {
          comp_range_cur = range_tuple_t(cur::jx1, cur::jx3 + 1);
        }
        NTTHostErrorIf(direction.size() != (std::size_t)D, "Wrong direction size");

        for (short d { 0 }; d < (short)(direction.size()); ++d) {
          const auto dir = direction[d];
          if (dir == -1) {
            range_to.emplace_back(mblock.i_min(d) - N_GHOSTS, mblock.i_min(d));
            range_from.emplace_back(mblock.i_max(d) - N_GHOSTS, mblock.i_max(d));
          } else if (dir == 0) {
            range_to.emplace_back(mblock.i_min(d), mblock.i_max(d));
            range_from.emplace_back(mblock.i_min(d), mblock.i_max(d));
          } else if (dir == 1) {
            range_to.emplace_back(mblock.i_max(d), mblock.i_max(d) + N_GHOSTS);
            range_from.emplace_back(mblock.i_min(d), mblock.i_min(d) + N_GHOSTS);
          } else {
            NTTHostError("Wrong direction");
          }
        }

        if ((comm & Comm_E) || (comm & Comm_B)) {
          CommunicateField<D, 6>(mblock.em, range_to, range_from, comp_range_fld);
        }
        if (comm & Comm_J) {
          CommunicateField<D, 3>(mblock.cur, range_to, range_from, comp_range_cur);
        }
      }
    }

    if (comm & Comm_Prtl) {
      for (auto& species : mblock.particles) {
        if constexpr (D == Dim1) {
          const auto ni1 = mblock.Ni1();
          Kokkos::parallel_for(
            "Exchange_particles", species.rangeActiveParticles(), Lambda(index_t p) {
              species.i1(p) += ni1 * static_cast<int>(species.i1(p) < 0)
                               - ni1 * static_cast<int>(species.i1(p) >= (int)ni1);
            });
        } else if constexpr (D == Dim2) {
          const auto ni1 = mblock.Ni1(), ni2 = mblock.Ni2();
          Kokkos::parallel_for(
            "Exchange_particles", species.rangeActiveParticles(), Lambda(index_t p) {
              species.i1(p) += ni1 * static_cast<int>(species.i1(p) < 0)
                               - ni1 * static_cast<int>(species.i1(p) >= (int)ni1);
              species.i2(p) += ni2 * static_cast<int>(species.i2(p) < 0)
                               - ni2 * static_cast<int>(species.i2(p) >= (int)ni2);
            });
        } else if constexpr (D == Dim3) {
          const auto ni1 = mblock.Ni1(), ni2 = mblock.Ni2(), ni3 = mblock.Ni3();
          Kokkos::parallel_for(
            "Exchange_particles", species.rangeActiveParticles(), Lambda(index_t p) {
              species.i1(p) += ni1 * static_cast<int>(species.i1(p) < 0)
                               - ni1 * static_cast<int>(species.i1(p) >= (int)ni1);
              species.i2(p) += ni2 * static_cast<int>(species.i2(p) < 0)
                               - ni2 * static_cast<int>(species.i2(p) >= (int)ni2);
              species.i3(p) += ni3 * static_cast<int>(species.i3(p) < 0)
                               - ni3 * static_cast<int>(species.i3(p) >= (int)ni3);
            });
        }
      }
    }
  }
#  else     // not MINKOWSKI_METRIC
  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Communicate(CommTags) {
    // no single-meshblock communication necessary
  }
#  endif    // MINKOWSKI_METRIC

#else       // not MPI_ENABLED
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
    constexpr auto mpi_real_t { std::is_same_v<real_t, double> ? MPI_DOUBLE : MPI_FLOAT };
    std::size_t    nsend { comps.second - comps.first }, nrecv { comps.second - comps.first };
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
                   mpi_real_t,
                   send_to->mpiRank(),
                   0,
                   recv_fld.data(),
                   nrecv,
                   mpi_real_t,
                   recv_from->mpiRank(),
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    } else if (send_to != nullptr) {
      MPI_Send(send_fld.data(), nsend, mpi_real_t, send_to->mpiRank(), 0, MPI_COMM_WORLD);
    } else if (recv_from != nullptr) {
      MPI_Recv(recv_fld.data(),
               nrecv,
               mpi_real_t,
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
  }

#endif      // MPI_ENABLED
}    // namespace ntt

#ifndef MPI_ENABLED
#  ifdef MINKOWSKI_METRIC
template void ntt::CommunicateField<ntt::Dim1, 3>(const ntt::ndfield_t<ntt::Dim1, 3>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim2, 3>(const ntt::ndfield_t<ntt::Dim2, 3>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim3, 3>(const ntt::ndfield_t<ntt::Dim3, 3>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim1, 6>(const ntt::ndfield_t<ntt::Dim1, 6>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim2, 6>(const ntt::ndfield_t<ntt::Dim2, 6>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim3, 6>(const ntt::ndfield_t<ntt::Dim3, 6>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const std::vector<ntt::range_tuple_t>&,
                                                  const ntt::range_tuple_t&);
#  endif
#else    // MPI_ENABLED

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