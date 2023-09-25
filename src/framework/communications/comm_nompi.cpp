#ifndef MPI_ENABLED
  #include "wrapper.h"

  #include "simulation.h"

  #include "meshblock/fields.h"
  #include "meshblock/meshblock.h"

  #include <vector>

namespace ntt {

  /* -------------------------------------------------------------------------- */
  /*                    Single meshblock self-communications */
  /* -------------------------------------------------------------------------- */

  #ifdef MINKOWSKI_METRIC

  // helper function
  template <Dimension D, int N>
  auto CommunicateField(const ndfield_t<D, N>&            fld,
                        const std::vector<range_tuple_t>& range_to,
                        const std::vector<range_tuple_t>& range_from,
                        const range_tuple_t&              comps) -> void {
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

  template <Dimension D>
  auto SyncCurrents(const ndfield_t<D, 3>&            cur,
                    const std::vector<range_tuple_t>& range_to,
                    const std::vector<range_tuple_t>& range_from) -> void {
    if constexpr (D == Dim1) {
      const auto offset_x1 = (long int)(range_from[0].first) -
                             (long int)(range_to[0].first);
      Kokkos::parallel_for(
        "CommCurrents",
        CreateRangePolicy<Dim1>({ range_to[0].first }, { range_to[0].second }),
        Lambda(index_t i1) {
    #pragma unroll
          for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
            cur(i1, comp) += cur(i1 + offset_x1, comp);
          }
        });
    } else if constexpr (D == Dim2) {
      const auto offset_x1 = (long int)(range_from[0].first) -
                             (long int)(range_to[0].first);
      const auto offset_x2 = (long int)(range_from[1].first) -
                             (long int)(range_to[1].first);
      Kokkos::parallel_for(
        "CommCurrents",
        CreateRangePolicy<Dim2>({ range_to[0].first, range_to[1].first },
                                { range_to[0].second, range_to[1].second }),
        Lambda(index_t i1, index_t i2) {
    #pragma unroll
          for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
            cur(i1, i2, comp) += cur(i1 + offset_x1, i2 + offset_x2, comp);
          }
        });
    } else if constexpr (D == Dim3) {
      auto offset_x1 = (long int)(range_from[0].first) -
                       (long int)(range_to[0].first);
      auto offset_x2 = (long int)(range_from[1].first) -
                       (long int)(range_to[1].first);
      auto offset_x3 = (long int)(range_from[2].first) -
                       (long int)(range_to[2].first);
      Kokkos::parallel_for(
        "CommCurrents",
        CreateRangePolicy<Dim3>(
          { range_to[0].first, range_to[1].first, range_to[2].first },
          { range_to[0].second, range_to[1].second, range_to[2].second }),
        Lambda(index_t i1, index_t i2, index_t i3) {
    #pragma unroll
          for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
            cur(i1, i2, i3, comp) += cur(i1 + offset_x1,
                                         i2 + offset_x2,
                                         i3 + offset_x3,
                                         comp);
          }
        });
    }
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::CurrentsSynchronize() {
    auto& mblock = this->meshblock;
    for (auto& bcs : mblock.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf(bc != BoundaryCondition::PERIODIC,
                       "Minkowski only supports periodic boundaries");
      }
    }
    for (auto& direction : Directions<D>::all) {
      auto range_to   = std::vector<range_tuple_t> {};
      auto range_from = std::vector<range_tuple_t> {};
      NTTHostErrorIf(direction.size() != (std::size_t)D, "Wrong direction size");

      for (short d { 0 }; d < (short)(direction.size()); ++d) {
        const auto dir = direction[d];
        if (dir == -1) {
          range_to.emplace_back(mblock.i_min(d), mblock.i_min(d) + N_GHOSTS);
          range_from.emplace_back(mblock.i_max(d), mblock.i_max(d) + N_GHOSTS);
        } else if (dir == 0) {
          range_to.emplace_back(mblock.i_min(d), mblock.i_max(d));
          range_from.emplace_back(mblock.i_min(d), mblock.i_max(d));
        } else if (dir == 1) {
          range_to.emplace_back(mblock.i_max(d) - N_GHOSTS, mblock.i_max(d));
          range_from.emplace_back(mblock.i_min(d) - N_GHOSTS, mblock.i_min(d));
        } else {
          NTTHostError("Wrong direction");
        }
      }
      SyncCurrents<D>(mblock.cur, range_to, range_from);
    }
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Communicate(CommTags comm) {
    auto& mblock = this->meshblock;
    NTTHostErrorIf(comm == Comm_None, "Communicate called with Comm_None");
    for (auto& bcs : mblock.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf(bc != BoundaryCondition::PERIODIC,
                       "Minkowski only supports periodic boundaries");
      }
    }
    NTTHostErrorIf((comm & Comm_D) || (comm & Comm_H) || (comm & Comm_D0) ||
                     (comm & Comm_B0),
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
      if ((params()->shuffleInterval() > 0) &&
          (tstep() % params()->shuffleInterval() == 0)) {
        for (auto& species : mblock.particles) {
          species.ReshuffleByTags();
        }
      }
    }
    NTTLog();
  }

  #else  // not MINKOWSKI_METRIC

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::CurrentsSynchronize() {}

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Communicate(CommTags comm) {
    if (comm & Comm_Prtl) {
      if ((params()->shuffleInterval() > 0) &&
          (tstep() % params()->shuffleInterval() == 0)) {
        for (auto& species : this->meshblock.particles) {
          species.ReshuffleByTags();
        }
      }
    }
    NTTLog();
  }
  #endif // MINKOWSKI_METRIC

} // namespace ntt

  #ifdef MINKOWSKI_METRIC
template void ntt::CommunicateField<ntt::Dim1, 3>(
  const ntt::ndfield_t<ntt::Dim1, 3>&,
  const std::vector<ntt::range_tuple_t>&,
  const std::vector<ntt::range_tuple_t>&,
  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim2, 3>(
  const ntt::ndfield_t<ntt::Dim2, 3>&,
  const std::vector<ntt::range_tuple_t>&,
  const std::vector<ntt::range_tuple_t>&,
  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim3, 3>(
  const ntt::ndfield_t<ntt::Dim3, 3>&,
  const std::vector<ntt::range_tuple_t>&,
  const std::vector<ntt::range_tuple_t>&,
  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim1, 6>(
  const ntt::ndfield_t<ntt::Dim1, 6>&,
  const std::vector<ntt::range_tuple_t>&,
  const std::vector<ntt::range_tuple_t>&,
  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim2, 6>(
  const ntt::ndfield_t<ntt::Dim2, 6>&,
  const std::vector<ntt::range_tuple_t>&,
  const std::vector<ntt::range_tuple_t>&,
  const ntt::range_tuple_t&);

template void ntt::CommunicateField<ntt::Dim3, 6>(
  const ntt::ndfield_t<ntt::Dim3, 6>&,
  const std::vector<ntt::range_tuple_t>&,
  const std::vector<ntt::range_tuple_t>&,
  const ntt::range_tuple_t&);
  #endif

#endif // MPI_ENABLED