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
#endif