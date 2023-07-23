#include "wrapper.h"

#include "simulation.h"

#include "meshblock/fields.h"
#include "meshblock/meshblock.h"

#ifdef MPI_ENABLED
#  include <mpi.h>
#endif

namespace ntt {
#ifndef MPI_ENABLED
  // Single meshblock self-synchronization of currents

#  ifdef MINKOWSKI_METRIC
  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::CurrentsSynchronize() {
    auto& mblock = this->meshblock;
    for (auto& bcs : mblock.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf((bc != BoundaryCondition::PERIODIC),
                       "Minkowski only supports periodic boundaries");
      }
    }
    if constexpr (S == GRPICEngine) {
      NTTHostError("Wrong currents synchronize call");
    } else {
      for (auto& direction : Directions<D>::unique) {
        tuple_t<std::size_t, D> range_min;
        tuple_t<std::size_t, D> range_max;
        NTTHostErrorIf(direction.size() != (std::size_t)D, "Wrong direction size");
        for (short d { 0 }; d < (short)(direction.size()); ++d) {
          const auto dir = direction[d];
          if ((dir == 1) || (dir == -1)) {
            range_min[d] = 0;
            range_max[d] = N_GHOSTS;
          } else if (dir == 0) {
            range_min[d] = mblock.i_min(d);
            range_max[d] = mblock.i_max(d);
          }
        }
        auto        range = CreateRangePolicy<D>(range_min, range_max);

        std::size_t I1L1 { 0 }, I1L2 { 0 }, I1R1 { 0 }, I1R2 { 0 };
        std::size_t I2L1 { 0 }, I2L2 { 0 }, I2R1 { 0 }, I2R2 { 0 };
        std::size_t I3L1 { 0 }, I3L2 { 0 }, I3R1 { 0 }, I3R2 { 0 };

        if constexpr (D == Dim1 || D == Dim2 || D == Dim3) {
          if (direction[0] != 0) {
            I1L1 = mblock.i1_min();
            I1R1 = mblock.i1_max();
            I1L2 = mblock.i1_max() - N_GHOSTS;
            I1R2 = mblock.i1_min() - N_GHOSTS;
          }
          if (direction[0] < 0) {
            std::swap(I1L1, I1L2);
            std::swap(I1R1, I1R2);
          }
        }
        if constexpr (D == Dim2 || D == Dim3) {
          if (direction[1] != 0) {
            I2L1 = mblock.i2_min();
            I2R1 = mblock.i2_max();
            I2L2 = mblock.i2_max() - N_GHOSTS;
            I2R2 = mblock.i2_min() - N_GHOSTS;
          }
          if (direction[1] < 0) {
            std::swap(I2L1, I2L2);
            std::swap(I2R1, I2R2);
          }
        }
        if constexpr (D == Dim3) {
          if (direction[2] != 0) {
            I3L1 = mblock.i3_min();
            I3R1 = mblock.i3_max();
            I3L2 = mblock.i3_max() - N_GHOSTS;
            I3R2 = mblock.i3_min() - N_GHOSTS;
          }
          if (direction[2] < 0) {
            std::swap(I3L1, I3L2);
            std::swap(I3R1, I3R2);
          }
        }
        if constexpr (D == Dim1) {
          Kokkos::parallel_for(
            "CurrentsSynchronize", range, Lambda(index_t i1) {
#    pragma unroll
              for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
                mblock.cur(I1L1 + i1, comp) += mblock.cur(I1R1 + i1, comp);
                mblock.cur(I1L2 + i1, comp) += mblock.cur(I1R2 + i1, comp);
              }
            });
        } else if constexpr (D == Dim2) {
          Kokkos::parallel_for(
            "CurrentsSynchronize", range, Lambda(index_t i1, index_t i2) {
#    pragma unroll
              for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
                mblock.cur(I1L1 + i1, I2L1 + i2, comp)
                  += mblock.cur(I1R1 + i1, I2R1 + i2, comp);
                mblock.cur(I1L2 + i1, I2L2 + i2, comp)
                  += mblock.cur(I1R2 + i1, I2R2 + i2, comp);
              }
            });
        } else if constexpr (D == Dim3) {
          Kokkos::parallel_for(
            "CurrentsSynchronize", range, Lambda(index_t i1, index_t i2, index_t i3) {
#    pragma unroll
              for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
                mblock.cur(I1L1 + i1, I2L1 + i2, I3L1 + i3, comp)
                  += mblock.cur(I1R1 + i1, I2R1 + i2, I3R1 + i3, comp);
                mblock.cur(I1L2 + i1, I2L2 + i2, I3L2 + i3, comp)
                  += mblock.cur(I1R2 + i1, I2R2 + i2, I3R2 + i3, comp);
              }
            });
        }
      }
    }
  }
#  else     // not MINKOWSKI_METRIC
  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::CurrentsSynchronize() {
    // no cross-meshblock current synchronization necessary
  }

#  endif    // MINKOWSKI_METRIC

#endif      // MPI_ENABLED

}    // namespace ntt