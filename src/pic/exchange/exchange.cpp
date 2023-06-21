#include "wrapper.h"

#include "pic.h"

#include "io/output.h"
#include "meshblock/meshblock.h"

namespace ntt {
#ifdef MINKOWSKI_METRIC
  /**
   * @brief 1d periodic bc (minkowski).
   */
  template <>
  void PIC<Dim1>::Exchange(const GhostCells& quantity) {
    auto&         mblock = this->meshblock;
    const auto    i1min = mblock.i1_min(), i1max = mblock.i1_max(), ni1 = mblock.Ni1();
    range_tuple_t i1_range_to;
    range_tuple_t i1_range_from;
    NTTHostErrorIf((mblock.boundaries[0][0] != BoundaryCondition::PERIODIC),
                   "1d minkowski only supports periodic boundaries");

    if ((quantity == GhostCells::fields) || (quantity == GhostCells::currents)) {
      for (auto dir1 { -1 }; dir1 < 2; ++dir1) {
        if (dir1 == -1) {
          i1_range_to   = range_tuple_t(i1min - N_GHOSTS, i1min);
          i1_range_from = range_tuple_t(i1max - N_GHOSTS, i1max);
        } else if (dir1 == 0) {
          continue;
        } else {
          i1_range_to   = range_tuple_t(i1max, i1max + N_GHOSTS);
          i1_range_from = range_tuple_t(i1min, i1min + N_GHOSTS);
        }
        if (quantity == GhostCells::fields) {
          Kokkos::deep_copy(Kokkos::subview(mblock.em, i1_range_to, Kokkos::ALL()),
                            Kokkos::subview(mblock.em, i1_range_from, Kokkos::ALL()));
        } else {
          Kokkos::deep_copy(Kokkos::subview(mblock.cur, i1_range_to, Kokkos::ALL()),
                            Kokkos::subview(mblock.cur, i1_range_from, Kokkos::ALL()));
        }
      }
    } else if (quantity == GhostCells::particles) {
      for (auto& species : mblock.particles) {
        Kokkos::parallel_for(
          "Exchange_particles", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni1;
            } else if (species.i1(p) >= (int)ni1) {
              species.i1(p) -= ni1;
            }
          });
      }
    }
  }

  /**
   * @brief 2d periodic bc.
   *
   */
  template <>
  void PIC<Dim2>::Exchange(const GhostCells& quantity) {
    auto&         mblock = this->meshblock;
    const auto    i1min = mblock.i1_min(), i1max = mblock.i1_max(), ni1 = mblock.Ni1();
    const auto    i2min = mblock.i2_min(), i2max = mblock.i2_max(), ni2 = mblock.Ni2();
    range_tuple_t i1_range_to, i2_range_to;
    range_tuple_t i1_range_from, i2_range_from;
    NTTHostErrorIf((mblock.boundaries[0][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[1][0] != BoundaryCondition::PERIODIC),
                   "2d minkowski only supports periodic boundaries");
    if ((quantity == GhostCells::fields) || (quantity == GhostCells::currents)) {
      for (auto dir1 { -1 }; dir1 < 2; ++dir1) {
        if (dir1 == -1) {
          i1_range_to   = range_tuple_t(i1min - N_GHOSTS, i1min);
          i1_range_from = range_tuple_t(i1max - N_GHOSTS, i1max);
        } else if (dir1 == 0) {
          i1_range_to   = range_tuple_t(i1min, i1max);
          i1_range_from = range_tuple_t(i1min, i1max);
        } else {
          i1_range_to   = range_tuple_t(i1max, i1max + N_GHOSTS);
          i1_range_from = range_tuple_t(i1min, i1min + N_GHOSTS);
        }
        for (auto dir2 { -1 }; dir2 < 2; ++dir2) {
          if (dir2 == -1) {
            i2_range_to   = range_tuple_t(i2min - N_GHOSTS, i2min);
            i2_range_from = range_tuple_t(i2max - N_GHOSTS, i2max);
          } else if (dir2 == 0) {
            i2_range_to   = range_tuple_t(i2min, i2max);
            i2_range_from = range_tuple_t(i2min, i2max);
          } else {
            i2_range_to   = range_tuple_t(i2max, i2max + N_GHOSTS);
            i2_range_from = range_tuple_t(i2min, i2min + N_GHOSTS);
          }
          if (dir1 == 0 && dir2 == 0) {
            continue;
          }
          if (quantity == GhostCells::fields) {
            Kokkos::deep_copy(
              Kokkos::subview(mblock.em, i1_range_to, i2_range_to, Kokkos::ALL()),
              Kokkos::subview(mblock.em, i1_range_from, i2_range_from, Kokkos::ALL()));
          } else {
            Kokkos::deep_copy(
              Kokkos::subview(mblock.cur, i1_range_to, i2_range_to, Kokkos::ALL()),
              Kokkos::subview(mblock.cur, i1_range_from, i2_range_from, Kokkos::ALL()));
          }
        }
      }
    } else if (quantity == GhostCells::particles) {
      for (auto& species : mblock.particles) {
        Kokkos::parallel_for(
          "Exchange_particles", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni1;
            } else if (species.i1(p) >= (int)ni1) {
              species.i1(p) -= ni1;
            }
            if (species.i2(p) < 0) {
              species.i2(p) += ni2;
            } else if (species.i2(p) >= (int)ni2) {
              species.i2(p) -= ni2;
            }
          });
      }
    }
  }

  /**
   * @brief 3d periodic bc.
   *
   */
  template <>
  void PIC<Dim3>::Exchange(const GhostCells& quantity) {
    auto&         mblock = this->meshblock;
    const auto    i1min = mblock.i1_min(), i1max = mblock.i1_max(), ni1 = mblock.Ni1();
    const auto    i2min = mblock.i2_min(), i2max = mblock.i2_max(), ni2 = mblock.Ni2();
    const auto    i3min = mblock.i3_min(), i3max = mblock.i3_max(), ni3 = mblock.Ni3();
    range_tuple_t i1_range_to, i2_range_to, i3_range_to;
    range_tuple_t i1_range_from, i2_range_from, i3_range_from;
    NTTHostErrorIf((mblock.boundaries[0][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[1][0] != BoundaryCondition::PERIODIC)
                     || (mblock.boundaries[2][0] != BoundaryCondition::PERIODIC),
                   "3d minkowski only supports periodic boundaries");

    if ((quantity == GhostCells::fields) || (quantity == GhostCells::currents)) {
      for (auto dir1 { -1 }; dir1 < 2; ++dir1) {
        if (dir1 == -1) {
          i1_range_to   = range_tuple_t(i1min - N_GHOSTS, i1min);
          i1_range_from = range_tuple_t(i1max - N_GHOSTS, i1max);
        } else if (dir1 == 0) {
          i1_range_to   = range_tuple_t(i1min, i1max);
          i1_range_from = range_tuple_t(i1min, i1max);
        } else {
          i1_range_to   = range_tuple_t(i1max, i1max + N_GHOSTS);
          i1_range_from = range_tuple_t(i1min, i1min + N_GHOSTS);
        }
        for (auto dir2 { -1 }; dir2 < 2; ++dir2) {
          if (dir2 == -1) {
            i2_range_to   = range_tuple_t(i2min - N_GHOSTS, i2min);
            i2_range_from = range_tuple_t(i2max - N_GHOSTS, i2max);
          } else if (dir2 == 0) {
            i2_range_to   = range_tuple_t(i2min, i2max);
            i2_range_from = range_tuple_t(i2min, i2max);
          } else {
            i2_range_to   = range_tuple_t(i2max, i2max + N_GHOSTS);
            i2_range_from = range_tuple_t(i2min, i2min + N_GHOSTS);
          }
          for (auto dir3 { -1 }; dir3 < 2; ++dir3) {
            if (dir3 == -1) {
              i3_range_to   = range_tuple_t(i3min - N_GHOSTS, i3min);
              i3_range_from = range_tuple_t(i3max - N_GHOSTS, i3max);
            } else if (dir2 == 0) {
              i3_range_to   = range_tuple_t(i3min, i3max);
              i3_range_from = range_tuple_t(i3min, i3max);
            } else {
              i3_range_to   = range_tuple_t(i3max, i3max + N_GHOSTS);
              i3_range_from = range_tuple_t(i3min, i3min + N_GHOSTS);
            }
            if (dir1 == 0 && dir2 == 0 && dir3 == 0) {
              continue;
            }
            if (quantity == GhostCells::fields) {
              Kokkos::deep_copy(
                Kokkos::subview(
                  mblock.em, i1_range_to, i2_range_to, i3_range_to, Kokkos::ALL()),
                Kokkos::subview(
                  mblock.em, i1_range_from, i2_range_from, i3_range_from, Kokkos::ALL()));
            } else {
              Kokkos::deep_copy(
                Kokkos::subview(
                  mblock.cur, i1_range_to, i2_range_to, i3_range_to, Kokkos::ALL()),
                Kokkos::subview(
                  mblock.cur, i1_range_from, i2_range_from, i3_range_from, Kokkos::ALL()));
            }
          }
        }
      }
    } else if (quantity == GhostCells::particles) {
      for (auto& species : mblock.particles) {
        Kokkos::parallel_for(
          "Exchange_particles", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni1;
            } else if (species.i1(p) >= (int)ni1) {
              species.i1(p) -= ni1;
            }
            if (species.i2(p) < 0) {
              species.i2(p) += ni2;
            } else if (species.i2(p) >= (int)ni2) {
              species.i2(p) -= ni2;
            }
            if (species.i3(p) < 0) {
              species.i3(p) += ni2;
            } else if (species.i3(p) >= (int)ni3) {
              species.i3(p) -= ni3;
            }
          });
      }
    }
  }
#else
  template <Dimension D>
  void PIC<D>::Exchange(const GhostCells&) {}
#endif
}    // namespace ntt

#ifndef MINKOWSKI_METRIC
template void ntt::PIC<ntt::Dim1>::Exchange(const ntt::GhostCells&);
template void ntt::PIC<ntt::Dim2>::Exchange(const ntt::GhostCells&);
template void ntt::PIC<ntt::Dim3>::Exchange(const ntt::GhostCells&);
#endif