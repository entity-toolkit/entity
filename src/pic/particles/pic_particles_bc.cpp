#include "global.h"
#include "pic.h"

#include <plog/Log.h>

#include <stdexcept>
#include <iostream>

namespace ntt {
  /**
   * @brief 1d periodic particle bc.
   *
   */
  template <>
  void PIC<Dimension::ONE_D>::particleBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      for (auto& species : m_mblock.particles) {
        auto ni {m_mblock.Ni1()};
        Kokkos::parallel_for(
          "prtl_bc", species.loopParticles(), Lambda(index_t p) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni;
            } else if (species.i1(p) >= ni) {
              species.i1(p) -= ni;
            }
          });
      }
    } else {
      for (auto& species : m_mblock.particles) {
        auto ni {m_mblock.Ni1()};
        Kokkos::parallel_for(
          "prtl_bc", species.loopParticles(), Lambda(index_t p) {
            species.is_dead(p) = ((species.i1(p) < 0) || (species.i1(p) >= ni));
          });
      }
    }
#else
    (void)(index_t {});
    NTTError("only minkowski possible in 1d");
#endif
  }

  /**
   * @brief 2d periodic particle bc.
   *
   */
  template <>
  void PIC<Dimension::TWO_D>::particleBoundaryConditions(const real_t&) {

#if (METRIC == MINKOWSKI_METRIC)
    if (m_mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      for (auto& species : m_mblock.particles) {
        auto ni {m_mblock.Ni1()};
        auto nj {m_mblock.Ni2()};
        Kokkos::parallel_for(
          "prtl_bc", species.loopParticles(), Lambda(index_t p) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni;
            } else if (species.i1(p) >= ni) {
              species.i1(p) -= ni;
            }
            if (species.i2(p) < 0) {
              species.i2(p) += nj;
            } else if (species.i2(p) >= nj) {
              species.i2(p) -= nj;
            }
          });
      }
    } else {
      for (auto& species : m_mblock.particles) {
        auto ni {m_mblock.Ni1()};
        auto nj {m_mblock.Ni2()};
        Kokkos::parallel_for(
          "prtl_bc", species.loopParticles(), Lambda(index_t p) {
            species.is_dead(p) = ((species.i1(p) < 0) || (species.i1(p) >= ni)
                                  || (species.i2(p) < 0) || (species.i2(p) >= nj));
          });
      }
    }
#else
    (void)(index_t {});
    for (auto& species : m_mblock.particles) {
      auto ni {m_mblock.Ni1()};
      Kokkos::parallel_for(
        "prtl_bc", species.loopParticles(), Lambda(index_t p) {
          species.is_dead(p) = ((species.i1(p) < -1) || (species.i1(p) >= ni + 1));
        });
    }
#endif
  }

  /**
   * @brief 3d periodic particle bc.
   *
   */
  template <>
  void PIC<Dimension::THREE_D>::particleBoundaryConditions(const real_t&) {
    NTTError("not implemented");
  }

} // namespace ntt
