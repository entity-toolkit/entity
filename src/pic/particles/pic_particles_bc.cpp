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
      NTTError("boundary condition not implemented");
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
      NTTError("boundary condition not implemented");
    }
#elif (METRIC == SPHERICAL_METRIC) || (METRIC == QSPHERICAL_METRIC)
    (void)(index_t {});
    for (auto& species : m_mblock.particles) {
      (void)(species);
      Kokkos::parallel_for("prtl_bc",
                           species.loopParticles(),
                           Lambda(index_t) {
                             // if (species.i1(p) < 0) {
                             //   species.i1(p) += m_mblock.Ni1();
                             // } else if (species.i1(p) >= m_mblock.Ni1()) {
                             //   species.i1(p) -= m_mblock.Ni1();
                             // }
                             // if (species.i2(p) < 0) {
                             //   species.i2(p) += m_mblock.Ni2();
                             // } else if (species.i2(p) >= m_mblock.Ni2()) {
                             //   species.i2(p) -= m_mblock.Ni2();
                             // }
                           });

      // NTTError("non-minkowski particle boundary condition not implemented");
    }
#else
    (void)(index_t {});
    for (auto& species : m_mblock.particles) {
      NTTError("2d boundary condition for metric not implemented");
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
