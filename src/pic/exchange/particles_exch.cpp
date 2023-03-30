#include "wrapper.h"

#include "pic.h"

namespace ntt {

#ifdef MINKOWSKI_METRIC
  /**
   * @brief 1d periodic particle bc (minkowski).
   */
  template <>
  void PIC<Dim1>::ParticlesExchange() {
    auto& mblock      = this->meshblock;
    auto  periodic_x1 = (mblock.boundaries[0][0] == BoundaryCondition::PERIODIC);
    auto  ni1         = mblock.Ni1();
    for (auto& species : mblock.particles) {
      Kokkos::parallel_for(
        "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
          if (periodic_x1) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni1;
            } else if (species.i1(p) >= ni1) {
              species.i1(p) -= ni1;
            }
          } else {
            if ((species.i1(p) < 0) || (species.i1(p) >= ni1)) {
              species.tag(p) = static_cast<short>(ParticleTag::dead);
            }
          }
        });
    }
  }

  /**
   * @brief 2d periodic particle bc (minkowski).
   */
  template <>
  void PIC<Dim2>::ParticlesExchange() {
    auto& mblock      = this->meshblock;
    auto  periodic_x1 = (mblock.boundaries[0][0] == BoundaryCondition::PERIODIC);
    auto  periodic_x2 = (mblock.boundaries[1][0] == BoundaryCondition::PERIODIC);
    auto  ni1         = mblock.Ni1();
    auto  ni2         = mblock.Ni2();
    for (auto& species : mblock.particles) {
      Kokkos::parallel_for(
        "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
          if (periodic_x1) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni1;
            } else if (species.i1(p) >= ni1) {
              species.i1(p) -= ni1;
            }
          } else {
            if ((species.i1(p) < 0) || (species.i1(p) >= ni1)) {
              species.tag(p) = static_cast<short>(ParticleTag::dead);
            }
          }
          if (periodic_x2) {
            if (species.i2(p) < 0) {
              species.i2(p) += ni2;
            } else if (species.i2(p) >= ni2) {
              species.i2(p) -= ni2;
            }
          } else {
            if ((species.i2(p) < 0) || (species.i2(p) >= ni2)) {
              species.tag(p) = static_cast<short>(ParticleTag::dead);
            }
          }
        });
    }
  }

  /**
   * @brief 3d periodic particle bc (minkowski).
   */
  template <>
  void PIC<Dim3>::ParticlesExchange() {
    NTTHostError("not implemented");
  }
#else
  template <Dimension D>
  void PIC<D>::ParticlesExchange() {}

#endif    // MINKOWSKI_METRIC

}    // namespace ntt

#ifndef MINKOWSKI_METRIC
template void ntt::PIC<ntt::Dim1>::ParticlesExchange();
template void ntt::PIC<ntt::Dim2>::ParticlesExchange();
template void ntt::PIC<ntt::Dim3>::ParticlesExchange();
#endif
