#include "wrapper.h"
#include "pic.h"

namespace ntt {

#ifdef MINKOWSKI_METRIC
  /**
   * @brief 1d periodic particle bc (minkowski).
   */
  template <>
  void PIC<Dim1>::ParticlesExchange() {
    auto& mblock = this->meshblock;
    if (mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      for (auto& species : mblock.particles) {
        auto ni = mblock.Ni1();
        Kokkos::parallel_for(
          "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.i1(p) < 0) {
              species.i1(p) += ni;
            } else if (species.i1(p) >= ni) {
              species.i1(p) -= ni;
            }
          });
      }
    } else {
      for (auto& species : mblock.particles) {
        auto ni = mblock.Ni1();
        Kokkos::parallel_for(
          "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
            species.is_dead(p) = ((species.i1(p) < 0) || (species.i1(p) >= ni));
          });
      }
    }
  }

  /**
   * @brief 2d periodic particle bc (minkowski).
   */
  template <>
  void PIC<Dim2>::ParticlesExchange() {
    auto& mblock = this->meshblock;
    if (mblock.boundaries[0] == BoundaryCondition::PERIODIC) {
      for (auto& species : mblock.particles) {
        auto ni = mblock.Ni1();
        auto nj = mblock.Ni2();
        Kokkos::parallel_for(
          "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
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
      for (auto& species : mblock.particles) {
        auto ni = mblock.Ni1();
        auto nj = mblock.Ni2();
        Kokkos::parallel_for(
          "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
            species.is_dead(p) = ((species.i1(p) < 0) || (species.i1(p) >= ni)
                                  || (species.i2(p) < 0) || (species.i2(p) >= nj));
          });
      }
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

#endif // MINKOWSKI_METRIC

} // namespace ntt

template void ntt::PIC<ntt::Dim1>::ParticlesExchange();
template void ntt::PIC<ntt::Dim2>::ParticlesExchange();
template void ntt::PIC<ntt::Dim3>::ParticlesExchange();