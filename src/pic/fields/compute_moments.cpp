#include "wrapper.h"

#include "fields.h"
#include "pic.h"

namespace ntt {
  template <>
  void PIC<Dim1>::ComputeDensity(const short& smooth) {
    auto& mblock = this->meshblock;
    auto& params = *(this->params());
    Kokkos::deep_copy(mblock.buff, ZERO);
    auto   scatter_buff = Kokkos::Experimental::create_scatter_view(mblock.buff);
    real_t contrib      = (1.0 / params.ppc0()) / (2.0 * smooth + 1.0);
    for (auto& species : mblock.particles) {
      if (species.mass() != 0.0) {
        Kokkos::parallel_for(
          "density", species.rangeActiveParticles(), Lambda(index_t p) {
            auto i1          = species.i1(p);
            auto dens_access = scatter_buff.access();
            for (int i1_ = i1 - smooth + N_GHOSTS; i1_ <= i1 + smooth + N_GHOSTS; ++i1_) {
              dens_access(i1, fld::dens) += contrib;
            }
          });
      }
    }
    Kokkos::Experimental::contribute(mblock.buff, scatter_buff);
  }

  template <>
  void PIC<Dim2>::ComputeDensity(const short& smooth) {
    auto& mblock = this->meshblock;
    auto& params = *(this->params());
    Kokkos::deep_copy(mblock.buff, ZERO);
    auto   scatter_buff = Kokkos::Experimental::create_scatter_view(mblock.buff);
    real_t contrib      = (1.0 / params.ppc0()) / SQR(2.0 * smooth + 1.0);
    auto   ni1          = mblock.Ni1();
    auto   ni2          = mblock.Ni2();
    for (auto& species : mblock.particles) {
      if (species.mass() != 0.0) {
        Kokkos::parallel_for(
          "density", species.rangeActiveParticles(), Lambda(index_t p) {
            auto i1          = species.i1(p);
            auto i2          = species.i2(p);
            auto dens_access = scatter_buff.access();
            for (int i2_ { IMIN(IMAX(i2 - smooth + N_GHOSTS, 0), ni2 + 2 * N_GHOSTS) };
                 i2_ <= IMIN(IMAX(i2 + smooth + N_GHOSTS, 0), ni2 + 2 * N_GHOSTS);
                 ++i2_) {
              for (int i1_ { IMIN(IMAX(i1 - smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS) };
                   i1_ <= IMIN(IMAX(i1 + smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
                   ++i1_) {
                dens_access(i1_, i2_, fld::dens) += contrib;
              }
            }
          });
      }
    }
    Kokkos::Experimental::contribute(mblock.buff, scatter_buff);
  }

  template <>
  void PIC<Dim3>::ComputeDensity(const short& smooth) {
    auto& mblock = this->meshblock;
    auto& params = *(this->params());
    Kokkos::deep_copy(mblock.buff, ZERO);
    auto   scatter_buff = Kokkos::Experimental::create_scatter_view(mblock.buff);
    real_t contrib      = (1.0 / params.ppc0()) / CUBE(2.0 * smooth + 1.0);
    for (auto& species : mblock.particles) {
      if (species.mass() != 0.0) {
        Kokkos::parallel_for(
          "density", species.rangeActiveParticles(), Lambda(index_t p) {
            auto i1          = species.i1(p);
            auto i2          = species.i2(p);
            auto i3          = species.i3(p);
            auto dens_access = scatter_buff.access();
            for (int i3_ = i3 - smooth + N_GHOSTS; i3_ <= i3 + smooth + N_GHOSTS; ++i3_) {
              for (int i2_ = i2 - smooth + N_GHOSTS; i2_ <= i2 + smooth + N_GHOSTS; ++i2_) {
                for (int i1_ = i1 - smooth + N_GHOSTS; i1_ <= i1 + smooth + N_GHOSTS; ++i1_) {
                  dens_access(i1_, i2_, i3_, fld::dens) += contrib;
                }
              }
            }
          });
      }
    }
    Kokkos::Experimental::contribute(mblock.buff, scatter_buff);
  }
}    // namespace ntt
