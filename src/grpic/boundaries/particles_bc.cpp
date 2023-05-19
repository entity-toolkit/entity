/**
 * @file particles_bc.cpp
 * @brief Special "reflecting" boundary conditions on the axis ...
 *        ... for particles in 2D axisymmetric simulations.
 * @implements: `ParticlesBoundaryConditions` method of the `GRPIC` class
 * @includes: --
 * @depends: `grpic.h`
 *
 */

#include "wrapper.h"

#include "grpic.h"
#include "particle_macros.h"

namespace ntt {

  /**
   * @brief 2d particle bc.
   */
  template <>
  void GRPIC<Dim2>::ParticlesBoundaryConditions() {
    auto&         mblock = this->meshblock;
    auto          params = *(this->params());
    const auto    rh     = params.metricParameters()[5];
    coord_t<Dim2> xh_CU { ZERO };
    mblock.metric.x_Sph2Code({ rh, ZERO }, xh_CU);
    auto       i1h      = static_cast<int>(xh_CU[0]);
    // !TODO: make this more rigorous
    const auto buffer_h = 5;
    i1h -= buffer_h;
    i1h = std::max(i1h, 0);
    // NTTHostErrorIf(i1h <= 0, "not enough buffer at rmin below the horizon");

    for (auto& species : mblock.particles) {
      auto ni1 = mblock.Ni1();
      auto ni2 = mblock.Ni2();
      Kokkos::parallel_for(
        "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
          coord_t<Dim2> xp_CU { ZERO }, xp { ZERO };
          xp_CU[0] = get_prtl_x1(species, p);
          xp_CU[1] = get_prtl_x2(species, p);
          mblock.metric.x_Code2Sph(xp_CU, xp);

          // radial boundary conditions
          if ((species.i1(p) < i1h) || (species.i1(p) >= ni1)) {
            species.tag(p) = static_cast<short>(ParticleTag::dead);
          }
          // axis boundaries
          if ((species.i2(p) < 0) || (species.i2(p) >= ni2)) {
            if (species.i2(p) < 0) {
              // reflect particle coordinate
              species.i2(p) = 0;
            } else {
              // reflect particle coordinate
              species.i2(p) = ni2 - 1;
            }
            species.dx2(p) = 1.0f - species.dx2(p);
            species.phi(p) = species.phi(p) + constant::PI;
            // reverse u^theta
            species.ux2(p) = -species.ux2(p);
          }
        });
    }
  }

  /**
   * @brief 3d particle bc.
   */
  template <>
  void GRPIC<Dim3>::ParticlesBoundaryConditions() {
    NTTHostError("not implemented");
  }

}    // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::ParticlesBoundaryConditions();
template void ntt::GRPIC<ntt::Dim3>::ParticlesBoundaryConditions();