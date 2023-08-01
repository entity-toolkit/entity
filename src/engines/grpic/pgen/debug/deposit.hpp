#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "particle_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"
#include "utils/qmath.h"

#include "utils/archetypes.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : TargetFields<D, S>(params, mblock) {}
    Inline auto operator()(const em&, const coord_t<D>&) const -> real_t {
      return ZERO;
    }
  };
  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params)
      : x1 { params.get<std::vector<real_t>>("problem", "x1") },
        x2 { params.get<std::vector<real_t>>("problem", "x2") },
        x3 { params.get<std::vector<real_t>>("problem", "x3") },
        ux1 { params.get<std::vector<real_t>>("problem", "ux1") },
        ux2 { params.get<std::vector<real_t>>("problem", "ux2") } {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

  private:
    const std::vector<real_t> x1, x2, x3, ux1, ux2;
  };

  template <>
  inline void ProblemGenerator<Dim2, GRPICEngine>::UserDriveParticles(
    const real_t&, const SimulationParams& params, Meshblock<Dim2, GRPICEngine>& mblock) {
    auto&      lecs     = mblock.particles[0];
    const auto r_absorb = params.metricParameters()[2];
    Kokkos::parallel_for(
      "UserInitParticles", lecs.rangeAllParticles(), ClassLambda(index_t p) {
        coord_t<Dim2> x_cu { ZERO }, x_ph { ZERO };
        x_cu[0] = (real_t)lecs.i1(p) + (real_t)lecs.dx1(p);
        mblock.metric.x_Code2Sph(x_cu, x_ph);
        if (x_ph[0] >= r_absorb) {
          lecs.ux1(p) = -lecs.ux1(p);
        }
      });
  }

  template <>
  inline void ProblemGenerator<Dim2, GRPICEngine>::UserInitParticles(
    const SimulationParams&, Meshblock<Dim2, GRPICEngine>& mblock) {
    NTTHostErrorIf(x1.size() != x2.size() || x1.size() != ux1.size(),
                   "particle initial condition lenghts should be equal");
    auto&            lecs  = mblock.particles[0];
    auto&            ions  = mblock.particles[1];
    const auto       npart = x1.size();
    array_t<real_t*> x1_d("x1", npart), x2_d("x2", npart), x3_d("x3", npart),
      ux1_d("ux1", npart), ux2_d("ux2", npart);
    auto x1_h  = Kokkos::create_mirror_view(x1_d);
    auto x2_h  = Kokkos::create_mirror_view(x2_d);
    auto x3_h  = Kokkos::create_mirror_view(x3_d);
    auto ux1_h = Kokkos::create_mirror_view(ux1_d);
    auto ux2_h = Kokkos::create_mirror_view(ux2_d);
    for (std::size_t i { 0 }; i < npart; ++i) {
      x1_h(i)  = x1[i];
      x2_h(i)  = x2[i];
      x3_h(i)  = x3[i];
      ux1_h(i) = ux1[i];
      ux2_h(i) = ux2[i];
    }
    Kokkos::deep_copy(x1_d, x1_h);
    Kokkos::deep_copy(x2_d, x2_h);
    Kokkos::deep_copy(x3_d, x3_h);
    Kokkos::deep_copy(ux1_d, ux1_h);
    Kokkos::deep_copy(ux2_d, ux2_h);
    Kokkos::parallel_for(
      "UserInitParticles", npart, ClassLambda(index_t p) {
        init_prtl_2d_covariant(
          mblock, lecs, p, x1_d(p), x2_d(p), ux1_d(p), ux2_d(p), ZERO, ONE);
        init_prtl_2d_covariant(
          mblock, ions, p, x1_d(p), x2_d(p), -ux1_d(p), ux2_d(p), ZERO, ONE);
        lecs.phi(p) = x3_d(p);
        ions.phi(p) = x3_d(p);
      });
    lecs.setNpart(npart);
    ions.setNpart(npart);
  }
}    // namespace ntt

#endif
