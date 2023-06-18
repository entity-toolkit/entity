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
      : r0 { params.get<std::vector<real_t>>("problem", "r0") },
        th0 { params.get<std::vector<real_t>>("problem", "th0") },
        ph0 { params.get<std::vector<real_t>>("problem", "ph0") },
        ur0 { params.get<std::vector<real_t>>("problem", "ur0") } {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}

  private:
    const std::vector<real_t> r0, th0, ph0, ur0;
  };

  template <>
  inline void ProblemGenerator<Dim2, GRPICEngine>::UserInitParticles(
    const SimulationParams&, Meshblock<Dim2, GRPICEngine>& mblock) {
    NTTHostErrorIf(r0.size() != th0.size() || r0.size() != ur0.size(),
                   "particle initial condition lenghts should be equal");
    auto&            lecs  = mblock.particles[0];
    auto&            ions  = mblock.particles[1];
    const auto       npart = r0.size();
    array_t<real_t*> r0_d("r0", npart), th0_d("th0", npart), ph0_d("ph0", npart),
      ur0_d("ur0", npart);
    auto r0_h  = Kokkos::create_mirror_view(r0_d);
    auto th0_h = Kokkos::create_mirror_view(th0_d);
    auto ph0_h = Kokkos::create_mirror_view(ph0_d);
    auto ur0_h = Kokkos::create_mirror_view(ur0_d);
    for (std::size_t i { 0 }; i < npart; ++i) {
      r0_h(i)  = r0[i];
      th0_h(i) = th0[i];
      ph0_h(i) = ph0[i];
      ur0_h(i) = ur0[i];
    }
    Kokkos::deep_copy(r0_d, r0_h);
    Kokkos::deep_copy(th0_d, th0_h);
    Kokkos::deep_copy(ph0_d, ph0_h);
    Kokkos::deep_copy(ur0_d, ur0_h);
    Kokkos::parallel_for(
      "UserInitParticles", npart, ClassLambda(index_t p) {
        init_prtl_2d_covariant(mblock, lecs, p, r0_d(p), th0_d(p), ur0_d(p), ZERO, ZERO, ONE);
        init_prtl_2d_covariant(mblock, ions, p, r0_d(p), th0_d(p), ZERO, ZERO, ZERO, ONE);
        lecs.phi(p) = ph0_d(p);
        ions.phi(p) = ph0_d(p);
      });
    lecs.setNpart(npart);
    ions.setNpart(npart);
  }
}    // namespace ntt

#endif
