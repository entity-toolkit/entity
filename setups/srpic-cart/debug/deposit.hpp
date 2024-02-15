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
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) :
      x1 { params.get<std::vector<real_t>>("problem", "x1") },
      x2 { params.get<std::vector<real_t>>("problem", "x2") },
      ux1 { params.get<std::vector<real_t>>("problem", "ux1") },
      ux2 { params.get<std::vector<real_t>>("problem", "ux2") } {}

    inline void UserInitParticles(const SimulationParams&,
                                  Meshblock<D, S>&) override {}

  private:
    const std::vector<real_t> x1, x2, ux1, ux2;
  };

#ifndef MINKOWSKI_METRIC
  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      TargetFields<D, S>(params, mblock) {}

    Inline auto operator()(const em&, const coord_t<D>&) const -> real_t {
      return ZERO;
    }
  };

#endif

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams&,
    Meshblock<Dim2, PICEngine>& mblock) {
    NTTHostErrorIf(x1.size() != x2.size() || x1.size() != ux1.size(),
                   "particle initial condition lenghts should be equal");
    auto&            lecs  = mblock.particles[0];
    auto&            ions  = mblock.particles[1];
    const auto       npart = x1.size();
    array_t<real_t*> x1_d { "x1", npart }, x2_d { "x2", npart },
      ux1_d { "ux1", npart }, ux2_d { "ux2", npart };
    auto x1_h  = Kokkos::create_mirror_view(x1_d);
    auto x2_h  = Kokkos::create_mirror_view(x2_d);
    auto ux1_h = Kokkos::create_mirror_view(ux1_d);
    auto ux2_h = Kokkos::create_mirror_view(ux2_d);
    for (std::size_t i { 0 }; i < npart; ++i) {
      x1_h(i)  = x1[i];
      x2_h(i)  = x2[i];
      ux1_h(i) = ux1[i];
      ux2_h(i) = ux2[i];
    }
    Kokkos::deep_copy(x1_d, x1_h);
    Kokkos::deep_copy(x2_d, x2_h);
    Kokkos::deep_copy(ux1_d, ux1_h);
    Kokkos::deep_copy(ux2_d, ux2_h);
    auto lecs_idx    = array_t<std::size_t>("lecs_idx");
    auto ions_idx    = array_t<std::size_t>("ions_idx");
    auto lecs_offset = lecs.npart();
    auto ions_offset = ions.npart();
    Kokkos::parallel_for(
      "UserInitParticles",
      npart,
      Lambda(index_t p) {
        InjectParticle_2D(mblock,
                          lecs,
                          lecs_idx,
                          lecs_offset,
                          x1_d(p),
                          x2_d(p),
                          ux1_d(p),
                          ux2_d(p),
                          ZERO,
                          ONE);
        InjectParticle_2D(mblock,
                          ions,
                          ions_idx,
                          ions_offset,
                          x1_d(p),
                          x2_d(p),
                          -ux1_d(p),
                          -ux2_d(p),
                          ZERO,
                          ONE);
      });
    auto lecs_idx_h = Kokkos::create_mirror_view(lecs_idx);
    auto ions_idx_h = Kokkos::create_mirror_view(ions_idx);
    Kokkos::deep_copy(lecs_idx_h, lecs_idx);
    Kokkos::deep_copy(ions_idx_h, ions_idx);
    lecs.setNpart(lecs.npart() + lecs_idx_h());
    ions.setNpart(ions.npart() + ions_idx_h());
  }
} // namespace ntt

#endif