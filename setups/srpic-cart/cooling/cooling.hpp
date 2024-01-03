#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override;

    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override;
  };

  template <Dimension D>
  Inline void bzField(const coord_t<D>&, vec_t<Dim3>&, vec_t<Dim3>& b_out, real_t) {
    b_out[2] = ONE;
  }

  template <Dimension D, SimulationEngine S>
  inline void ProblemGenerator<D, S>::UserInitFields(const SimulationParams&,
                                                     Meshblock<D, S>& mblock) {
    set_em_fields(mblock, bzField<D>, ZERO);
  }

  template <Dimension D, SimulationEngine S>
  inline void ProblemGenerator<D, S>::UserInitParticles(
    const SimulationParams& params,
    Meshblock<D, S>&        mblock) {
    auto&        prtls = mblock.particles[0];
    const real_t ux { params.get<real_t>("problem", "ux") },
      uy { params.get<real_t>("problem", "uy") },
      uz { params.get<real_t>("problem", "uz") };

    auto prtls_idx    = array_t<std::size_t>("lecs_idx");
    auto prtls_offset = prtls.npart();
    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "UserInitParticles",
        1,
        Lambda(index_t) {
          InjectParticle_1D(mblock, prtls, prtls_idx, prtls_offset, ZERO, ux, uy, uz, ONE);
        });
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for(
        "UserInitParticles",
        1,
        Lambda(index_t) {
          InjectParticle_2D(mblock,
                            prtls,
                            prtls_idx,
                            prtls_offset,
                            ZERO,
                            ZERO,
                            ux,
                            uy,
                            uz,
                            ONE);
        });
    } else if constexpr (D == Dim3) {
      Kokkos::parallel_for(
        "UserInitParticles",
        1,
        Lambda(index_t) {
          InjectParticle_3D(mblock,
                            prtls,
                            prtls_idx,
                            prtls_offset,
                            ZERO,
                            ZERO,
                            ZERO,
                            ux,
                            uy,
                            uz,
                            ONE);
        });
    }
    auto prtls_idx_h = Kokkos::create_mirror_view(prtls_idx);
    Kokkos::deep_copy(prtls_idx_h, prtls_idx);
    prtls.setNpart(prtls.npart() + prtls_idx_h());
    std::cout << "Number of particles injected: " << prtls.npart() << std::endl;
  }

} // namespace ntt

#endif