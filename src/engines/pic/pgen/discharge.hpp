#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "particle_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

#ifdef GUI_ENABLED
#  include "nttiny/api.h"
#endif

namespace ntt {

  /**
   * Define a structure which will initialize the particle energy distribution.
   * This is used below in the UserInitParticles function.
   */
  template <Dimension D, SimulationEngine S>
  struct ThermalBackground : public EnergyDistribution<D, S> {
    ThermalBackground(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        maxwellian { mblock },
        temperature { params.get<real_t>("problem", "T", 0.1) } {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      maxwellian(v, temperature);
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           temperature;
  };

  /**
   * Main problem generator class with all the required functions to define
   * the initial/boundary conditions and the source terms.
   */
  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    real_t work_done { ZERO };
    inline ProblemGenerator(const SimulationParams& params)
      : m_T { params.get<real_t>("problem", "T") },
        m_C { params.get<real_t>("problem", "contrast") },
        m_Rstar { params.get<real_t>("problem", "Rstar") },
        m_h { params.get<real_t>("problem", "h") },
        m_rGJ { math::log(m_C) * m_h + m_Rstar } {}
    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override;
    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override;
    inline void UserInitParticles(const SimulationParams& params,
                                  Meshblock<D, S>&        mblock) override {}
#ifdef EXTERNAL_FORCE
    Inline auto ext_force_x1(const real_t&, const coord_t<D>& x_ph) const -> real_t override {
      // real_t wid = ONE / (0.1 * m_h);
      return -(HALF * m_T / m_h);
      //  * (math::tanh((x_ph[0] - m_Rstar) * wid) - math::tanh((x_ph[0] - m_rGJ) * wid));
    }
    Inline auto ext_force_x2(const real_t&, const coord_t<D>&) const -> real_t override {
      return ZERO;
    }
    Inline auto ext_force_x3(const real_t&, const coord_t<D>&) const -> real_t override {
      return ZERO;
    }
#endif

  private:
    const real_t          m_T, m_C, m_Rstar, m_h, m_rGJ;
    ndarray_t<(short)(D)> m_ppc_per_spec;
  };

  Inline void background_fields_1d(const coord_t<Dim1>& x_ph,     // physical coordinate
                                   vec_t<Dim3>&         e_out,    // electric field [out]
                                   vec_t<Dim3>&         b_out,    // magnetic field [out]
                                   const real_t         time) {
    b_out[0] = 1.0;
  }

  Inline void background_fields_2d(const coord_t<Dim2>& x_ph,     // physical coordinate
                                   vec_t<Dim3>&         e_out,    // electric field [out]
                                   vec_t<Dim3>&         b_out,    // magnetic field [out]
                                   const real_t         time) {
    b_out[0] = 1.0;
  }

  Inline void background_fields_3d(const coord_t<Dim3>& x_ph,     // physical coordinate
                                   vec_t<Dim3>&         e_out,    // electric field [out]
                                   vec_t<Dim3>&         b_out,    // magnetic field [out]
                                   const real_t         time) {
    b_out[0] = 1.0;
  }

  template <>
  inline void ProblemGenerator<Dim1, PICEngine>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim1, PICEngine>& mblock) {
    // initialize buffer array
    m_ppc_per_spec = ndarray_t<1>("ppc_per_spec", mblock.Ni1());

    // check that the star surface is far-enough from the boundary
    coord_t<Dim1> star_cu { ZERO };
    mblock.metric.x_Phys2Code({ m_Rstar }, star_cu);
    if ((int)(star_cu[0]) < (int)params.currentFilters()) {
      NTTWarn("The star boundary is smaller than the current filter stencil.");
    }
  }

  /**
   * Field initialization for 2D:
   */
  template <Dimension D, SimulationEngine S>
  inline void ProblemGenerator<D, S>::UserInitFields(const SimulationParams&,
                                                     Meshblock<D, S>& mblock) {
    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i) {
          set_em_fields_1d(mblock, i, background_fields_1d, ZERO);
        });
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for(
        "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
          set_em_fields_2d(mblock, i, j, background_fields_2d, ZERO);
        });
    } else if constexpr (D == Dim3) {
      Kokkos::parallel_for(
        "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j, index_t k) {
          set_em_fields_3d(mblock, i, j, k, background_fields_3d, ZERO);
        });
    }
  }

  template <Dimension D, SimulationEngine S>
  inline void ProblemGenerator<D, S>::UserDriveParticles(const real_t&,
                                                         const SimulationParams& params,
                                                         Meshblock<D, S>&        mblock) {
    const short buff_idx = 0;
    mblock.ComputeMoments(params, FieldID::N, {}, { 1, 2 }, buff_idx, 0);

    const auto ppc0 = params.ppc0();

    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "ComputeDeltaNdens", mblock.rangeActiveCells(), ClassLambda(index_t i1) {
          const auto          i1_ = static_cast<int>(i1) - N_GHOSTS;
          const coord_t<Dim1> x_cu { static_cast<real_t>(i1_) + HALF };
          coord_t<Dim1>       x_ph { ZERO };
          mblock.metric.x_Code2Phys(x_cu, x_ph);
          m_ppc_per_spec(i1_)
            = m_C * math::exp(-(x_ph[0] - m_Rstar) / m_h) * (x_ph[0] > m_Rstar);

          const auto target_ndens = mblock.buff(i1, buff_idx);
          if (m_ppc_per_spec(i1_) > target_ndens) {
            m_ppc_per_spec(i1_) -= target_ndens;
          } else {
            m_ppc_per_spec(i1_) = ZERO;
          }
          // 2 -- for two species
          m_ppc_per_spec(i1_) *= ppc0 / TWO;
        });
    } else if constexpr (D == Dim2) {
      NTTHostError("not implemented");
    } else if constexpr (D == Dim3) {
      NTTHostError("not implemented");
    }
    InjectNonUniform<D, PICEngine, ThermalBackground>(params, mblock, { 1, 2 }, m_ppc_per_spec);

    for (auto& species : mblock.particles) {
      Kokkos::parallel_for(
        "UserDriveParticles", species.rangeActiveParticles(), ClassLambda(index_t p) {
          species.tag(p) = ParticleTag::alive * (species.i1(p) > 1);
        });
    }
  }
}    // namespace ntt

#endif