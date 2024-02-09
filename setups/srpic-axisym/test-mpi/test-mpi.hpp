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
  #include "nttiny/api.h"
#endif

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct ThermalBackground : public EnergyDistribution<D, S> {
    ThermalBackground(const SimulationParams& params,
                      const Meshblock<D, S>&  mblock) :
      EnergyDistribution<D, S>(params, mblock),
      maxwellian { mblock },
      temperature { params.get<real_t>("problem", "T", 0.1) } {}

    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v, const int&) const override {
      maxwellian(v, temperature);
      v[1] = ZERO;
      v[2] = ZERO;
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

    inline ProblemGenerator(const SimulationParams& params) :
      m_T { params.get<real_t>("problem", "T") },
      m_psr_Rstar { params.get<real_t>("problem", "atm_buff") +
                    params.extent()[0] } {}

    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
    }

    inline void UserInitParticles(const SimulationParams&,
                                  Meshblock<D, S>&) override {}

    Inline auto ext_force_x1(const real_t&, const coord_t<PrtlCoordD>& x_ph) const
      -> real_t override {
      return ZERO;
    }

    Inline auto ext_force_x2(const real_t&, const coord_t<PrtlCoordD>&) const
      -> real_t override {
      return ZERO;
    }

    Inline auto ext_force_x3(const real_t&, const coord_t<PrtlCoordD>&) const
      -> real_t override {
      return ZERO;
    }

  private:
    const real_t          m_psr_Rstar, m_T;
    ndarray_t<(short)(D)> m_ppc_per_spec;
  };

  template <Dimension D>
  Inline void mainBField(const coord_t<D>& x_ph,
                         vec_t<Dim3>&,
                         vec_t<Dim3>& b_out,
                         real_t       _rstar) {
    b_out[0] = math::cos(x_ph[1]) / CUBE(x_ph[0] / _rstar);
    b_out[1] = HALF * math::sin(x_ph[1]) / CUBE(x_ph[0] / _rstar);
    b_out[2] = ZERO;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    // initialize buffer array
    m_ppc_per_spec = ndarray_t<2>("ppc_per_spec", mblock.Ni1(), mblock.Ni2());

    // inject stars in the atmosphere

    const auto ppc0 = params.ppc0();

    Kokkos::parallel_for(
      "ComputeDeltaNdens",
      mblock.rangeActiveCells(),
      ClassLambda(index_t i1, index_t i2) {
        const auto i1_ = static_cast<int>(i1) - N_GHOSTS;
        const auto i2_ = static_cast<int>(i2) - N_GHOSTS;
        const auto r = mblock.metric.x1_Code2Phys(static_cast<real_t>(i1_) + HALF);
        const auto phi = mblock.metric.x2_Code2Phys(static_cast<real_t>(i2_) + HALF);
        m_ppc_per_spec(
          i1_,
          i2_) = 2 * (r > m_psr_Rstar) * (r < 1.2 * m_psr_Rstar) *
                 ((phi > 0.1 * constant::PI) * (phi < 0.4 * constant::PI) +
                  (phi > 0.6 * constant::PI) * (phi < 0.7 * constant::PI));
        // 2 -- for two species
        m_ppc_per_spec(i1_, i2_) *= ppc0 / TWO;
      });
    InjectNonUniform<Dim2, PICEngine, ThermalBackground>(params,
                                                         mblock,
                                                         { 1, 2 },
                                                         m_ppc_per_spec);
  }

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      TargetFields<D, S>(params, mblock),
      _rstar { params.get<real_t>("problem", "atm_buff") + params.extent()[0] },
      _bsurf { params.get<real_t>("problem", "psr_Bsurf", ONE) } {}

    Inline real_t operator()(const em& comp, const coord_t<D>& xi) const override {
      if ((comp == em::bx1) || (comp == em::bx2)) {
        vec_t<Dim3> e_out { ZERO }, b_out { ZERO };
        coord_t<D>  x_ph { ZERO };
        (this->m_mblock).metric.x_Code2Phys(xi, x_ph);
        mainBField<D>(x_ph, e_out, b_out, _rstar);
        return (comp == em::bx1) ? b_out[0] : b_out[1];
      } else {
        return ZERO;
      }
    }

  private:
    const real_t _rstar, _bsurf;
  };

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const real_t&               time,
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& m_mblock) {
    const auto Equator { m_mblock.Ni2() / 2 };
    if (time < 0.01) {
      for (std::size_t s { 0 }; s < 2; ++s) {
        auto& species = m_mblock.particles[s];
        Kokkos::parallel_for(
          "pushParticles",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            vec_t<Dim3> b_int_Cart;
            {
              vec_t<Dim3> b_int;
              real_t      c000, c100, c010, c110, c00, c10;

              const auto   i { species.i1(p) + N_GHOSTS };
              const real_t dx1 { species.dx1(p) };
              const auto   j { species.i2(p) + N_GHOSTS };
              const real_t dx2 { species.dx2(p) };
              // Bx1
              c000     = HALF * (BX1(i, j) + BX1(i, j - 1));
              c100     = HALF * (BX1(i + 1, j) + BX1(i + 1, j - 1));
              c010     = HALF * (BX1(i, j) + BX1(i, j + 1));
              c110     = HALF * (BX1(i + 1, j) + BX1(i + 1, j + 1));
              c00      = c000 * (ONE - dx1) + c100 * dx1;
              c10      = c010 * (ONE - dx1) + c110 * dx1;
              b_int[0] = c00 * (ONE - dx2) + c10 * dx2;
              // Bx2
              c000     = HALF * (BX2(i - 1, j) + BX2(i, j));
              c100     = HALF * (BX2(i, j) + BX2(i + 1, j));
              c010     = HALF * (BX2(i - 1, j + 1) + BX2(i, j + 1));
              c110     = HALF * (BX2(i, j + 1) + BX2(i + 1, j + 1));
              c00      = c000 * (ONE - dx1) + c100 * dx1;
              c10      = c010 * (ONE - dx1) + c110 * dx1;
              b_int[1] = c00 * (ONE - dx2) + c10 * dx2;
              // Bx3
              c000     = INV_4 * (BX3(i - 1, j - 1) + BX3(i - 1, j) +
                              BX3(i, j - 1) + BX3(i, j));
              c100 = INV_4 * (BX3(i, j - 1) + BX3(i, j) + BX3(i + 1, j - 1) +
                              BX3(i + 1, j));
              c010 = INV_4 * (BX3(i - 1, j) + BX3(i - 1, j + 1) + BX3(i, j) +
                              BX3(i, j + 1));
              c110 = INV_4 * (BX3(i, j) + BX3(i, j + 1) + BX3(i + 1, j) +
                              BX3(i + 1, j + 1));
              c00  = c000 * (ONE - dx1) + c100 * dx1;
              c10  = c010 * (ONE - dx1) + c110 * dx1;
              b_int[2] = c00 * (ONE - dx2) + c10 * dx2;

              const vec_t<Dim3> xp { static_cast<real_t>(species.i1(p)) +
                                       static_cast<real_t>(species.dx1(p)),
                                     static_cast<real_t>(species.i2(p)) +
                                       static_cast<real_t>(species.dx2(p)),
                                     species.phi(p) };
              m_mblock.metric.v3_Cntrv2Cart(xp, b_int, b_int_Cart);
            }
            auto babs { NORM(b_int_Cart[0], b_int_Cart[1], b_int_Cart[2]) };
            b_int_Cart[0] /= (babs + 1e-12);
            b_int_Cart[1] /= (babs + 1e-12);
            b_int_Cart[2] /= (babs + 1e-12);
            if (species.i2(p) < Equator) {
              species.ux1(p) = 100.0 * b_int_Cart[0];
              species.ux2(p) = 100.0 * b_int_Cart[1];
              species.ux3(p) = 100.0 * b_int_Cart[2];
            } else {
              species.ux1(p) = -100.0 * b_int_Cart[0];
              species.ux2(p) = -100.0 * b_int_Cart[1];
              species.ux3(p) = -100.0 * b_int_Cart[2];
            }
          });
      }
    }
  }

  /**
   * Field initialization for 2D:
   */

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    const auto rstar = m_psr_Rstar;
    Kokkos::parallel_for(
      "UserInitFields",
      mblock.rangeActiveCells(),
      ClassLambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, mainBField<Dim2>, rstar);
      });
  }
} // namespace ntt

#endif
