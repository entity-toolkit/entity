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
  enum FieldMode {
    MonopoleField = 1,
    DipoleField   = 2
  };

  /**
   * Define a structure which will initialize the particle energy distribution.
   * This is used below in the UserInitParticles function.
   */
  template <Dimension D, SimulationEngine S>
  struct ThermalBackground : public EnergyDistribution<D, S> {
    ThermalBackground(const SimulationParams& params,
                      const Meshblock<D, S>&  mblock) :
      EnergyDistribution<D, S>(params, mblock),
      maxwellian { mblock },
      temperature { params.get<real_t>("problem", "atm_T") } {}

    Inline void operator()(const coord_t<PrtlCoordD>&,
                           vec_t<Dim3>& v,
                           const int&) const override {
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
      m_atm_T { params.get<real_t>("problem", "atm_T") },
      m_atm_C { params.get<real_t>("problem", "atm_contrast") },
      m_atm_h { params.get<real_t>("problem", "atm_h") },
      m_psr_Rstar { params.get<real_t>("problem", "atm_buff") + params.extent()[0] },
      m_psr_Bsurf { params.get<real_t>("problem", "psr_Bsurf", ONE) },
      m_psr_omega { params.get<real_t>("problem", "psr_omega") },
      m_psr_spinup_time { params.get<real_t>("problem", "psr_spinup_time", ZERO) },
      m_rGJ { math::log(m_atm_C) * m_atm_h + m_psr_Rstar },
      m_gravity { m_atm_T / m_atm_h },
      m_psr_field_mode { params.get<int>("problem", "psr_field_mode", 2) == 2
                           ? DipoleField
                           : MonopoleField } {}

    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
    }

    inline void UserInitParticles(const SimulationParams&,
                                  Meshblock<D, S>&) override {}

    inline void UserDriveFields(const real_t&,
                                const SimulationParams&,
                                Meshblock<D, S>&) override {}

    Inline auto ext_force_x1(const real_t&, const coord_t<PrtlCoordD>& x_ph) const
      -> real_t override {
      return -m_gravity * SQR(m_psr_Rstar / x_ph[0]) *
             (x_ph[0] < m_psr_Rstar + m_atm_h * TWO);
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
    const real_t    m_atm_T, m_atm_C, m_atm_h;
    const real_t    m_psr_Rstar, m_psr_Bsurf, m_psr_omega, m_psr_spinup_time;
    const real_t    m_rGJ, m_gravity;
    const FieldMode m_psr_field_mode;
    ndarray_t<(short)(D)> m_ppc_per_spec;
  };

  Inline void mainBField(const coord_t<PrtlCoordD>& x_ph,
                         vec_t<Dim3>&,
                         vec_t<Dim3>& b_out,
                         real_t       _rstar,
                         real_t       _bsurf,
                         int          _mode) {
    if (_mode == 2) {
      b_out[0] = _bsurf * math::cos(x_ph[1]) / CUBE(x_ph[0] / _rstar);
      b_out[1] = _bsurf * HALF * math::sin(x_ph[1]) / CUBE(x_ph[0] / _rstar);
      b_out[2] = ZERO;
    } else {
      b_out[0] = _bsurf * SQR(_rstar / x_ph[0]);
      b_out[1] = ZERO;
      b_out[2] = ZERO;
    }
  }

  Inline void surfaceRotationField(const coord_t<PrtlCoordD>& x_ph,
                                   vec_t<Dim3>&          e_out,
                                   vec_t<Dim3>&          b_out,
                                   real_t                _rstar,
                                   real_t                _bsurf,
                                   int                   _mode,
                                   real_t                _omega) {
    mainBField(x_ph, e_out, b_out, _rstar, _bsurf, _mode);
    e_out[0] = _omega * b_out[1] * x_ph[0] * math::sin(x_ph[1]);
    e_out[1] = -_omega * b_out[0] * x_ph[0] * math::sin(x_ph[1]);
    e_out[2] = 0.0;
  }

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      TargetFields<D, S>(params, mblock),
      _rstar { params.get<real_t>("problem", "atm_buff") + params.extent()[0] },
      _bsurf { params.get<real_t>("problem", "psr_Bsurf", ONE) },
      _mode { params.get<int>("problem", "psr_field_mode", 2) } {}

    Inline real_t operator()(const em&             comp,
                             const coord_t<PrtlCoordD>& xi) const override {
      if ((comp == em::bx1) || (comp == em::bx2)) {
        vec_t<Dim3>    e_out { ZERO }, b_out { ZERO };
        coord_t<PrtlCoordD> x_ph { ZERO };
        (this->m_mblock).metric.x_Code2Phys(xi, x_ph);
        mainBField(x_ph, e_out, b_out, _rstar, _bsurf, _mode);
        return (comp == em::bx1) ? b_out[0] : b_out[1];
      } else {
        return ZERO;
      }
    }

  private:
    const real_t _rstar, _bsurf;
    const int    _mode;
  };

  Inline auto densityProfile(real_t r, real_t C, real_t h, real_t Rstar) -> real_t {
    return C * math::exp(-(Rstar / h) * (ONE - (Rstar / r)));
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    // initialize buffer array
    m_ppc_per_spec = ndarray_t<2>("ppc_per_spec", mblock.Ni1(), mblock.Ni2());

    // inject stars in the atmosphere
    {
      const auto ppc0  = params.ppc0();
      const auto C     = m_atm_C;
      const auto h     = m_atm_h;
      const auto rstar = m_psr_Rstar;

      Kokkos::parallel_for(
        "ComputeDeltaNdens",
        mblock.rangeActiveCells(),
        ClassLambda(index_t i1, index_t i2) {
          const auto i1_ = static_cast<int>(i1) - N_GHOSTS;
          const auto i2_ = static_cast<int>(i2) - N_GHOSTS;
          const auto r = mblock.metric.x1_Code2Phys(static_cast<real_t>(i1_) + HALF);
          m_ppc_per_spec(i1_, i2_) = densityProfile(r, C, h, rstar) * (r > rstar) *
                                     (r < rstar + static_cast<real_t>(10) * h);
          // 2 -- for two species
          m_ppc_per_spec(i1_, i2_) *= ppc0 / TWO;
        });
      InjectNonUniform<Dim2, PICEngine, ThermalBackground>(params,
                                                           mblock,
                                                           { 1, 2 },
                                                           m_ppc_per_spec);
    }
  }

  /**
   * Field initialization for 2D:
   */

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    {
      const auto rmin   = mblock.metric.x1_min;
      const auto x1_psr = mblock.metric.x1_Phys2Code(m_psr_Rstar);
      NTTHostErrorIf(rmin >= m_psr_Rstar, "rmin > r_surf");
      NTTHostErrorIf(x1_psr < params.currentFilters(), "r_surf - rmin < filters");
    }
    {
      const auto rstar = m_psr_Rstar;
      const auto bsurf = m_psr_Bsurf;
      const auto mode  = m_psr_field_mode;
      Kokkos::parallel_for(
        "UserInitFields",
        mblock.rangeActiveCells(),
        ClassLambda(index_t i, index_t j) {
          set_em_fields_2d(mblock, i, j, mainBField, rstar, bsurf, mode);
        });
    }
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveFields(
    const real_t& time,
    const SimulationParams&,
    Meshblock<Dim2, PICEngine>& mblock) {
    const unsigned int i1_surf {
      static_cast<int>(mblock.metric.x1_Phys2Code(m_psr_Rstar)) + N_GHOSTS
    };
    const auto i1_min = mblock.i1_min();

    {
      const auto rstar = m_psr_Rstar;
      const auto bsurf = m_psr_Bsurf;
      const auto mode  = m_psr_field_mode;
      const auto omega = m_psr_omega * ((time < m_psr_spinup_time)
                                          ? (time / m_psr_spinup_time)
                                          : ONE);

      Kokkos::parallel_for(
        "UserDriveFields_rmin",
        CreateRangePolicy<Dim2>({ i1_min, mblock.i2_min() },
                                { i1_surf, mblock.i2_max() }),
        ClassLambda(index_t i1, index_t i2) {
          set_ex2_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
          set_ex3_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
          set_bx1_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
          if (i1 < i1_surf - 1) {
            set_ex1_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
            set_bx2_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
            set_bx3_2d(mblock, i1, i2, surfaceRotationField, rstar, bsurf, mode, omega);
          }
        });
    }
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const real_t&,
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    constexpr short buff_idx = 0;
    {
      // compute the number densities
      const short smooth = 0;
      mblock.ComputeMoments(params, FieldID::N, {}, { 1, 2 }, buff_idx, smooth);
    }
    Kokkos::deep_copy(m_ppc_per_spec, ZERO);
    {
      const auto C     = m_atm_C;
      const auto h     = m_atm_h;
      const auto rstar = m_psr_Rstar;
      const auto ppc0  = params.ppc0();
      const auto frac  = static_cast<real_t>(0.9) * (ONE - ONE / ppc0);

      // determine how much to inject
      Kokkos::parallel_for(
        "ComputeDeltaNdens",
        mblock.rangeActiveCells(),
        ClassLambda(index_t i1, index_t i2) {
          const auto i1_ = i1 - static_cast<int>(N_GHOSTS),
                     i2_ = i2 - static_cast<int>(N_GHOSTS);
          const auto r   = mblock.metric.x1_Code2Phys(i1_ + HALF);

          m_ppc_per_spec(i1_, i2_) = densityProfile(r, C, h, rstar) *
                                     (r > rstar) * (r < rstar + h);

          const auto actual_ndens = mblock.buff(i1, i2, buff_idx);
          if (frac * m_ppc_per_spec(i1_, i2_) > actual_ndens) {
            m_ppc_per_spec(i1_, i2_) = m_ppc_per_spec(i1_, i2_) - actual_ndens;
          } else {
            m_ppc_per_spec(i1_, i2_) = ZERO;
          }
          // 2 -- for two species
          m_ppc_per_spec(i1_, i2_) = int(ppc0 * m_ppc_per_spec(i1_, i2_) / TWO);
        });

      // injection
      InjectNonUniform<Dim2, PICEngine, ThermalBackground>(params,
                                                           mblock,
                                                           { 1, 2 },
                                                           m_ppc_per_spec);
    }
  }
} // namespace ntt

#endif
