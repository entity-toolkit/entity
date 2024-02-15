
#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

namespace ntt {
  enum FieldMode { MonopoleField = 1, DipoleField = 2 };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params)
      : r_surf { params.get<real_t>("problem", "r_surf", (real_t)(1.0)) },
        b_surf { params.get<real_t>("problem", "b_surf", (real_t)(1.0)) },
        spin_omega { params.get<real_t>("problem", "spin_omega") },
        spinup_time { params.get<real_t>("problem", "spinup_time", 0.0) },
        inj_fraction { params.get<real_t>("problem", "inj_fraction", (real_t)(0.1)) },
        field_mode { params.get<int>("problem", "field_mode", 2) == 2 ? DipoleField
                                                                      : MonopoleField },
        inj_rmax { params.get<real_t>("problem", "inj_rmax", (real_t)(1.5)) } {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserDriveFields(const real_t&,
                                const SimulationParams&,
                                Meshblock<D, S>&) override {}
    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

  private:
    const real_t    r_surf, b_surf, spin_omega, spinup_time, inj_fraction, inj_rmax;
    const FieldMode field_mode;
  };

  Inline void mainBField(const coord_t<Dim2>& x_ph,
                         vec_t<Dim3>&,
                         vec_t<Dim3>& b_out,
                         real_t       _rsurf,
                         real_t       _bsurf,
                         int          _mode) {
    if (_mode == 2) {
      b_out[0] = _bsurf * math::cos(x_ph[1]) / CUBE(x_ph[0] / _rsurf);
      b_out[1] = _bsurf * HALF * math::sin(x_ph[1]) / CUBE(x_ph[0] / _rsurf);
      b_out[2] = ZERO;
    } else {
      b_out[0] = _bsurf * SQR(_rsurf / x_ph[0]);
      b_out[1] = ZERO;
      b_out[2] = ZERO;
    }
  }

  Inline void surfaceRotationField(const coord_t<Dim2>& x_ph,
                                   vec_t<Dim3>&         e_out,
                                   vec_t<Dim3>&         b_out,
                                   real_t               _rsurf,
                                   real_t               _bsurf,
                                   int                  _mode,
                                   real_t               _omega) {
    mainBField(x_ph, e_out, b_out, _rsurf, _bsurf, _mode);
    e_out[0] = _omega * b_out[1] * x_ph[0] * math::sin(x_ph[1]);
    e_out[1] = -_omega * b_out[0] * x_ph[0] * math::sin(x_ph[1]);
    e_out[2] = 0.0;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    const auto _rsurf = r_surf;
    {
      const auto    _rmin = mblock.metric.x1_min;
      coord_t<Dim2> x_ph { r_surf, ZERO };
      coord_t<Dim2> xi { ZERO };
      mblock.metric.x_Sph2Code(x_ph, xi);
      NTTHostErrorIf(_rmin >= _rsurf, "rmin > r_surf");
      NTTHostErrorIf(xi[0] < params.currentFilters(), "r_surf - rmin < filters");
    }
    const auto _bsurf = b_surf;
    const int  _mode  = field_mode;
    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, mainBField, _rsurf, _bsurf, _mode);
      });
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveFields(
    const real_t& time, const SimulationParams&, Meshblock<Dim2, PICEngine>& mblock) {
    {
      coord_t<Dim2> x_ph { r_surf, ZERO };
      coord_t<Dim2> xi { ZERO };
      mblock.metric.x_Sph2Code(x_ph, xi);
      const auto i1_surf = (unsigned int)(xi[0] + N_GHOSTS);
      const auto _mode   = field_mode;
      const auto _rsurf  = r_surf;
      const auto _bsurf  = b_surf;
      const auto i1_min  = mblock.i1_min();
      const auto _omega
        = (time < spinup_time) ? (time / spinup_time) * spin_omega : spin_omega;

      Kokkos::parallel_for(
        "UserDriveFields_rmin",
        CreateRangePolicy<Dim2>({ i1_min, mblock.i2_min() }, { i1_surf, mblock.i2_max() }),
        Lambda(index_t i1, index_t i2) {
          set_ex2_2d(mblock, i1, i2, surfaceRotationField, _rsurf, _bsurf, _mode, _omega);
          set_ex3_2d(mblock, i1, i2, surfaceRotationField, _rsurf, _bsurf, _mode, _omega);
          set_bx1_2d(mblock, i1, i2, surfaceRotationField, _rsurf, _bsurf, _mode, _omega);
          if (i1 < i1_surf - 1) {
            set_ex1_2d(mblock, i1, i2, surfaceRotationField, _rsurf, _bsurf, _mode, _omega);
            set_bx2_2d(mblock, i1, i2, surfaceRotationField, _rsurf, _bsurf, _mode, _omega);
            set_bx3_2d(mblock, i1, i2, surfaceRotationField, _rsurf, _bsurf, _mode, _omega);
          }
        });
    }
  }

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : TargetFields<D, S>(params, mblock),
        _rsurf { params.get<real_t>("problem", "r_surf", (real_t)(1.0)) },
        _bsurf { params.get<real_t>("problem", "b_surf", (real_t)(1.0)) },
        _mode { params.get<int>("problem", "field_mode", 2) } {}
    Inline real_t operator()(const em& comp, const coord_t<D>& xi) const override {
      if ((comp == em::bx1) || (comp == em::bx2)) {
        vec_t<Dim3> e_out { ZERO }, b_out { ZERO };
        coord_t<D>  x_ph { ZERO };
        (this->m_mblock).metric.x_Code2Sph(xi, x_ph);
        mainBField(x_ph, e_out, b_out, _rsurf, _bsurf, _mode);
        return (comp == em::bx1) ? b_out[0] : b_out[1];
      } else {
        return ZERO;
      }
    }

  private:
    const real_t _rsurf, _bsurf;
    const int    _mode;
  };

  template <Dimension D, SimulationEngine S>
  struct RadialKick : public EnergyDistribution<D, S> {
    RadialKick(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        u_kick { params.get<real_t>("problem", "u_kick", ZERO) } {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v, const int&) const override {
      v[0] = u_kick;
    }

  private:
    const real_t u_kick;
  };

  template <Dimension D, SimulationEngine S>
  struct InjectionShell : public SpatialDistribution<D, S> {
    explicit InjectionShell(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock),
        _inj_rmin { params.get<real_t>("problem", "r_surf", (real_t)(1.0)) },
        _inj_rmax { params.get<real_t>("problem", "inj_rmax", (real_t)(1.5)) } {
      NTTHostErrorIf(_inj_rmin >= _inj_rmax, "inj_rmin >= inj_rmax");
    }
    Inline real_t operator()(const coord_t<D>& x_ph) const {
      return ((x_ph[0] <= _inj_rmax) && (x_ph[0] > _inj_rmin)) ? ONE : ZERO;
    }

  private:
    const real_t _inj_rmin, _inj_rmax;
  };

  template <Dimension D, SimulationEngine S>
  struct MaxDensCrit : public InjectionCriterion<D, S> {
    explicit MaxDensCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock),
        _inj_maxdens { params.get<real_t>("problem", "inj_maxdens", (real_t)(5.0)) } {}
    Inline bool operator()(const coord_t<D>&) const {
      return true;
    }

  private:
    const real_t _inj_maxdens;
  };

  template <>
  Inline bool MaxDensCrit<Dim2, PICEngine>::operator()(const coord_t<Dim2>& xph) const {
    coord_t<Dim2> xi { ZERO };
    (this->m_mblock).metric.x_Sph2Code(xph, xi);
    auto i1 = (std::size_t)(xi[0]) + N_GHOSTS;
    auto i2 = (std::size_t)(xi[1]) + N_GHOSTS;
    if (i1 < (this->m_mblock).buff.extent(0) && i2 < (this->m_mblock).buff.extent(1)) {
      // return true;
      return (this->m_mblock).buff(i1, i2, 2) < _inj_maxdens;
    } else {
      return false;
    }
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const real_t&, const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    mblock.ComputeMoments(params, FieldID::Rho, {}, { 1, 2 }, 2, 0);
    WaitAndSynchronize();
    auto nppc_per_spec = (real_t)(params.ppc0()) * inj_fraction * HALF;
    InjectInVolume<Dim2, PICEngine, RadialKick, InjectionShell, MaxDensCrit>(
      params,
      mblock,
      { 1, 2 },
      nppc_per_spec,
      { mblock.metric.x1_min, inj_rmax, mblock.metric.x2_min, mblock.metric.x2_max });
  }

}    // namespace ntt

#endif