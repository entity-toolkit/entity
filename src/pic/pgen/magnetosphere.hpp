
#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {
  enum FieldMode { MonopoleField = 1, DipoleField = 2 };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params)
      : bsurf { params.get<real_t>("problem", "bsurf", (real_t)(1.0)) },
        inj_fraction { params.get<real_t>("problem", "inj_fraction", (real_t)(0.1)) },
        field_mode { params.get<int>("problem", "field_mode", 2) == 2 ? DipoleField
                                                                      : MonopoleField } {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserDriveFields(const real_t&,
                                const SimulationParams&,
                                Meshblock<D, S>&) override {}
    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

  private:
    const real_t    bsurf, inj_fraction;
    const FieldMode field_mode;
  };

  Inline void mainBField(const coord_t<Dim2>& x_ph,
                         vec_t<Dim3>&,
                         vec_t<Dim3>& b_out,
                         real_t       rmin,
                         real_t       bsurf,
                         int          mode) {
    if (mode == 2) {
      b_out[0] = bsurf * TWO * math::cos(x_ph[1]) / CUBE(x_ph[0] / rmin);
      b_out[1] = bsurf * math::sin(x_ph[1]) / CUBE(x_ph[0] / rmin);
    } else {
      b_out[0] = bsurf * SQR(rmin / x_ph[0]);
    }
  }

  Inline void surfaceRotationField(const coord_t<Dim2>& x_ph,
                                   vec_t<Dim3>&         e_out,
                                   vec_t<Dim3>&         b_out,
                                   real_t               rmin,
                                   real_t               bsurf,
                                   int                  mode,
                                   real_t               omega) {
    mainBField(x_ph, e_out, b_out, rmin, bsurf, mode);
    e_out[1] = omega * bsurf * math::sin(x_ph[1]);
    e_out[2] = 0.0;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    const auto rmin   = mblock.metric.x1_min;
    const auto bsurf_ = bsurf;
    const int  mode   = field_mode;
    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, mainBField, rmin, bsurf_, mode);
      });
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveFields(
    const real_t& time, const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    {
      const auto spin_omega  = params.get<real_t>("problem", "spin_omega");
      const auto spinup_time = params.get<real_t>("problem", "spinup_time", 0.0);
      const int  mode        = field_mode;
      const auto rmin        = mblock.metric.x1_min;
      const auto bsurf_      = bsurf;
      const auto i1_min      = mblock.i1_min();
      const auto buff_cells  = 5;
      const auto i1_max      = mblock.i1_min() + buff_cells;
      const auto omega = (time < spinup_time) ? (time / spinup_time) * spin_omega : spin_omega;
      NTTHostErrorIf(buff_cells > mblock.Ni1(), "buff_cells > ni1");

      Kokkos::parallel_for(
        "UserDriveFields_rmin",
        CreateRangePolicy<Dim2>({ i1_min, mblock.i2_min() }, { i1_max, mblock.i2_max() }),
        Lambda(index_t i1, index_t i2) {
          if (i1 < i1_max - 1) {
            mblock.em(i1, i2, em::ex1) = ZERO;
          }
          set_ex2_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf_, mode, omega);
          set_ex3_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf_, mode, omega);
          set_bx1_2d(mblock, i1, i2, surfaceRotationField, rmin, bsurf_, mode, omega);
        });
    }
    {
      const auto i1_max = mblock.i1_max();
      Kokkos::parallel_for(
        "UserDriveFields_rmax",
        CreateRangePolicy<Dim1>({ mblock.i2_min() }, { mblock.i2_max() }),
        Lambda(index_t i2) {
          mblock.em(i1_max, i2, em::ex2) = 0.0;
          mblock.em(i1_max, i2, em::ex3) = 0.0;
          mblock.em(i1_max, i2, em::bx1) = 0.0;
        });
    }
  }

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : TargetFields<D, S>(params, mblock),
        rmin { mblock.metric.x1_min },
        bsurf { params.get<real_t>("problem", "bsurf", (real_t)(1.0)) },
        mode { params.get<int>("problem", "field_mode", 2) } {}
    Inline real_t operator()(const em& comp, const coord_t<D>& xi) const override {
      if ((comp == em::bx1) || (comp == em::bx2)) {
        vec_t<Dim3> e_out { ZERO }, b_out { ZERO };
        coord_t<D>  x_ph { ZERO };
        (this->m_mblock).metric.x_Code2Sph(xi, x_ph);
        mainBField(x_ph, e_out, b_out, rmin, bsurf, mode);
        return (comp == em::bx1) ? b_out[0] : b_out[1];
      } else {
        return ZERO;
      }
    }

  private:
    const real_t rmin, bsurf;
    const int    mode;
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
      : SpatialDistribution<D, S>(params, mblock) {
      inj_rmax = params.get<real_t>("problem", "inj_rmax", 1.5 * mblock.metric.x1_min);
      const int  buff_cells = 10;
      coord_t<D> xcu { ZERO }, xph { ZERO };
      xcu[0] = (real_t)buff_cells;
      mblock.metric.x_Code2Sph(xcu, xph);
      inj_rmin = xph[0];
    }
    Inline real_t operator()(const coord_t<D>& x_ph) const {
      return ((x_ph[0] <= inj_rmax) && (x_ph[0] > inj_rmin)
              && ((x_ph[1] > 0.01) && (x_ph[1] <= constant::PI - 0.01)))
               ? ONE
               : ZERO;
    }

  private:
    real_t inj_rmax, inj_rmin;
  };

  template <Dimension D, SimulationEngine S>
  struct MaxDensCrit : public InjectionCriterion<D, S> {
    explicit MaxDensCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock),
        inj_maxDens { params.get<real_t>("problem", "inj_maxDens", 5.0) } {}
    Inline bool operator()(const coord_t<D>&) const {
      return false;
    }

  private:
    const real_t inj_maxDens;
  };

  template <>
  Inline bool MaxDensCrit<Dim2, PICEngine>::operator()(const coord_t<Dim2>& xph) const {
    coord_t<Dim2> xi { ZERO };
    (this->m_mblock).metric.x_Sph2Code(xph, xi);
    std::size_t i1 = (std::size_t)(xi[0] + N_GHOSTS);
    std::size_t i2 = (std::size_t)(xi[1] + N_GHOSTS);
    return (this->m_mblock).buff(i1, i2, fld::dens) < inj_maxDens;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const real_t& time, const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    auto nppc_per_spec = (real_t)(params.ppc0()) * inj_fraction;
    InjectInVolume<Dim2, PICEngine, RadialKick, InjectionShell, MaxDensCrit>(
      params, mblock, { 1, 2 }, nppc_per_spec);
  }

}    // namespace ntt

#undef FIELD_DIPOLE
#undef FIELD_MONOPOLE

#endif