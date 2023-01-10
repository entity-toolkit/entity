#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "input.h"
#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "injector.hpp"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct RadialKick : public EnergyDistribution<D, S> {
    RadialKick(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v) const override {
      v[0] = 0.5;
    }
  };

  template <Dimension D, SimulationType S>
  struct RadialDist : public SpatialDistribution<D, S> {
    explicit RadialDist(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {
      inj_rmax              = readFromInput<real_t>(params.inputdata(), "problem", "inj_rmax");
      auto       buff_cells = readFromInput<int>(params.inputdata(), "problem", "buff_cells");
      coord_t<D> xcu { ZERO }, xph { ZERO };
      xcu[0] = (real_t)buff_cells;
      mblock.metric.x_Code2Sph(xcu, xph);
      inj_rmin = xph[0];
    }
    Inline real_t operator()(const coord_t<D>&) const;

  private:
    real_t inj_rmax, inj_rmin;
  };

  template <Dimension D, SimulationType S>
  struct MaxDensCrit : public InjectionCriterion<D, S> {
    explicit MaxDensCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock),
        inj_maxDens { readFromInput<real_t>(
          params.inputdata(), "problem", "inj_maxDens", 5.0) },
        ppc0 { params.ppc0() } {}
    Inline bool operator()(const coord_t<D>&) const;

  private:
    const real_t inj_maxDens, ppc0;
  };

  template <>
  Inline real_t RadialDist<Dim2, TypePIC>::operator()(const coord_t<Dim2>& x_ph) const {
    return ((x_ph[0] <= inj_rmax) && (x_ph[0] > inj_rmin)) ? ONE : ZERO;
  }

  template <Dimension D, SimulationType S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : TargetFields<D, S>(params, mblock), r_min { mblock.metric.x1_min } {}
    Inline real_t operator()(const em& comp, const coord_t<D>& xi) const override {
      if (comp == em::bx1) {
        coord_t<D> x_ph { ZERO };
        this->m_mblock.metric.x_Code2Sph(xi, x_ph);
        return SQR(r_min / x_ph[0]);
      } else {
        return ZERO;
      }
    }

  private:
    const real_t r_min;
  };

  Inline void monopoleField(const coord_t<Dim2>& x_ph,
                            vec_t<Dim3>&,
                            vec_t<Dim3>& b_out,
                            real_t       rmin) {
    b_out[0] = SQR(rmin / x_ph[0]);
  }

  Inline void surfaceRotationField(const coord_t<Dim2>& x_ph,
                                   vec_t<Dim3>&         e_out,
                                   vec_t<Dim3>&         b_out,
                                   real_t               rmin,
                                   real_t               omega) {
    monopoleField(x_ph, e_out, b_out, rmin);
    e_out[0] = 0.0;
    e_out[1] = omega * math::sin(x_ph[1]);
    e_out[2] = 0.0;
  }

  template <>
  Inline bool MaxDensCrit<Dim2, TypePIC>::operator()(const coord_t<Dim2>& xph) const {
    coord_t<Dim2> xi { ZERO };
    this->m_mblock.metric.x_Sph2Code(xph, xi);
    std::size_t i1 = (std::size_t)(xi[0] + N_GHOSTS);
    std::size_t i2 = (std::size_t)(xi[1] + N_GHOSTS);
    return this->m_mblock.buff(i1, i2, fld::dens) < inj_maxDens;
  }

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params)
      : spin_omega { readFromInput<real_t>(params.inputdata(), "problem", "spin_omega") },
        inj_fraction { readFromInput<real_t>(params.inputdata(), "problem", "inj_fraction") },
        inj_rmax { readFromInput<real_t>(params.inputdata(), "problem", "inj_rmax") },
        buff_cells { readFromInput<int>(params.inputdata(), "problem", "buff_cells") } {}

    inline void UserInitParticles(const SimulationParams& params,
                                  Meshblock<D, S>&        mblock) override {}
    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override;
    inline void UserDriveFields(const real_t&,
                                const SimulationParams&,
                                Meshblock<D, S>&) override;
    inline void UserDriveParticles(const real_t&           time,
                                   const SimulationParams& params,
                                   Meshblock<D, S>&        mblock) override {
      auto nppc_per_spec = (real_t)(params.ppc0()) * inj_fraction;
      InjectInVolume<D, S, RadialKick, RadialDist, MaxDensCrit>(
        params, mblock, { 1, 2 }, nppc_per_spec, {}, time);
    }

  private:
    const real_t spin_omega, inj_fraction, inj_rmax;
    const int    buff_cells;
  };

  template <>
  inline void ProblemGenerator<Dim2, TypePIC>::UserInitFields(
    const SimulationParams&, Meshblock<Dim2, TypePIC>& mblock) {
    auto rmin = mblock.metric.x1_min;
    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, monopoleField, rmin);
      });
  }

  template <>
  inline void ProblemGenerator<Dim2, TypePIC>::UserDriveFields(
    const real_t& time, const SimulationParams&, Meshblock<Dim2, TypePIC>& mblock) {
    {
      // Set the boundary conditions at r-min
      const auto omega  = spin_omega;
      const auto rmin   = mblock.metric.x1_min;
      const auto i1_min = mblock.i1_min();
      const auto i1_max = mblock.i1_min() + buff_cells;
      if (buff_cells > mblock.Ni1()) {
        NTTHostError("buff_cells > ni1");
      }

      /**
       *
       *    ...........................................
       *    .                                         .
       *    .                                         .
       *    .  ^===================================^  .
       *    .  |******                             \  .
       *    .  |******                             \  .
       *    .  |******                             \  .
       *    .  |******                             \  .
       *    .  |******                             \  .
       *    .  |******                             \  .
       *    .  |******                             \  .
       *    .  ^-----------------------------------^  .
       *    .  |______|                               .
       *    .      |                                  .
       *    .......|...................................
       *           |
       *      buff_cells
       *
       */
      Kokkos::parallel_for(
        "UserDriveFields_rmin",
        CreateRangePolicy<Dim2>({ i1_min, mblock.i2_min() }, { i1_max, mblock.i2_max() }),
        Lambda(index_t i, index_t j) {
          if (i < i1_max - 1) {
            mblock.em(i, j, em::ex1) = ZERO;
          }
          set_ex2_2d(mblock, i, j, surfaceRotationField, rmin, omega);
          set_ex3_2d(mblock, i, j, surfaceRotationField, rmin, omega);
          set_bx1_2d(mblock, i, j, surfaceRotationField, rmin, omega);
        });
    }
    {
      // Set the boundary conditions at r-max
      const auto i1_max = mblock.i1_max();
      /**
       *
       *    ...........................................
       *    .                                         .
       *    .                                         .
       *    .  ^===================================^  .
       *    .  |                                   \* .
       *    .  |                                   \* .
       *    .  |                                   \* .
       *    .  |                                   \* .
       *    .  |                                   \* .
       *    .  |                                   \* .
       *    .  |                                   \* .
       *    .  ^-----------------------------------^  .
       *    .                                         .
       *    .                                         .
       *    ...........................................
       *
       */
      Kokkos::parallel_for(
        "UserDriveFields_rmax",
        CreateRangePolicy<Dim1>({ mblock.i2_min() }, { mblock.i2_max() }),
        Lambda(index_t j) {
          mblock.em(i1_max, j, em::ex2) = 0.0;
          mblock.em(i1_max, j, em::ex3) = 0.0;
          mblock.em(i1_max, j, em::bx1) = 0.0;
        });
    }
  }

  /**
   *  1D and 3D dummy functions
   */

  template <>
  Inline real_t RadialDist<Dim1, TypePIC>::operator()(const coord_t<Dim1>&) const {
    return ZERO;
  }
  template <>
  Inline real_t RadialDist<Dim3, TypePIC>::operator()(const coord_t<Dim3>&) const {
    return ZERO;
  }
  template <>
  Inline bool MaxDensCrit<Dim1, TypePIC>::operator()(const coord_t<Dim1>&) const {
    return false;
  }
  template <>
  Inline bool MaxDensCrit<Dim3, TypePIC>::operator()(const coord_t<Dim3>&) const {
    return false;
  }

  template <>
  inline void ProblemGenerator<Dim1, TypePIC>::UserInitFields(const SimulationParams&,
                                                              Meshblock<Dim1, TypePIC>&) {}

  template <>
  inline void ProblemGenerator<Dim1, TypePIC>::UserDriveFields(const real_t&,
                                                               const SimulationParams&,
                                                               Meshblock<Dim1, TypePIC>&) {}

  template <>
  inline void ProblemGenerator<Dim3, TypePIC>::UserInitFields(const SimulationParams&,
                                                              Meshblock<Dim3, TypePIC>&) {}

  template <>
  inline void ProblemGenerator<Dim3, TypePIC>::UserDriveFields(const real_t&,
                                                               const SimulationParams&,
                                                               Meshblock<Dim3, TypePIC>&) {}

}    // namespace ntt

#endif
