#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

#include <time.h>

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
        temperature { params.get<real_t>("problem", "temperature", 0.1) } {}
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
  public:
    real_t work_done { ZERO };
    // read additional parameters from the input file [default to 1 if not specified]
    inline ProblemGenerator(const SimulationParams& params)
      : temperature { params.get<real_t>("problem", "temperature", 0.1) },
        _atmo_max { params.get<real_t>("problem", "atmo_max", (real_t)(1.0)) },
        _atmo_rmin { params.get<real_t>("problem", "atmo_rmin", (real_t)(0.1)) },
        _atmo_stretch { params.get<real_t>("problem", "atmo_stretch", (real_t)(50.0)) } {}
    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}
    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserInitParticles(const SimulationParams& params,
                                  Meshblock<D, S>&        mblock) override {}
#ifdef EXTERNAL_FORCE
    Inline auto ext_force_x1(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {

      real_t sign = tanh(_atmo_stretch*x_ph[0]);
      real_t gacc = _atmo_stretch*temperature*sign;
      return gacc;
      
    }
    Inline auto ext_force_x2(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {
      return ZERO;
    }
    Inline auto ext_force_x3(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {
      return ZERO;
    }
#endif

#ifdef GUI_ENABLED
    inline void UserInitBuffers_nttiny(
      const SimulationParams&,
      const Meshblock<D, S>&,
      std::map<std::string, nttiny::ScrollingBuffer>& buffers) override {
      nttiny::ScrollingBuffer NPRTL;
      buffers.insert({ "NPRTL", std::move(NPRTL) });
      // nttiny::ScrollingBuffer PRTL_energy;
      // nttiny::ScrollingBuffer FORCE_energy;
      // nttiny::ScrollingBuffer TOTAL_energy;
      // buffers.insert({ "EM", std::move(EM_energy) });
      // buffers.insert({ "PRTL", std::move(PRTL_energy) });
      // buffers.insert({ "FORCE", std::move(FORCE_energy) });
      // buffers.insert({ "TOTAL", std::move(TOTAL_energy) });
    }

    inline void UserSetBuffers_nttiny(
      const real_t&                                   time,
      const SimulationParams&                         params,
      Meshblock<D, S>&                                mblock,
      std::map<std::string, nttiny::ScrollingBuffer>& buffers) override {
      buffers["NPRTL"].AddPoint(time,
                                mblock.particles[0].npart() + mblock.particles[1].npart());
      // if constexpr (D == Dim2) {
      //   real_t em_sum { ZERO }, prtl_sum { ZERO };
      //   Kokkos::parallel_reduce(
      //     "EMEnergy",
      //     mblock.rangeActiveCells(),
      //     Lambda(index_t i, index_t j, real_t & sum) {
      //       const real_t i_ = static_cast<real_t>(i);
      //       const real_t j_ = static_cast<real_t>(j);
      //       vec_t<Dim3>  E { ZERO }, B { ZERO };
      //       mblock.metric.v3_Cntrv2Hat(
      //         { i_ + HALF, j_ + HALF },
      //         { mblock.em(i, j, em::ex1), mblock.em(i, j, em::ex2), mblock.em(i, j, em::ex3)
      //         }, E);
      //       mblock.metric.v3_Cntrv2Hat(
      //         { i_ + HALF, j_ + HALF },
      //         { mblock.em(i, j, em::bx1), mblock.em(i, j, em::bx2), mblock.em(i, j, em::bx3)
      //         }, B);
      //       sum += (SQR(E[0]) + SQR(E[1]) + SQR(E[2]) + SQR(B[0]) + SQR(B[1]) + SQR(B[2]))
      //              * HALF;
      //     },
      //     em_sum);

      //   em_sum /= SQR(params.larmor0());
      //   em_sum *= mblock.metric.min_cell_volume();

      //   for (auto& species : mblock.particles) {
      //     real_t global_a = ZERO;
      //     Kokkos::parallel_reduce(
      //       "ParticleEnergy",
      //       species.npart(),
      //       Lambda(index_t p, real_t & sum) {
      //         sum += math::sqrt(ONE + SQR(species.ux1(p)) + SQR(species.ux2(p))
      //                           + SQR(species.ux3(p)));
      //       },
      //       global_a);
      //     prtl_sum += global_a;
      //   }
      //   buffers["EM"].AddPoint(time, em_sum);
      //   buffers["PRTL"].AddPoint(time, prtl_sum);
      //   buffers["FORCE"].AddPoint(time, work_done);
      //   buffers["TOTAL"].AddPoint(time, em_sum + prtl_sum - work_done);
      // }
    }
#endif    // GUI_ENABLED
  private:
    // additional problem-specific parameters (i.e., wave numbers in x1, x2, x3 directions)
    const real_t temperature, _atmo_max, _atmo_rmin, _atmo_stretch;
  };

  Inline void background_fields_2d(const coord_t<Dim2>& x_ph,     // physical coordinate
                                   vec_t<Dim3>&         e_out,    // electric field [out]
                                   vec_t<Dim3>&         b_out,    // magnetic field [out]
                                   real_t               time) {
    // fields in physical units
    // (
    //    in physical units, when B = 1, and |q|/m = 1
    //    the larmor radius is equal to the fiducial value
    //    i.e., `m c^2 / |q| B == larmor0`
    // )
    e_out[0] = 0.0;
    e_out[1] = 0.0;
    e_out[2] = 0.0;
    // some background magnetic field goes here [TO BE MODIFIED]
    b_out[0] = 0.0;
    b_out[1] = 0.0;
    b_out[2] = 0.0;
  }

  Inline void background_fields_3d(const coord_t<Dim3>& x_ph,     // physical coordinate
                                   vec_t<Dim3>&         e_out,    // electric field [out]
                                   vec_t<Dim3>&         b_out,    // magnetic field [out]
                                   real_t               time) {
    e_out[0] = 0.0;
    e_out[1] = 0.0;
    e_out[2] = 0.0;
    // some background magnetic field goes here [TO BE MODIFIED]
    b_out[0] = 0.0;
    b_out[1] = 0.0;
    b_out[2] = 0.0;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitParticles(
    const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    const auto _time      = this->time();

    auto       delta_nppc = array_t<real_t**>(
      "delta_nppc", mblock.Ni1() + 2 * N_GHOSTS, mblock.Ni2() + 2 * N_GHOSTS);

    const auto ppc0 = params.ppc0();
    const auto xmax = mblock.metric.x1_max;
    const auto xmin = mblock.metric.x1_min;

    Kokkos::parallel_for(
      "ComputeDeltaNPPC", mblock.rangeActiveCells(), ClassLambda(index_t i, index_t j) {
        const real_t  i1_ { static_cast<real_t>(static_cast<int>(i)) };
        const real_t  i2_ { static_cast<real_t>(static_cast<int>(j)) };
        auto          _atmo_max_m { this->_atmo_max };
        auto          _atmo_stretch_m { this->_atmo_stretch };
        auto          _atmo_rmin_m { this->_atmo_rmin };
        coord_t<Dim2> x_ph { ZERO };
        mblock.metric.x_Code2Cart({ i1_, i2_ }, x_ph);
        if(x_ph[0] < xmin + _atmo_rmin_m || x_ph[0] > xmax - _atmo_rmin_m) {
          delta_nppc(i, j) = ZERO;
        } else {
        delta_nppc(i, j)
          = ppc0 * _atmo_max_m
            * (math::exp(-(_atmo_stretch_m * (- xmin + x_ph[0] - _atmo_rmin_m)))
               + math::exp(-(_atmo_stretch_m * (xmax - x_ph[0] - _atmo_rmin_m))));
        }
      });

    InjectToFloor<Dim2, PICEngine, ThermalBackground>(params, mblock, { 1, 2 }, delta_nppc);
  }

  /**
   * Field initialization for 2D:
   */
  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams&, Meshblock<Dim2, PICEngine>& mblock) {
    const auto _time = this->time();

    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, background_fields_2d, _time);
      });
  }

  /**
   * Field initialization for 3D:
   */
  template <>
  inline void ProblemGenerator<Dim3, PICEngine>::UserInitFields(
    const SimulationParams&, Meshblock<Dim3, PICEngine>& mblock) {
    const auto _time = this->time();

    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j, index_t k) {
        set_em_fields_3d(mblock, i, j, k, background_fields_3d, _time);
      });
  }

  // /**
  //  *
  //  */
  // template <>
  // inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
  //   const real_t&, const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
  //   const auto _time = this->time();
  // }

  // /**
  //  *
  //  */
  // template <>
  // inline void ProblemGenerator<Dim3, PICEngine>::UserDriveParticles(
  //   const real_t&, const SimulationParams& params, Meshblock<Dim3, PICEngine>& mblock) {
  //   const auto _time = this->time();
  // }

  template <Dimension D, SimulationEngine S>
  struct MaxDensCrit : public InjectionCriterion<D, S> {
    explicit MaxDensCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock) {}
    Inline bool operator()(const coord_t<D>&) const {
      return true;
    }
  };

  template <>
  Inline bool MaxDensCrit<Dim2, PICEngine>::operator()(const coord_t<Dim2>& xph) const {
    return true;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const real_t& time, const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    auto delta_nppc = array_t<real_t**>(
      "delta_nppc", mblock.Ni1() + 2 * N_GHOSTS, mblock.Ni2() + 2 * N_GHOSTS);

    mblock.ComputeMoments(params, FieldID::Nppc, {}, { 1, 2 }, 0, 0);
    const auto ppc0 = params.ppc0();
    const auto xmax = mblock.metric.x1_max;
    const auto xmin = mblock.metric.x1_min;
    auto          _atmo_max_m { this->_atmo_max };
    auto          _atmo_stretch_m { this->_atmo_stretch };
    auto          _atmo_rmin_m { this->_atmo_rmin };

    Kokkos::parallel_for(
      "ComputeDeltaNPPC", mblock.rangeActiveCells(), ClassLambda(index_t i, index_t j) {
        const real_t  i1_ { static_cast<real_t>(static_cast<int>(i)) };
        const real_t  i2_ { static_cast<real_t>(static_cast<int>(j)) };
        coord_t<Dim2> x_ph { ZERO };
        mblock.metric.x_Code2Cart({ i1_, i2_ }, x_ph);
        if(x_ph[0] < xmin + _atmo_rmin_m || x_ph[0] > xmax - _atmo_rmin_m) {
          delta_nppc(i, j) = ZERO;
        } else {
        delta_nppc(i, j)
          = ppc0 * _atmo_max_m
            * (math::exp(-(_atmo_stretch_m * (- xmin + x_ph[0] - _atmo_rmin_m)))
               + math::exp(-(_atmo_stretch_m * (xmax - x_ph[0] - _atmo_rmin_m))));
        delta_nppc(i, j) -= mblock.buff(i, j, 0);
        }
      });

    InjectToFloor<Dim2, PICEngine, ThermalBackground>(params, mblock, { 1, 2 }, delta_nppc);

    for (auto& species : mblock.particles) {
      Kokkos::parallel_for(
        "prtl_bc", species.rangeActiveParticles(), ClassLambda(index_t p) {
        const real_t  i1_ { static_cast<real_t>(species.i1(p) + N_GHOSTS) };
        const real_t  i2_ { static_cast<real_t>(species.i2(p) + N_GHOSTS)};
        coord_t<Dim2> x_ph { ZERO };
        mblock.metric.x_Code2Cart({ i1_, i2_ }, x_ph);
        if(x_ph[0] < xmin + 0.5 * _atmo_rmin_m || x_ph[0] > xmax - 0.5 * _atmo_rmin_m) {
            species.tag(p) = static_cast<short>(ParticleTag::dead);
          }
        });
    }

  }

}    // namespace ntt

#endif