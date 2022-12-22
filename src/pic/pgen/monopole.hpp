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
      v[0] = 0.1;
    }
  };

  template <Dimension D, SimulationType S>
  struct RadialDist : public SpatialDistribution<D, S> {
    explicit RadialDist(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {
      inj_rmax = readFromInput<real_t>(params.inputdata(), "problem", "inj_rmax", 1.1);
    }
    Inline real_t operator()(const coord_t<D>&) const;

  private:
    real_t inj_rmax;
  };

  template <Dimension D, SimulationType S>
  struct EdotBCrit : public InjectionCriterion<D, S> {
    explicit EdotBCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock) {
      inj_maxEdotB
        = readFromInput<real_t>(params.inputdata(), "problem", "inj_maxEdotB", 0.01);
    }
    Inline bool operator()(const coord_t<D>& xi) const;

  private:
    real_t inj_maxEdotB;
  };

  template <>
  Inline real_t RadialDist<Dim2, TypePIC>::operator()(const coord_t<Dim2>& x_ph) const {
    return (x_ph[0] <= inj_rmax) ? ONE : ZERO;
  }

  template <>
  Inline bool EdotBCrit<Dim2, TypePIC>::operator()(const coord_t<Dim2>& xi) const {
    std::size_t i1 = (std::size_t)(xi[0] + N_GHOSTS);
    std::size_t i2 = (std::size_t)(xi[1] + N_GHOSTS);
    vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO };
    this->m_mblock.metric.v_Cntrv2Hat(xi,
                                      { this->m_mblock.em(i1, i2, em::ex1),
                                        this->m_mblock.em(i1, i2, em::ex2),
                                        this->m_mblock.em(i1, i2, em::ex3) },
                                      e_hat);
    this->m_mblock.metric.v_Cntrv2Hat(xi,
                                      { this->m_mblock.em(i1, i2, em::bx1),
                                        this->m_mblock.em(i1, i2, em::bx2),
                                        this->m_mblock.em(i1, i2, em::bx3) },
                                      b_hat);
    real_t Bsqr  = SQR(b_hat[0]) + SQR(b_hat[1]) + SQR(b_hat[2]);
    real_t EdotB = e_hat[0] * b_hat[0] + e_hat[1] * b_hat[1] + e_hat[2] * b_hat[2];
    return ABS(EdotB / Bsqr) > inj_maxEdotB;
  }

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) {
      spinup_time  = readFromInput<real_t>(params.inputdata(), "problem", "spinup_time");
      omega_max    = readFromInput<real_t>(params.inputdata(), "problem", "omega_max");
      inj_fraction = readFromInput<real_t>(params.inputdata(), "problem", "inj_fraction");
      inj_rmax     = readFromInput<real_t>(params.inputdata(), "problem", "inj_rmax", 1.1);
    }

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
      InjectInVolume<D, S, ColdDist, RadialDist, NoCriterion>(
        params, mblock, { 1, 2 }, nppc_per_spec, {}, time);
    }
    // inline void UserDriveParticles(const real_t&           time,
    //                                const SimulationParams& params,
    //                                Meshblock<D, S>&        mblock) override {
    //   auto nppc_per_spec  = (real_t)(params.ppc0()) * inj_fraction;
    //   auto ncells         = (std::size_t)(mblock.Ni1() * mblock.Ni2() * mblock.Ni3());
    //   auto npart_per_spec = (std::size_t)((double)(ncells * nppc_per_spec));
    //   if constexpr (D == Dim2) {
    //     auto  random_pool   = *(mblock.random_pool_ptr);
    //     auto& sp1           = mblock.particles[0];
    //     auto& sp2           = mblock.particles[1];
    //     auto  npart_before1 = sp1.npart();
    //     auto  npart_before2 = sp2.npart();
    //     sp1.setNpart(npart_before1 + npart_per_spec);
    //     sp2.setNpart(npart_before2 + npart_per_spec);
    //     Kokkos::parallel_for(
    //       "inject", CreateRangePolicy<Dim1>({ 0 }, { npart_per_spec }), Lambda(index_t p) {
    //         typename RandomNumberPool_t::generator_type rand_gen = random_pool.get_state();
    //         coord_t<Dim2>                               x { ZERO };
    //         x[0] = rand_gen.frand(1.05, 1.1);
    //         x[1] = rand_gen.frand(0.1, constant::PI - 0.1);
    //         init_prtl_2d(mblock, sp1, p + npart_before1, x[0], x[1], 0.0, 0.0, 0.0);
    //         init_prtl_2d(mblock, sp2, p + npart_before2, x[0], x[1], 0.0, 0.0, 0.0);
    //         random_pool.free_state(rand_gen);
    //       });
    //   }
    //   // InjectUniform<D, TypePIC, ColdDist>(
    //   //   params, mblock, { 1, 2 }, nppc_per_spec, { 1.0, inj_rmax, 0.0, constant::PI },
    //   //   time);
    // }

    inline real_t UserTargetField(const Meshblock<D, S>&,
                                  const em&,
                                  const real_t&,
                                  const coord_t<D>&) const override;

  private:
    real_t spinup_time, omega_max, inj_fraction, inj_rmax;
  };

  template <Dimension D, SimulationType S>
  inline real_t ProblemGenerator<D, S>::UserTargetField(const Meshblock<D, S>& mblock,
                                                        const em&              comp,
                                                        const real_t&          time,
                                                        const coord_t<D>&      xi) const {
    if (comp == em::bx1) {
      coord_t<D> rth_ { ZERO };
      real_t     r_min { mblock.metric.x1_min };
      mblock.metric.x_Code2Sph(xi, rth_);
      return ONE * SQR(r_min / rth_[0]);
    } else {
      return ZERO;
    }
  }

  Inline void monopoleField(const coord_t<Dim2>& x_ph,
                            vec_t<Dim3>&         e_out,
                            vec_t<Dim3>&         b_out,
                            real_t               rmin) {
    b_out[0] = SQR(rmin / x_ph[0]);
  }

  Inline void surfaceRotationField(const coord_t<Dim2>& x_ph,
                                   vec_t<Dim3>&         e_out,
                                   vec_t<Dim3>&         b_out,
                                   real_t               rmin,
                                   real_t               omega) {
    b_out[0] = SQR(rmin / x_ph[0]);
    e_out[1] = omega * math::sin(x_ph[1]);
    e_out[2] = 0.0;
  }

  template <>
  inline void ProblemGenerator<Dim2, TypePIC>::UserInitFields(
    const SimulationParams&, Meshblock<Dim2, TypePIC>& mblock) {
    auto r_min = mblock.metric.x1_min;
    Kokkos::parallel_for(
      "UserInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, monopoleField, r_min);
      });
  }

  template <>
  inline void ProblemGenerator<Dim2, TypePIC>::UserDriveFields(
    const real_t& time, const SimulationParams&, Meshblock<Dim2, TypePIC>& mblock) {
    real_t omega;
    auto   r_min = mblock.metric.x1_min;
    if (time < spinup_time) {
      omega = omega_max * time / spinup_time;
    } else {
      omega = omega_max;
    }
    Kokkos::parallel_for(
      "UserBcFlds_rmin",
      CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() },
                              { mblock.i1_min() + 1, mblock.i2_max() }),
      Lambda(index_t i, index_t j) {
        set_ex2_2d(mblock, i, j, surfaceRotationField, r_min, omega);
        set_ex3_2d(mblock, i, j, surfaceRotationField, r_min, omega);
        set_bx1_2d(mblock, i, j, surfaceRotationField, r_min, omega);
      });

    Kokkos::parallel_for(
      "UserBcFlds_rmax",
      CreateRangePolicy<Dim2>({ mblock.i1_max(), mblock.i2_min() },
                              { mblock.i1_max() + 1, mblock.i2_max() }),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::ex3) = 0.0;
        mblock.em(i, j, em::ex2) = 0.0;
        mblock.em(i, j, em::bx1) = 0.0;
      });
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
  Inline bool EdotBCrit<Dim3, TypePIC>::operator()(const coord_t<Dim3>&) const {
    return false;
  }
  template <>
  Inline bool EdotBCrit<Dim1, TypePIC>::operator()(const coord_t<Dim1>&) const {
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