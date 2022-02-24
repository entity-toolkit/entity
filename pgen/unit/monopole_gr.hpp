#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

#include <stdexcept>

#if SIMTYPE == GRPIC_SIMTYPE

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator {
    ProblemGenerator(const SimulationParams& sim_params);
    real_t epsilon {static_cast<real_t>(1e-1)};
    // RealFieldND<D, 1> Bru0;

    void userInitFields(const SimulationParams&, Meshblock<D, S>&);
    void userInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    void userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&);

    static real_t A0(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      (void) mblock;
      (void) x;
      return ZERO;
    }

    static real_t A1(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      (void) mblock;
      (void) x;
      return ZERO;
    }
    
    static real_t A3(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      coord_t<D> rth_;
      mblock.metric.x_Code2Sph(x, rth_);
      return ONE - std::cos(rth_[1]) ;
    }

    Inline auto userTargetField_br_hat(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      coord_t<D> rth_;
      real_t r_min {mblock.metric.x1_min};
      mblock.metric.x_Code2Sph(x, rth_);
      return ZERO;
      }

    Inline auto userTargetField_bth_hat(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      return ZERO;
    }

    Inline auto userTargetField_br_cntrv(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      coord_t<D> x0m, x0p;
      real_t inv_sqrt_detH_ijP  {ONE / mblock.metric.sqrt_det_h(x)};
      x0m[0] = x[0], x0m[1] = x[1] - HALF * epsilon;
      x0p[0] = x[0], x0p[1] = x[1] + HALF * epsilon;
      return (A3(mblock, x0p) - A3(mblock, x0m)) * inv_sqrt_detH_ijP / epsilon;
    }

  };

} // namespace ntt

#else
  NTTError("Problem generator relevant in GRPIC only.");
#endif

#endif
