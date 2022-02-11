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
    real_t epsilon;

    void userInitFields(const SimulationParams&, Meshblock<D, S>&);
    void userInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    void userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&);

    static real_t Aphi(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      return HALF * (mblock.metric.h_33(x) + TWO * mblock.metric.spin() * mblock.metric.h_13(x) * mblock.metric.betar(x));
    }

    static real_t Ar(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      return HALF * (mblock.metric.h_13(x) + TWO * mblock.metric.spin() * mblock.metric.h_11(x) * mblock.metric.betar(x));
    }

    static real_t At(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      (void)(mblock);
      (void)(x);
      return ZERO;
    }

    Inline auto userTargetField_br_hat(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      coord_t<D> rth_;
      rth_[0] = ZERO;
      real_t r_min {mblock.metric.x1_min};
      mblock.metric.x_Code2Sph(x, rth_);
      return ONE * r_min * r_min / (rth_[0] * rth_[0]);
    }

  };

} // namespace ntt

#else
  NTTError("Problem generator relevant in GRPIC only.");
#endif

#endif
