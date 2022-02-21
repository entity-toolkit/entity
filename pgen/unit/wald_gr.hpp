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
    // RealFieldND<D, 1> Bru0;

    void userInitFields(const SimulationParams&, Meshblock<D, S>&);
    void userInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    void userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&);

    static real_t A0(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      real_t g00 {- mblock.metric.alpha(x) * mblock.metric.alpha(x) + mblock.metric.h_11(x) * mblock.metric.beta1u(x) * mblock.metric.beta1u(x)};
      return HALF * (mblock.metric.h_13(x) * mblock.metric.beta1u(x) + TWO * mblock.metric.spin() * g00);
    }

    static real_t A1(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      return HALF * (mblock.metric.h_13(x) + TWO * mblock.metric.spin() * mblock.metric.h_11(x) * mblock.metric.beta1u(x));
    }

    static real_t A3(const Meshblock<D, S>& mblock, const coord_t<D>& x)
    {
      return HALF * (mblock.metric.h_33(x) + TWO * mblock.metric.spin() * mblock.metric.h_13(x) * mblock.metric.beta1u(x));
    }

    Inline auto userTargetField_br_hat(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      coord_t<D> rth_;
      mblock.metric.x_Code2Sph(x, rth_);
      return ONE * std::cos(rth_[1]); // Vertical field at infinity
    }

    Inline auto userTargetField_bth_hat(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      coord_t<D> rth_;
      mblock.metric.x_Code2Sph(x, rth_);
      return - ONE * std::sin(rth_[1]); // Vertical field at infinity
    }

  };

} // namespace ntt

#else
  NTTError("Problem generator relevant in GRPIC only.");
#endif

#endif
