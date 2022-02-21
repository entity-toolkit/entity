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
      return HALF * rth_[0] * rth_[0] * std::sin(rth_[1]) * std::sin(rth_[1]);
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
