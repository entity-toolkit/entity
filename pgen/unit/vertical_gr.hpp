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

    void userInitFields(const SimulationParams&, Meshblock<D, S>&);
    void userInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    // void userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&);

    static real_t A0(const Meshblock<D, S>& mblock, const coord_t<D>& x) {
      (void)mblock;
      (void)x;
      return ZERO;
    }

    static real_t A1(const Meshblock<D, S>& mblock, const coord_t<D>& x) {
      (void)mblock;
      (void)x;
      return ZERO;
    }

    static real_t A3(const Meshblock<D, S>& mblock, const coord_t<D>& x) {
      coord_t<D> rth_;
      mblock.metric.x_Code2Sph(x, rth_);
      return HALF * std::sin(rth_[1]) * std::sin(rth_[1]) * rth_[0] * rth_[0];
    }

    Inline auto userTargetField_br_cntrv(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      coord_t<D> x0m, x0p;
      real_t inv_sqrt_detH_ijP {ONE / mblock.metric.sqrt_det_h(x)};
      if constexpr (D == Dimension::TWO_D) {
        x0m[0] = x[0], x0m[1] = x[1] - HALF * epsilon;
        x0p[0] = x[0], x0p[1] = x[1] + HALF * epsilon;
      } else {
        NTTError("Only 2D is supported.");
      }
      return (A3(mblock, x0p) - A3(mblock, x0m)) * inv_sqrt_detH_ijP / epsilon;
    }

    Inline auto userTargetField_bth_cntrv(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      coord_t<D> x0m, x0p;
      real_t inv_sqrt_detH_iPj {ONE / mblock.metric.sqrt_det_h(x)};
      if constexpr (D == Dimension::TWO_D) {
        x0m[0] = x[0] + HALF - HALF * epsilon, x0m[1] = x[1];
        x0p[0] = x[0] + HALF + HALF * epsilon, x0p[1] = x[1];
      } else {
        NTTError("Only 2D is supported.");
      }
      if (x[1] == ZERO) {
        return ZERO;
      } else {
        return -(A3(mblock, x0p) - A3(mblock, x0m)) * inv_sqrt_detH_iPj / epsilon;
      }
    }
  };

} // namespace ntt

#else
NTTError("Problem generator relevant in GRPIC only.");
#endif

#endif
