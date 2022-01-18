#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator {
    ProblemGenerator(const SimulationParams& sim_params);

    void userInitFields(const SimulationParams&, Meshblock<D, S>&);
    void userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&);

    Inline auto userTargetField_br_hat(const Meshblock<D, S>& mblock, const coord_t<D>& x) const -> real_t {
      coord_t<D> rth_;
      real_t r_min {mblock.metric->x1_min};
      mblock.metric->x_Code2Sph(x, rth_);
      return ONE * r_min * r_min / (rth_[0] * rth_[0]);
    }
  };

} // namespace ntt

#endif
