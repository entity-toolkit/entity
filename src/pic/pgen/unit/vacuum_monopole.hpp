#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator {
    ProblemGenerator(const SimulationParams&);

    void        UserInitFields(const SimulationParams&, Meshblock<D, S>&);
    void        UserInitParticles(const SimulationParams&, Meshblock<D, S>&);
    void        UserBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&);
    Inline auto UserTargetField_br_hat(const Meshblock<D, S>& mblock, const coord_t<D>& x) const
      -> real_t {
      coord_t<D> rth_;
      rth_[0] = ZERO;
      real_t r_min {mblock.metric.x1_min};
      mblock.metric.x_Code2Sph(x, rth_);
      return ONE * SQR(r_min / rth_[0]);
    }
    void UserDriveParticles(const real_t&, const SimulationParams&, Meshblock<D, S>&);

  private:
    real_t spinup_time;
    real_t omega_max;
  };

} // namespace ntt

#endif