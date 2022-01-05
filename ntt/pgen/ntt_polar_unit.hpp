#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D>
  struct ProblemGenerator : PGen<D> {
    ProblemGenerator(SimulationParams&);
    ~ProblemGenerator() = default;

    void userInitFields(SimulationParams&, Meshblock<D>&);
    void userBCFields(const real_t&, SimulationParams&, Meshblock<D>&);

    Inline auto userTargetField_bx1(Meshblock<D>& mblock,
                                    const real_t& x1, const real_t& x2) const -> real_t {
      auto [r_, th_] = mblock.m_coord_system->coord_CU_to_Sph(x1, x2 + HALF);
      auto r_min {mblock.m_coord_system->x1_min};
      auto br_hat {ONE * r_min * r_min / (r_ * r_)};
      return mblock.m_coord_system->vec_HAT_to_CNT_x1(br_hat, x1, x2 + HALF);
    }

  };

} // namespace ntt

#endif
