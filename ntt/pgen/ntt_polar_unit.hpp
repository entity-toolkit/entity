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
      real_t dx1 {(x1 - mblock.m_extent[0]) / ((mblock.m_extent[1] - mblock.m_extent[0]) / static_cast<real_t>(mblock.m_resolution[0]))};
      real_t dx2 {(x2 - mblock.m_extent[2]) / ((mblock.m_extent[3] - mblock.m_extent[2]) / static_cast<real_t>(mblock.m_resolution[1]))};

      auto r0 {mblock.m_coord_system->getSpherical_r(mblock.convert_iTOx1(N_GHOSTS), ZERO)};
      auto rr {mblock.m_coord_system->getSpherical_r(x1, ZERO)};

      auto bx1 {ONE * r0 * r0 / (rr * rr)};
      return mblock.m_coord_system->convert_LOC_to_CNT_x1(bx1, x1, x2 + 0.5 * dx1);
    }

  };

} // namespace ntt

#endif
