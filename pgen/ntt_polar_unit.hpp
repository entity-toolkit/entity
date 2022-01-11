// #ifndef PROBLEM_GENERATOR_H
// #define PROBLEM_GENERATOR_H

// #include "global.h"
// #include "pgen.h"
// #include "sim_params.h"
// #include "meshblock.h"

// namespace ntt {

//   template <Dimension D>
//   struct ProblemGenerator : PGen<D> {
//     ProblemGenerator(SimulationParams&);
//     ~ProblemGenerator() = default;

//     void userInitFields(SimulationParams&, Meshblock<D>&);
//     void userBCFields(const real_t&, SimulationParams&, Meshblock<D>&);

//     Inline auto userTargetField_br_HAT(Meshblock<D>&, const real_t&) const -> real_t;
//     Inline auto userTargetField_br_HAT(Meshblock<D>&, const real_t&, const real_t&) const -> real_t;
//     Inline auto userTargetField_br_HAT(Meshblock<D>&, const real_t&, const real_t&, const real_t&) const -> real_t;
//   };

//   template <>
//   Inline auto ProblemGenerator<TWO_D>::userTargetField_br_HAT(Meshblock<TWO_D>& mblock,
//                                                               const real_t& x1,
//                                                               const real_t& x2) const -> real_t {
//     auto [r_, th_] = mblock.grid->coord_CU_to_Sph(x1, x2);
//     auto r_min {mblock.grid->x1_min};
//     return ONE * r_min * r_min / (r_ * r_);
//   }

// } // namespace ntt

// #endif

#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    ProblemGenerator(const SimulationParams& sim_params);

    void userInitFields(const SimulationParams&, Meshblock<D, S>&) override;
    void userBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&) override;
  };

} // namespace ntt

#endif
