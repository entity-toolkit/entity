#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "global.h"
#include "pgen.h"

namespace ntt {

  template <Dimension D>
  struct ProblemGenerator : PGen<D> {};

} // namespace ntt

#endif
