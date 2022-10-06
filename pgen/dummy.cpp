#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

#ifdef PIC_SIMTYPE
template struct ntt::ProblemGenerator<ntt::Dim1, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::TypePIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::TypePIC>;
#elif defined(GRPIC_SIMTYPE)
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::SimulationType::GRPIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::SimulationType::GRPIC>;
#endif
