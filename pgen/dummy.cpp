#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

#include "problem_generator.hpp"

#ifdef PIC_SIMTYPE
template struct ntt::ProblemGenerator<ntt::Dim1, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::SimulationType::PIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::SimulationType::PIC>;
#elif defined(GRPIC_SIMTYPE)
template struct ntt::ProblemGenerator<ntt::Dim2, ntt::SimulationType::GRPIC>;
template struct ntt::ProblemGenerator<ntt::Dim3, ntt::SimulationType::GRPIC>;
#endif
