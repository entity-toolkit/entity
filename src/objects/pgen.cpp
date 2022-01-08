#include "global.h"
#include "pgen.h"
#include "meshblock.h"

template struct ntt::PGen<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template struct ntt::PGen<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template struct ntt::PGen<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
