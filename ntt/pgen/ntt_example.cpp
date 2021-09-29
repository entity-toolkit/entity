#include "global.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"
#include "input.h"

#include <plog/Log.h>

namespace ntt {

// class userInitFields {
//   NTTArray<real_t*> ex1, ex2, ex3;
//   NTTArray<real_t*> bx1, bx2, bx3;
//   using size_type = NTTArray<real_t*>::size_type;
//   real_t coeff;
// public:
//   Faraday1DHalfstep (const NTTArray<real_t*>& ex1_,
//                      const NTTArray<real_t*>& ex2_,
//                      const NTTArray<real_t*>& ex3_,
//                      const NTTArray<real_t*>& bx1_,
//                      const NTTArray<real_t*>& bx2_,
//                      const NTTArray<real_t*>& bx3_,
//                      const real_t& coeff_) :
//                      ex1(ex1_), ex2(ex2_), ex3(ex3_),
//                      bx1(bx1_), bx2(bx2_), bx3(bx3_),
//                      coeff(coeff_) {}
//   Inline void operator() (const size_type i) const {
//     ex1(i) = 0.0;
//   }
// };

}
