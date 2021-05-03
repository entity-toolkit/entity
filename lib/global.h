#ifndef GLOBAL_H
#define GLOBAL_H

namespace ntt {
  #ifdef SINGLE_PRECISION
    typedef float real_t;
  #else
    typedef double real_t;
  #endif
}

#endif
