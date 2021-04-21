#ifndef GLOBAL_H
#define GLOBAL_H

#include <string>

namespace ntt {
  #ifdef SINGLE_PRECISION
    typedef float real_t;
  #else
    typedef double real_t;
  #endif

  class Simulation {
  private:
    std::string _title;
  public:
    const std::string& title = _title;
    void setTitle(const std::string _t) { _title = _t; }
    const std::size_t precision = sizeof(real_t);
  } sim;
}

#endif
