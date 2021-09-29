#ifndef PIC_FIELDSOLVER_FARADAY_H
#define PIC_FIELDSOLVER_FARADAY_H

#include "global.h"
#include "meshblock.h"
#include "simulation.h"

#include <Kokkos_Core.hpp>

namespace ntt {

class Faraday1DHalfstep_Cartesian {
  Meshblock<One_D> m_mblock;
  using size_type = NTTArray<real_t*>::size_type;
  real_t coeff;
public:
  Faraday1DHalfstep_Cartesian (const Meshblock<One_D>& m_mblock_,
                               const real_t& coeff_) :
                               m_mblock(m_mblock_), coeff(coeff_) {}
  Inline void operator() (const size_type i) const {
    // m_mblock.ex1(i) = coeff;
    // ...
  }
};

class Faraday2DHalfstep_Cartesian {
  Meshblock<Two_D> m_mblock;
  using size_type = NTTArray<real_t**>::size_type;
  real_t coeff;
public:
  Faraday2DHalfstep_Cartesian (const Meshblock<Two_D>& m_mblock_,
                               const real_t& coeff_) :
                               m_mblock(m_mblock_), coeff(coeff_) {}
  Inline void operator() (const size_type i, const size_type j) const {
    // m_mblock.ex1(i, j) = coeff;
    // ...
  }
};

class Faraday3DHalfstep_Cartesian {
  Meshblock<Three_D> m_mblock;
  using size_type = NTTArray<real_t***>::size_type;
  real_t coeff;
public:
  Faraday3DHalfstep_Cartesian (const Meshblock<Three_D>& m_mblock_,
                               const real_t& coeff_) :
                               m_mblock(m_mblock_), coeff(coeff_) {}
  Inline void operator() (const size_type i, const size_type j, const size_type k) const {
    // m_mblock.ex1(i, j, k) = coeff;
    // ...
  }
};

}

#endif
