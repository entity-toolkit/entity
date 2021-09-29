#ifndef PIC_FIELDSOLVER_FARADAY_H
#define PIC_FIELDSOLVER_FARADAY_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

class Faraday1DHalfstep_Cartesian : public FieldSolver1D {
  using size_type = NTTArray<real_t*>::size_type;
public:
  Faraday1DHalfstep_Cartesian (const Meshblock<One_D>& m_mblock_, const real_t& coeff_) : FieldSolver1D{m_mblock_, coeff_} {}
  Inline void operator() (const size_type i) const {
    // m_mblock.ex1(i) = coeff;
    // ...
  }
};

class Faraday2DHalfstep_Cartesian : public FieldSolver2D {
  using size_type = NTTArray<real_t**>::size_type;
public:
  Faraday2DHalfstep_Cartesian (const Meshblock<Two_D>& m_mblock_, const real_t& coeff_) : FieldSolver2D{m_mblock_, coeff_} {}
  Inline void operator() (const size_type i, const size_type j) const {
    // m_mblock.ex1(i, j) = coeff;
    // ...
  }
};

class Faraday3DHalfstep_Cartesian : public FieldSolver3D {
  using size_type = NTTArray<real_t***>::size_type;
public:
  Faraday3DHalfstep_Cartesian (const Meshblock<Three_D>& m_mblock_, const real_t& coeff_) : FieldSolver3D{m_mblock_, coeff_} {}
  Inline void operator() (const size_type i, const size_type j, const size_type k) const {
    // m_mblock.ex1(i, j, k) = coeff;
    // ...
  }
};

}

#endif
