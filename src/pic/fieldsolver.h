#ifndef PIC_FIELDSOLVER_H
#define PIC_FIELDSOLVER_H

#include "global.h"
#include "meshblock.h"

namespace ntt {

class FieldSolver1D {
protected:
  Meshblock<One_D> m_mblock;
  real_t coeff;
public:
  FieldSolver1D(const Meshblock<One_D>& m_mblock_, const real_t& coeff_) : m_mblock(m_mblock_), coeff(coeff_) {}
};

class FieldSolver2D {
protected:
  Meshblock<Two_D> m_mblock;
  real_t coeff;
public:
  FieldSolver2D(const Meshblock<Two_D>& m_mblock_, const real_t& coeff_) : m_mblock(m_mblock_), coeff(coeff_) {}
};

class FieldSolver3D {
protected:
  Meshblock<Three_D> m_mblock;
  real_t coeff;
public:
  FieldSolver3D(const Meshblock<Three_D>& m_mblock_, const real_t& coeff_) : m_mblock(m_mblock_), coeff(coeff_) {}
};
}

#endif
