#ifndef PIC_FIELDSOLVER_H
#define PIC_FIELDSOLVER_H

#include "global.h"
#include "meshblock.h"

namespace ntt {

class FieldSolver1D {
protected:
  Meshblock1D m_mblock;
  real_t coeff;
  using size_type = NTTArray<real_t*>::size_type;

public:
  FieldSolver1D(const Meshblock1D& m_mblock_, const real_t& coeff_)
      : m_mblock(m_mblock_), coeff(coeff_) {}
};

class FieldSolver2D {
protected:
  Meshblock2D m_mblock;
  real_t coeff;
  using size_type = NTTArray<real_t**>::size_type;

public:
  FieldSolver2D(const Meshblock2D& m_mblock_, const real_t& coeff_)
      : m_mblock(m_mblock_), coeff(coeff_) {}
};

class FieldSolver3D {
protected:
  Meshblock3D m_mblock;
  real_t coeff;
  using size_type = NTTArray<real_t***>::size_type;

public:
  FieldSolver3D(const Meshblock3D& m_mblock_, const real_t& coeff_)
      : m_mblock(m_mblock_), coeff(coeff_) {}
};
} // namespace ntt

#endif
