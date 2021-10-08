#ifndef PIC_FIELDSOLVER_ADDRESETCURRENTS_H
#define PIC_FIELDSOLVER_ADDRESETCURRENTS_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

class AddCurrents1D : public FieldSolver<ONE_D> {
public:
  AddCurrents1D(const Meshblock1D& m_mblock_) : FieldSolver<ONE_D>{m_mblock_, 0.0} {}
  Inline void operator()(const size_type i) const {
    m_mblock.ex1(i) = m_mblock.ex1(i) + m_mblock.jx1(i);
    m_mblock.ex2(i) = m_mblock.ex2(i) + m_mblock.jx2(i);
    m_mblock.ex3(i) = m_mblock.ex3(i) + m_mblock.jx3(i);
  }
};

class AddCurrents2D : public FieldSolver<TWO_D> {
public:
  AddCurrents2D(const Meshblock2D& m_mblock_) : FieldSolver<TWO_D>{m_mblock_, 0.0} {}
  Inline void operator()(const size_type i, const size_type j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i, j) + m_mblock.jx1(i, j);
    m_mblock.ex2(i, j) = m_mblock.ex2(i, j) + m_mblock.jx2(i, j);
    m_mblock.ex3(i, j) = m_mblock.ex3(i, j) + m_mblock.jx3(i, j);
  }
};

class AddCurrents3D : public FieldSolver<THREE_D> {
public:
  AddCurrents3D(const Meshblock3D& m_mblock_) : FieldSolver<THREE_D>{m_mblock_, 0.0} {}
  Inline void operator()(const size_type i, const size_type j, const size_type k) const {
    m_mblock.ex1(i, j, k) = m_mblock.ex1(i, j, k) + m_mblock.jx1(i, j, k);
    m_mblock.ex2(i, j, k) = m_mblock.ex2(i, j, k) + m_mblock.jx2(i, j, k);
    m_mblock.ex3(i, j, k) = m_mblock.ex3(i, j, k) + m_mblock.jx3(i, j, k);
  }
};

class ResetCurrents1D : public FieldSolver<ONE_D> {
public:
  ResetCurrents1D(const Meshblock1D& m_mblock_) : FieldSolver<ONE_D>{m_mblock_, 0.0} {}
  Inline void operator()(const size_type i) const {
    m_mblock.jx1(i) = 0.0;
    m_mblock.jx2(i) = 0.0;
    m_mblock.jx3(i) = 0.0;
  }
};

class ResetCurrents2D : public FieldSolver<TWO_D> {
public:
  ResetCurrents2D(const Meshblock2D& m_mblock_) : FieldSolver<TWO_D>{m_mblock_, 0.0} {}
  Inline void operator()(const size_type i, const size_type j) const {
    m_mblock.jx1(i, j) = 0.0;
    m_mblock.jx2(i, j) = 0.0;
    m_mblock.jx3(i, j) = 0.0;
  }
};

class ResetCurrents3D : public FieldSolver<THREE_D> {
public:
  ResetCurrents3D(const Meshblock3D& m_mblock_) : FieldSolver<THREE_D>{m_mblock_, 0.0} {}
  Inline void operator()(const size_type i, const size_type j, const size_type k) const {
    m_mblock.jx1(i, j, k) = 0.0;
    m_mblock.jx2(i, j, k) = 0.0;
    m_mblock.jx3(i, j, k) = 0.0;
  }
};

} // namespace ntt

#endif
