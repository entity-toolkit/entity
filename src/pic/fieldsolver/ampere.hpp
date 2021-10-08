#ifndef PIC_FIELDSOLVER_AMPERE_H
#define PIC_FIELDSOLVER_AMPERE_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

class Ampere1D_Cartesian : public FieldSolver<ONE_D> {
public:
  Ampere1D_Cartesian(const Meshblock<ONE_D>& m_mblock_, const real_t& coeff_)
      : FieldSolver<ONE_D>{m_mblock_, coeff_} {}
  Inline void operator()(const index_t i) const {
    m_mblock.ex2(i) = m_mblock.ex2(i) + coeff * (m_mblock.bx3(i - 1) - m_mblock.bx3(i));
    m_mblock.ex3(i) = m_mblock.ex3(i) + coeff * (-m_mblock.bx2(i - 1) + m_mblock.bx2(i));
  }
};

class Ampere2D_Cartesian : public FieldSolver<TWO_D> {
public:
  Ampere2D_Cartesian(const Meshblock<TWO_D>& m_mblock_, const real_t& coeff_)
      : FieldSolver<TWO_D>{m_mblock_, coeff_} {}
  Inline void operator()(const index_t i, const index_t j) const {
    m_mblock.ex1(i, j)
        = m_mblock.ex1(i, j) + coeff * (-m_mblock.bx3(i, j - 1) + m_mblock.bx3(i, j));
    m_mblock.ex2(i, j) = m_mblock.ex2(i, j) + coeff * (m_mblock.bx3(i - 1, j) - m_mblock.bx3(i, j));
    m_mblock.ex3(i, j) = m_mblock.ex3(i, j)
                       + coeff
                             * (m_mblock.bx1(i, j - 1) - m_mblock.bx1(i, j) - m_mblock.bx2(i - 1, j)
                                + m_mblock.bx2(i, j));
  }
};

class Ampere3D_Cartesian : public FieldSolver<THREE_D> {
public:
  Ampere3D_Cartesian(const Meshblock<THREE_D>& m_mblock_, const real_t& coeff_)
      : FieldSolver<THREE_D>{m_mblock_, coeff_} {}
  Inline void operator()(const index_t i, const index_t j, const index_t k) const {
    m_mblock.ex1(i, j, k) = m_mblock.ex1(i, j, k)
                          + coeff
                                * (m_mblock.bx2(i, j, k - 1) - m_mblock.bx2(i, j, k)
                                   - m_mblock.bx3(i, j - 1, k) + m_mblock.bx3(i, j, k));
    m_mblock.ex2(i, j, k) = m_mblock.ex2(i, j, k)
                          + coeff
                                * (m_mblock.bx3(i - 1, j, k) - m_mblock.bx3(i, j, k)
                                   - m_mblock.bx1(i, j, k - 1) + m_mblock.bx1(i, j, k));
    m_mblock.ex3(i, j, k) = m_mblock.ex3(i, j, k)
                          + coeff
                                * (m_mblock.bx1(i, j - 1, k) - m_mblock.bx1(i, j, k)
                                   - m_mblock.bx2(i - 1, j, k) + m_mblock.bx2(i, j, k));
  }
};

} // namespace ntt

#endif
