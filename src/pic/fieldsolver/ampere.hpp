#ifndef PIC_FIELDSOLVER_AMPERE_H
#define PIC_FIELDSOLVER_AMPERE_H

namespace ntt {

class Ampere1D_Cartesian : public FieldSolver1D {
public:
  Ampere1D_Cartesian (const Meshblock<One_D>& m_mblock_, const real_t& coeff_) : FieldSolver1D{m_mblock_, coeff_} {}
  Inline void operator() (const size_type i) const {
    m_mblock.ex2(i) = m_mblock.ex2(i) + coeff * (m_mblock.bx3(i - 1) - m_mblock.bx3(i));
    m_mblock.ex3(i) = m_mblock.ex3(i) + coeff * (-m_mblock.bx2(i - 1) + m_mblock.bx2(i));
  }
};

class Ampere2D_Cartesian : public FieldSolver2D {
public:
  Ampere2D_Cartesian (const Meshblock<Two_D>& m_mblock_, const real_t& coeff_) : FieldSolver2D{m_mblock_, coeff_} {}
  Inline void operator() (const size_type i, const size_type j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i, j) + coeff * (-m_mblock.bx3(i, j - 1) + m_mblock.bx3(i, j));
    m_mblock.ex2(i, j) = m_mblock.ex2(i, j) + coeff * (m_mblock.bx3(i - 1, j) - m_mblock.bx3(i, j));
    m_mblock.ex3(i, j) = m_mblock.ex3(i, j) + coeff * (m_mblock.bx1(i, j - 1) - m_mblock.bx1(i, j) - m_mblock.bx2(i - 1, j) + m_mblock.bx2(i, j));
  }
};

class Ampere3D_Cartesian : public FieldSolver3D {
public:
  Ampere3D_Cartesian (const Meshblock<Three_D>& m_mblock_, const real_t& coeff_) : FieldSolver3D{m_mblock_, coeff_} {}
  Inline void operator() (const size_type i, const size_type j, const size_type k) const {
    m_mblock.ex1(i, j, k) = m_mblock.ex1(i, j, k) + coeff * (m_mblock.bx2(i, j, k - 1) - m_mblock.bx2(i, j, k) - m_mblock.bx3(i, j - 1, k) + m_mblock.bx3(i, j, k));
    m_mblock.ex2(i, j, k) = m_mblock.ex2(i, j, k) + coeff * (m_mblock.bx3(i - 1, j, k) - m_mblock.bx3(i, j, k) - m_mblock.bx1(i, j, k - 1) + m_mblock.bx1(i, j, k));
    m_mblock.ex3(i, j, k) = m_mblock.ex3(i, j, k) + coeff * (m_mblock.bx1(i, j - 1, k) - m_mblock.bx1(i, j, k) - m_mblock.bx2(i - 1, j, k) + m_mblock.bx2(i, j, k));
  }
};

}

#endif
