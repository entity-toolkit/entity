#ifndef PIC_FIELDSOLVER_FARADAY_H
#define PIC_FIELDSOLVER_FARADAY_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

// * * * * Faraday's law * * * * * * * * * * * * * * *
template <Dimension D>
class Faraday : public FieldSolver<D> {
  using index_t = typename RealArrND<D>::size_type;
  real_t coeff_x1, coeff_x2, coeff_x3;

public:
  Faraday(
      const Meshblock<D>& m_mblock_,
      const real_t& coeff_x1_,
      const real_t& coeff_x2_,
      const real_t& coeff_x3_)
      : FieldSolver<D> {m_mblock_}, coeff_x1(coeff_x1_), coeff_x2(coeff_x2_), coeff_x3(coeff_x3_) {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

template <>
Inline void Faraday<ONE_D>::operator()(const index_t i) const {
  m_mblock.bx2(i) = m_mblock.bx2(i) + coeff_x1 * (m_mblock.ex3(i + 1) - m_mblock.ex3(i));
  m_mblock.bx3(i) = m_mblock.bx3(i) + coeff_x1 * (-m_mblock.ex2(i + 1) + m_mblock.ex2(i));
}

template <>
Inline void Faraday<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.bx1(i, j)
      = m_mblock.bx1(i, j) + coeff_x2 * (-m_mblock.ex3(i, j + 1) + m_mblock.ex3(i, j));
  m_mblock.bx2(i, j)
      = m_mblock.bx2(i, j) + coeff_x1 * (m_mblock.ex3(i + 1, j) - m_mblock.ex3(i, j));
  m_mblock.bx3(i, j) = m_mblock.bx3(i, j) + coeff_x2 * (m_mblock.ex1(i, j + 1) - m_mblock.ex1(i, j))
                       + coeff_x1 * (-m_mblock.ex2(i + 1, j) + m_mblock.ex2(i, j));
}

template <>
Inline void Faraday<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  m_mblock.bx1(i, j, k) = m_mblock.bx1(i, j, k)
                          + coeff_x3 * (m_mblock.ex2(i, j, k + 1) - m_mblock.ex2(i, j, k))
                          + coeff_x2 * (-m_mblock.ex3(i, j + 1, k) + m_mblock.ex3(i, j, k));
  m_mblock.bx2(i, j, k) = m_mblock.bx2(i, j, k)
                          + coeff_x1 * (m_mblock.ex3(i + 1, j, k) - m_mblock.ex3(i, j, k))
                          + coeff_x3 * (-m_mblock.ex1(i, j, k + 1) + m_mblock.ex1(i, j, k));
  m_mblock.bx3(i, j, k) = m_mblock.bx3(i, j, k)
                          + coeff_x2 * (m_mblock.ex1(i, j + 1, k) - m_mblock.ex1(i, j, k))
                          + coeff_x1 * (-m_mblock.ex2(i + 1, j, k) + m_mblock.ex2(i, j, k));
}

} // namespace ntt

template class ntt::Faraday<ntt::ONE_D>;
template class ntt::Faraday<ntt::TWO_D>;
template class ntt::Faraday<ntt::THREE_D>;

#endif
