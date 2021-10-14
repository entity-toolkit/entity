#ifndef PIC_FIELDSOLVER_AMPERE_H
#define PIC_FIELDSOLVER_AMPERE_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

// * * * * Ampere's law * * * * * * * * * * * * * * * *
template <Dimension D>
class Ampere : public FieldSolver<D> {
  using index_t = typename RealArrND<D>::size_type;
  real_t coeff;

public:
  Ampere(const Meshblock<D>& m_mblock_, const real_t& coeff_)
      : FieldSolver<D> {m_mblock_}, coeff(coeff_) {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

template <>
Inline void Ampere<ONE_D>::operator()(const index_t i) const {
  m_mblock.ex2(i) = m_mblock.ex2(i) + coeff * (m_mblock.bx3(i - 1) - m_mblock.bx3(i));
  m_mblock.ex3(i) = m_mblock.ex3(i) + coeff * (-m_mblock.bx2(i - 1) + m_mblock.bx2(i));
}

template <>
Inline void Ampere<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.ex1(i, j) = m_mblock.ex1(i, j) + coeff * (-m_mblock.bx3(i, j - 1) + m_mblock.bx3(i, j));
  m_mblock.ex2(i, j) = m_mblock.ex2(i, j) + coeff * (m_mblock.bx3(i - 1, j) - m_mblock.bx3(i, j));
  m_mblock.ex3(i, j) = m_mblock.ex3(i, j)
                       + coeff
                             * (m_mblock.bx1(i, j - 1) - m_mblock.bx1(i, j) - m_mblock.bx2(i - 1, j)
                                + m_mblock.bx2(i, j));
}

template <>
Inline void Ampere<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
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

} // namespace ntt

template class ntt::Ampere<ntt::ONE_D>;
template class ntt::Ampere<ntt::TWO_D>;
template class ntt::Ampere<ntt::THREE_D>;

#endif
