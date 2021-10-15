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
  real_t coeff;

public:
  Faraday(const Meshblock<D>& m_mblock_, const real_t& coeff_)
      : FieldSolver<D> {m_mblock_}, coeff(coeff_) {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

template <>
Inline void Faraday<ONE_D>::operator()(const index_t i) const {
  // clang-format off
  m_mblock.bx2(i) += coeff * ( m_mblock.ex3(i + 1) - m_mblock.ex3(i));
  m_mblock.bx3(i) += coeff * (-m_mblock.ex2(i + 1) + m_mblock.ex2(i));
  // clang-format on
}

template <>
Inline void Faraday<TWO_D>::operator()(const index_t i, const index_t j) const {
  // clang-format off
  m_mblock.bx1(i, j) += coeff * (-m_mblock.ex3(    i, j + 1) + m_mblock.ex3(    i,    j));
  m_mblock.bx2(i, j) += coeff * ( m_mblock.ex3(i + 1,     j) - m_mblock.ex3(    i,    j));
  m_mblock.bx3(i, j) += coeff * ( m_mblock.ex1(    i, j + 1) - m_mblock.ex1(    i,    j)
                                - m_mblock.ex2(i + 1,     j) + m_mblock.ex2(    i,    j));
  // clang-format on
}

template <>
Inline void Faraday<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  // clang-format off
  m_mblock.bx1(i, j, k) += coeff * (m_mblock.ex2(    i,     j, k + 1) - m_mblock.ex2(    i,     j,     k)
                                  - m_mblock.ex3(    i, j + 1,     k) + m_mblock.ex3(    i,     j,     k));
  m_mblock.bx2(i, j, k) += coeff * (m_mblock.ex3(i + 1,     j,     k) - m_mblock.ex3(    i,     j,     k)
                                  - m_mblock.ex1(    i,     j, k + 1) + m_mblock.ex1(    i,     j,     k));
  m_mblock.bx3(i, j, k) += coeff * (m_mblock.ex1(    i, j + 1,     k) - m_mblock.ex1(    i,     j,     k)
                                  - m_mblock.ex2(i + 1,     j,     k) + m_mblock.ex2(    i,     j,     k));
  // clang-format on
}

} // namespace ntt

template class ntt::Faraday<ntt::ONE_D>;
template class ntt::Faraday<ntt::TWO_D>;
template class ntt::Faraday<ntt::THREE_D>;

#endif
