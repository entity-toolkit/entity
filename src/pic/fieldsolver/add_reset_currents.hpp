#ifndef PIC_FIELDSOLVER_ADDRESETCURRENTS_H
#define PIC_FIELDSOLVER_ADDRESETCURRENTS_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

// * * * * Add currents * * * * * * * * * * * * * * * *
template <Dimension D>
class AddCurrents : public FieldSolver<D> {
  using index_t = typename RealArrND<D>::size_type;

public:
  AddCurrents(const Meshblock<D>& m_mblock_) : FieldSolver<D> {m_mblock_} {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

template <>
Inline void AddCurrents<ONE_D>::operator()(const index_t i) const {
  m_mblock.ex1(i) = m_mblock.ex1(i) + m_mblock.jx1(i);
  m_mblock.ex2(i) = m_mblock.ex2(i) + m_mblock.jx2(i);
  m_mblock.ex3(i) = m_mblock.ex3(i) + m_mblock.jx3(i);
}

template <>
Inline void AddCurrents<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.ex1(i, j) = m_mblock.ex1(i, j) + m_mblock.jx1(i, j);
  m_mblock.ex2(i, j) = m_mblock.ex2(i, j) + m_mblock.jx2(i, j);
  m_mblock.ex3(i, j) = m_mblock.ex3(i, j) + m_mblock.jx3(i, j);
}

template <>
Inline void
AddCurrents<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  m_mblock.ex1(i, j, k) = m_mblock.ex1(i, j, k) + m_mblock.jx1(i, j, k);
  m_mblock.ex2(i, j, k) = m_mblock.ex2(i, j, k) + m_mblock.jx2(i, j, k);
  m_mblock.ex3(i, j, k) = m_mblock.ex3(i, j, k) + m_mblock.jx3(i, j, k);
}

// * * * * Reset currents * * * * * * * * * * * * * * *
template <Dimension D>
class ResetCurrents : public FieldSolver<D> {
  using index_t = typename RealArrND<D>::size_type;

public:
  ResetCurrents(const Meshblock<D>& m_mblock_) : FieldSolver<D> {m_mblock_} {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

template <>
Inline void ResetCurrents<ONE_D>::operator()(const index_t i) const {
  m_mblock.jx1(i) = 0.0;
  m_mblock.jx2(i) = 0.0;
  m_mblock.jx3(i) = 0.0;
}

template <>
Inline void ResetCurrents<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.jx1(i, j) = 0.0;
  m_mblock.jx2(i, j) = 0.0;
  m_mblock.jx3(i, j) = 0.0;
}

template <>
Inline void
ResetCurrents<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  m_mblock.jx1(i, j, k) = 0.0;
  m_mblock.jx2(i, j, k) = 0.0;
  m_mblock.jx3(i, j, k) = 0.0;
}

} // namespace ntt

template class ntt::AddCurrents<ntt::ONE_D>;
template class ntt::AddCurrents<ntt::TWO_D>;
template class ntt::AddCurrents<ntt::THREE_D>;

template class ntt::ResetCurrents<ntt::ONE_D>;
template class ntt::ResetCurrents<ntt::TWO_D>;
template class ntt::ResetCurrents<ntt::THREE_D>;

#endif
