#ifndef PIC_FIELDSOLVER_ADDRESETCURRENTS_H
#define PIC_FIELDSOLVER_ADDRESETCURRENTS_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

// * * * * Add currents * * * * * * * * * * * * * * * *
template <Dimension D>
class AddCurrents : public FieldSolver<D> {
  using index_t = typename RealFieldND<D, 3>::size_type;

public:
  AddCurrents(const Meshblock<D>& m_mblock_) : FieldSolver<D> {m_mblock_} {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

template <>
Inline void AddCurrents<ONE_D>::operator()(const index_t i) const {
  m_mblock.em_fields(i, fld::ex1) += m_mblock.j_fields(i, fld::jx1);
  m_mblock.em_fields(i, fld::ex2) += m_mblock.j_fields(i, fld::jx2);
  m_mblock.em_fields(i, fld::ex3) += m_mblock.j_fields(i, fld::jx3);
}

template <>
Inline void AddCurrents<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.em_fields(i, j, fld::ex1) += m_mblock.j_fields(i, j, fld::jx1);
  m_mblock.em_fields(i, j, fld::ex2) += m_mblock.j_fields(i, j, fld::jx2);
  m_mblock.em_fields(i, j, fld::ex3) += m_mblock.j_fields(i, j, fld::jx3);
}

template <>
Inline void
AddCurrents<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  m_mblock.em_fields(i, j, k, fld::ex1) += m_mblock.j_fields(i, j, k, fld::jx1);
  m_mblock.em_fields(i, j, k, fld::ex2) += m_mblock.j_fields(i, j, k, fld::jx2);
  m_mblock.em_fields(i, j, k, fld::ex3) += m_mblock.j_fields(i, j, k, fld::jx3);
}

// * * * * Reset currents * * * * * * * * * * * * * * *
template <Dimension D>
class ResetCurrents : public FieldSolver<D> {
  using index_t = typename RealFieldND<D, 3>::size_type;

public:
  ResetCurrents(const Meshblock<D>& m_mblock_) : FieldSolver<D> {m_mblock_} {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

template <>
Inline void ResetCurrents<ONE_D>::operator()(const index_t i) const {
  m_mblock.j_fields(i, fld::jx1) = 0.0;
  m_mblock.j_fields(i, fld::jx2) = 0.0;
  m_mblock.j_fields(i, fld::jx3) = 0.0;
}

template <>
Inline void ResetCurrents<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.j_fields(i, j, fld::jx1) = 0.0;
  m_mblock.j_fields(i, j, fld::jx2) = 0.0;
  m_mblock.j_fields(i, j, fld::jx3) = 0.0;
}

template <>
Inline void
ResetCurrents<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  m_mblock.j_fields(i, j, k, fld::jx1) = 0.0;
  m_mblock.j_fields(i, j, k, fld::jx2) = 0.0;
  m_mblock.j_fields(i, j, k, fld::jx3) = 0.0;
}

} // namespace ntt

template class ntt::AddCurrents<ntt::ONE_D>;
template class ntt::AddCurrents<ntt::TWO_D>;
template class ntt::AddCurrents<ntt::THREE_D>;

template class ntt::ResetCurrents<ntt::ONE_D>;
template class ntt::ResetCurrents<ntt::TWO_D>;
template class ntt::ResetCurrents<ntt::THREE_D>;

#endif
