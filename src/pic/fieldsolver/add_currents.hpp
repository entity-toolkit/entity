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
    AddCurrents(const Meshblock<D>& mblock_) : FieldSolver<D> {mblock_} {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void AddCurrents<ONE_D>::operator()(const index_t i) const {
    mblock.em_fields(i, fld::ex1) += mblock.j_fields(i, fld::jx1);
    mblock.em_fields(i, fld::ex2) += mblock.j_fields(i, fld::jx2);
    mblock.em_fields(i, fld::ex3) += mblock.j_fields(i, fld::jx3);
  }

  template <>
  Inline void AddCurrents<TWO_D>::operator()(const index_t i, const index_t j) const {
    mblock.em_fields(i, j, fld::ex1) += mblock.j_fields(i, j, fld::jx1);
    mblock.em_fields(i, j, fld::ex2) += mblock.j_fields(i, j, fld::jx2);
    mblock.em_fields(i, j, fld::ex3) += mblock.j_fields(i, j, fld::jx3);
  }

  template <>
  Inline void AddCurrents<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
    mblock.em_fields(i, j, k, fld::ex1) += mblock.j_fields(i, j, k, fld::jx1);
    mblock.em_fields(i, j, k, fld::ex2) += mblock.j_fields(i, j, k, fld::jx2);
    mblock.em_fields(i, j, k, fld::ex3) += mblock.j_fields(i, j, k, fld::jx3);
  }

} // namespace ntt

template class ntt::AddCurrents<ntt::ONE_D>;
template class ntt::AddCurrents<ntt::TWO_D>;
template class ntt::AddCurrents<ntt::THREE_D>;

#endif
