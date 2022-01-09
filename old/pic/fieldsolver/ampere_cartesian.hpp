#ifndef PIC_FIELDSOLVER_AMPERE_CARTESIAN_H
#define PIC_FIELDSOLVER_AMPERE_CARTESIAN_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

  // * * * * Cartesian Ampere's law * * * * * * * * * * * * * * * *
  template <Dimension D>
  class AmpereCartesian : public FieldSolver<D> {
    using index_t = typename RealFieldND<D, 1>::size_type;
    real_t coeff;

  public:
    AmpereCartesian(
        const Meshblock<D>& mblock_,
        const real_t& coeff_)
        : FieldSolver<D> {mblock_},
          coeff(coeff_)
        {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void AmpereCartesian<ONE_D>::operator()(const index_t i) const {
    mblock.em_fields(i, fld::ex2) += coeff * (
                                          mblock.em_fields(i - 1, fld::bx3) - mblock.em_fields(i, fld::bx3)
                                        );
    mblock.em_fields(i, fld::ex3) += coeff * (
                                          mblock.em_fields(i, fld::bx2) - mblock.em_fields(i - 1, fld::bx2)
                                        );
  }

  template <>
  Inline void AmpereCartesian<TWO_D>::operator()(const index_t i, const index_t j) const {
    mblock.em_fields(i, j, fld::ex1) += coeff * (
                                            mblock.em_fields(i, j, fld::bx3) - mblock.em_fields(i, j - 1, fld::bx3)
                                          );
    mblock.em_fields(i, j, fld::ex2) += coeff * (
                                            mblock.em_fields(i - 1, j, fld::bx3) - mblock.em_fields(i, j, fld::bx3)
                                          );
    mblock.em_fields(i, j, fld::ex3) += coeff * (
                                            mblock.em_fields(i, j - 1, fld::bx1) - mblock.em_fields(i, j, fld::bx1) +
                                            mblock.em_fields(i, j, fld::bx2) - mblock.em_fields(i - 1, j, fld::bx2)
                                          );
  }

  template <>
  Inline void AmpereCartesian<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
    mblock.em_fields(i, j, k, fld::ex1) += coeff * (
                                                mblock.em_fields(i, j, k - 1, fld::bx2) - mblock.em_fields(i, j, k, fld::bx2) +
                                                mblock.em_fields(i, j, k, fld::bx3) - mblock.em_fields(i, j - 1, k, fld::bx3)
                                              );
    mblock.em_fields(i, j, k, fld::ex2) += coeff * (
                                                mblock.em_fields(i - 1, j, k, fld::bx3) - mblock.em_fields(i, j, k, fld::bx3) +
                                                mblock.em_fields(i, j, k, fld::bx1) - mblock.em_fields(i, j, k - 1, fld::bx1)
                                              );
    mblock.em_fields(i, j, k, fld::ex3) += coeff * (
                                                mblock.em_fields(i, j - 1, k, fld::bx1) - mblock.em_fields(i, j, k, fld::bx1) +
                                                mblock.em_fields(i, j, k, fld::bx2) - mblock.em_fields(i - 1, j, k, fld::bx2)
                                              );
  }

} // namespace ntt

template class ntt::AmpereCartesian<ntt::ONE_D>;
template class ntt::AmpereCartesian<ntt::TWO_D>;
template class ntt::AmpereCartesian<ntt::THREE_D>;

#endif
