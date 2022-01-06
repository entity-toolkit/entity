#ifndef PIC_FIELDSOLVER_AMPERE_CURVILINEAR_H
#define PIC_FIELDSOLVER_AMPERE_CURVILINEAR_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

  // * * * * Curvilinear Ampere's law * * * * * * * * * * * * * * * *
  template <Dimension D>
  class AmpereCurvilinear : public FieldSolver<D> {
    using index_t = typename RealFieldND<D, 1>::size_type;
    real_t coeff;

  public:
    AmpereCurvilinear(
        const Meshblock<D>& m_mblock_,
        const real_t& coeff_)
        : FieldSolver<D> {m_mblock_},
          coeff(coeff_)
        {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void AmpereCurvilinear<ONE_D>::operator()(const index_t) const {
    throw std::logic_error("# 1d curvilinear ampere NOT IMPLEMENTED.");
  }

  template <>
  Inline void AmpereCurvilinear<TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};

    real_t inv_sqrt_detH_ij {
      ONE / m_mblock.grid->sqrt_det_h(i_, j_)
    };
    real_t inv_sqrt_detH_iPj {
      ONE / m_mblock.grid->sqrt_det_h(i_ + HALF, j_)
    };
    real_t inv_sqrt_detH_ijP {
      ONE / m_mblock.grid->sqrt_det_h(i_, j_ + HALF)
    };
    real_t h1_ijM {
      m_mblock.grid->h11(i_, j_ - HALF)
    };
    real_t h1_ijP {
      m_mblock.grid->h11(i_, j_ + HALF)
    };
    real_t h2_iPj {
      m_mblock.grid->h22(i_ + HALF, j_)
    };
    real_t h2_iMj {
      m_mblock.grid->h22(i_ - HALF, j_)
    };
    real_t h3_iMjP {
      m_mblock.grid->h33(i_ - HALF, j_ + HALF)
    };
    real_t h3_iPjM {
      m_mblock.grid->h33(i_ + HALF, j_ - HALF)
    };
    real_t h3_iPjP {
      m_mblock.grid->h33(i_ + HALF, j_ + HALF)
    };

    m_mblock.em_fields(i, j, fld::ex1) += coeff * inv_sqrt_detH_iPj * (
                                              h3_iPjP * m_mblock.em_fields(i, j, fld::bx3) - h3_iPjM * m_mblock.em_fields(i, j - 1, fld::bx3)
                                            );
    m_mblock.em_fields(i, j, fld::ex2) += coeff * inv_sqrt_detH_ijP * (
                                              h3_iMjP * m_mblock.em_fields(i - 1, j, fld::bx3) - h3_iPjP * m_mblock.em_fields(i, j, fld::bx3)
                                            );
    m_mblock.em_fields(i, j, fld::ex3) += coeff * inv_sqrt_detH_ij * (
                                              h1_ijM * m_mblock.em_fields(i, j - 1, fld::bx1) - h1_ijP * m_mblock.em_fields(i, j, fld::bx1) +
                                              h2_iPj * m_mblock.em_fields(i, j, fld::bx2) - h2_iMj * m_mblock.em_fields(i - 1, j, fld::bx2)
                                            );
  }

  template <>
  Inline void AmpereCurvilinear<THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    throw std::logic_error("# 3d curvilinear ampere NOT IMPLEMENTED.");
  }

} // namespace ntt

template class ntt::AmpereCurvilinear<ntt::ONE_D>;
template class ntt::AmpereCurvilinear<ntt::TWO_D>;
template class ntt::AmpereCurvilinear<ntt::THREE_D>;

#endif
