#ifndef PIC_FIELDSOLVER_FARADAY_CURVILINEAR_H
#define PIC_FIELDSOLVER_FARADAY_CURVILINEAR_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

  // * * * * Curvilinear Faraday's law * * * * * * * * * * * * * * *
  template <Dimension D>
  class FaradayCurvilinear : public FieldSolver<D> {
    using index_t = typename RealFieldND<D, 1>::size_type;
    real_t coeff;

  public:
    FaradayCurvilinear(
        const Meshblock<D>& m_mblock_,
        const real_t& coeff_)
        : FieldSolver<D> {m_mblock_},
          coeff(coeff_) {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void FaradayCurvilinear<ONE_D>::operator()(const index_t) const {
    throw std::logic_error("# 1d curvilinear faraday NOT IMPLEMENTED.");
  }

  template <>
  Inline void FaradayCurvilinear<TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t i_half {static_cast<real_t>(i - N_GHOSTS) + HALF};
    real_t i_one {static_cast<real_t>(i - N_GHOSTS) + ONE};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
    real_t j_half {static_cast<real_t>(j - N_GHOSTS) + HALF};
    real_t j_one {static_cast<real_t>(j - N_GHOSTS) + ONE};

    real_t inv_sqrt_detH_iPj {
      ONE / m_mblock.m_coord_system->sqrt_det_h(i_half, j_)
    };
    real_t inv_sqrt_detH_ijP {
      ONE / m_mblock.m_coord_system->sqrt_det_h(i_, j_half)
    };
    real_t inv_sqrt_detH_iPjP {
      ONE / m_mblock.m_coord_system->sqrt_det_h(i_half, j_half)
    };
    real_t h1_iPjP1 {
      m_mblock.m_coord_system->h11(i_half, j_one)
    };
    real_t h1_iPj {
      m_mblock.m_coord_system->h11(i_half, j_)
    };
    real_t h2_iP1jP {
      m_mblock.m_coord_system->h22(i_one, j_half)
    };
    real_t h2_ijP {
      m_mblock.m_coord_system->h22(i_, j_half)
    };
    real_t h3_ij {
      m_mblock.m_coord_system->h33(i_, j_)
    };
    real_t h3_iP1j {
      m_mblock.m_coord_system->h33(i_one, j_)
    };
    real_t h3_ijP1 {
      m_mblock.m_coord_system->h33(i_, j_one)
    };

    m_mblock.em_fields(i, j, fld::bx1) += coeff * inv_sqrt_detH_ijP * (
                                              h3_ij * m_mblock.em_fields(i, j, fld::ex3) - h3_ijP1 * m_mblock.em_fields(i, j + 1, fld::ex3)
                                            );
    m_mblock.em_fields(i, j, fld::bx2) += coeff * inv_sqrt_detH_iPj * (
                                              h3_iP1j * m_mblock.em_fields(i + 1, j, fld::ex3) - h3_ij * m_mblock.em_fields(i, j, fld::ex3)
                                            );
    m_mblock.em_fields(i, j, fld::bx3) += coeff * inv_sqrt_detH_iPjP * (
                                              h1_iPjP1 * m_mblock.em_fields(i, j + 1, fld::ex1) - h1_iPj * m_mblock.em_fields(i, j, fld::ex1) +
                                              h2_ijP * m_mblock.em_fields(i, j, fld::ex2) - h2_iP1jP * m_mblock.em_fields(i + 1, j, fld::ex2)
                                            );
  }

  template <>
  Inline void FaradayCurvilinear<THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    throw std::logic_error("# 3d curvilinear faraday NOT IMPLEMENTED.");
  }
} // namespace ntt

template class ntt::FaradayCurvilinear<ntt::ONE_D>;
template class ntt::FaradayCurvilinear<ntt::TWO_D>;
template class ntt::FaradayCurvilinear<ntt::THREE_D>;

#endif
