#ifndef PIC_FIELDSOLVER_FARADAY_AXISYMMETRIC_RMIN_H
#define PIC_FIELDSOLVER_FARADAY_AXISYMMETRIC_RMIN_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

  // * * * * Faraday's law for B_theta, Bphi at rmin * * * * * * * * * * * * * * *
  template <Dimension D>
  class FaradayAxisymmetricRmin : public FieldSolver<D> {
    using index_t = typename RealFieldND<D, 1>::size_type;
    real_t coeff_x1, coeff_x2;

  public:
    FaradayAxisymmetricRmin(
        const Meshblock<D>& m_mblock_,
        const real_t& coeff_x1_,
        const real_t& coeff_x2_)
        : FieldSolver<D> {m_mblock_}, coeff_x1(coeff_x1_), coeff_x2(coeff_x2_) {}
    Inline void operator()(const index_t) const;
  };

  template <>
  Inline void FaradayAxisymmetricRmin<TWO_D>::operator()(const index_t j) const {
    index_t i_dw {N_GHOSTS};

    real_t x1 {m_mblock.convert_iTOx1(i_dw)};
    real_t x2 {m_mblock.convert_jTOx2(j)};
    real_t dx1 {(m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])};
    real_t dx2 {(m_mblock.m_extent[3] - m_mblock.m_extent[2]) / static_cast<real_t>(m_mblock.m_resolution[1])};

    real_t inv_sqrt_detH_iPj {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx2, x2)};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx1, x2 + 0.5 * dx2)};
    real_t h1_iPjP1 {m_mblock.m_coord_system->hx1(x1 + 0.5 * dx1, x2 + dx2)};
    real_t h1_iPj {m_mblock.m_coord_system->hx1(x1 + 0.5 * dx1, x2)};
    real_t h2_iP1jP {m_mblock.m_coord_system->hx2(x1 + dx1, x2 + 0.5 * dx2)};
    real_t h2_ijP {m_mblock.m_coord_system->hx2(x1, x2 + 0.5 * dx2)};
    real_t h3_ij {m_mblock.m_coord_system->hx3(x1, x2)};
    real_t h3_iP1j {m_mblock.m_coord_system->hx3(x1 + dx1, x2)};

    m_mblock.em_fields(i_dw, j, fld::bx2) += inv_sqrt_detH_iPj * coeff_x1 * (h3_iP1j * m_mblock.em_fields(i_dw + 1, j, fld::ex3) - h3_ij * m_mblock.em_fields(i_dw, j, fld::ex3));
    m_mblock.em_fields(i_dw, j, fld::bx3) += inv_sqrt_detH_iPjP * (coeff_x2 * (h1_iPjP1 * m_mblock.em_fields(i_dw, j + 1, fld::ex1) - h1_iPj * m_mblock.em_fields(i_dw, j, fld::ex1)) + coeff_x1 * (h2_ijP * m_mblock.em_fields(i_dw, j, fld::ex2) - h2_iP1jP * m_mblock.em_fields(i_dw + 1, j, fld::ex2)));
  }
} // namespace ntt

template class ntt::FaradayAxisymmetricRmin<ntt::TWO_D>;

#endif
