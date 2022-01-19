#ifndef PIC_FIELDSOLVER_AMPERE_AXISYMMETRIC_RMIN_H
#define PIC_FIELDSOLVER_AMPERE_AXISYMMETRIC_RMIN_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

  // * * * * Ampere's law for Er at rmin * * * * * * * * * * * * * * * *
  template <Dimension D>
  class AmpereAxisymmetricRmin : public FieldSolver<D> {
    using index_t = typename RealFieldND<D, 1>::size_type;
    real_t coeff_x2;

  public:
    AmpereAxisymmetricRmin(
        const Meshblock<D>& m_mblock_,
        const real_t& coeff_x2_)
        : FieldSolver<D> {m_mblock_}, coeff_x2(coeff_x2_) {}
    Inline void operator()(const index_t) const;
  };

  template <>
  Inline void AmpereAxisymmetricRmin<TWO_D>::operator()(const index_t j) const {
    index_t i_dw {N_GHOSTS};

    real_t x1 {m_mblock.convert_iTOx1(i_dw)};
    real_t x2 {m_mblock.convert_jTOx2(j)};
    real_t dx1 {(m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])};
    real_t dx2 {(m_mblock.m_extent[3] - m_mblock.m_extent[2]) / static_cast<real_t>(m_mblock.m_resolution[1])};

    real_t inv_sqrt_detH_iPj {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx1, x2)};
    real_t h3_iPjM {m_mblock.m_coord_system->hx3(x1 + 0.5 * dx1, x2 - 0.5 * dx2)};
    real_t h3_iPjP {m_mblock.m_coord_system->hx3(x1 + 0.5 * dx1, x2 + 0.5 * dx2)};

    m_mblock.em_fields(i_dw, j, fld::ex1) += inv_sqrt_detH_iPj * coeff_x2 * (h3_iPjP * m_mblock.em_fields(i_dw, j, fld::bx3) - h3_iPjM * m_mblock.em_fields(i_dw, j - 1, fld::bx3));
  }


} // namespace ntt

template class ntt::AmpereAxisymmetricRmin<ntt::TWO_D>;

#endif
