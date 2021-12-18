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
    real_t coeff_x1, coeff_x2, coeff_x3;

  public:
    FaradayCurvilinear(
        const Meshblock<D>& m_mblock_,
        const real_t& coeff_x1_,
        const real_t& coeff_x2_,
        const real_t& coeff_x3_)
        : FieldSolver<D> {m_mblock_}, coeff_x1(coeff_x1_), coeff_x2(coeff_x2_), coeff_x3(coeff_x3_) {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void FaradayCurvilinear<ONE_D>::operator()(const index_t i) const {
    real_t x1 {m_mblock.convert_iTOx1(i)};
    real_t dx1 {(m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])};

    real_t inv_sqrt_detH_iP {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx1)};
    real_t h2_i {m_mblock.m_coord_system->hx2(x1)};
    real_t h2_iP1 {m_mblock.m_coord_system->hx2(x1 + dx1)};
    real_t h3_i {m_mblock.m_coord_system->hx3(x1)};
    real_t h3_iP1 {m_mblock.m_coord_system->hx3(x1 + dx1)};

    m_mblock.em_fields(i, fld::bx2) += inv_sqrt_detH_iP * coeff_x1 * (h3_iP1 * m_mblock.em_fields(i + 1, fld::ex3) - h3_i * m_mblock.em_fields(i, fld::ex3));
    m_mblock.em_fields(i, fld::bx3) += inv_sqrt_detH_iP * coeff_x1 * (h2_i * m_mblock.em_fields(i, fld::ex2) - h2_iP1 * m_mblock.em_fields(i + 1, fld::ex2));
  }

  template <>
  Inline void FaradayCurvilinear<TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t x1 {m_mblock.convert_iTOx1(i)};
    real_t x2 {m_mblock.convert_jTOx2(j)};
    real_t dx1 {(m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])};
    real_t dx2 {(m_mblock.m_extent[3] - m_mblock.m_extent[2]) / static_cast<real_t>(m_mblock.m_resolution[1])};

    real_t inv_sqrt_detH_iPj {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx2, x2)};
    real_t inv_sqrt_detH_ijP {ONE / m_mblock.m_coord_system->sqrt_det_h(x1, x2 + 0.5 * dx2)};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx1, x2 + 0.5 * dx2)};
    real_t h1_iPjP1 {m_mblock.m_coord_system->hx1(x1 + 0.5 * dx1, x2 + dx2)};
    real_t h1_iPj {m_mblock.m_coord_system->hx1(x1 + 0.5 * dx1, x2)};
    real_t h2_iP1jP {m_mblock.m_coord_system->hx2(x1 + dx1, x2 + 0.5 * dx2)};
    real_t h2_ijP {m_mblock.m_coord_system->hx2(x1, x2 + 0.5 * dx2)};
    real_t h3_ij {m_mblock.m_coord_system->hx3(x1, x2)};
    real_t h3_iP1j {m_mblock.m_coord_system->hx3(x1 + dx1, x2)};
    real_t h3_ijP1 {m_mblock.m_coord_system->hx3(x1, x2 + dx2)};

    m_mblock.em_fields(i, j, fld::bx1) += inv_sqrt_detH_ijP * coeff_x2 * (h3_ij * m_mblock.em_fields(i, j, fld::ex3) - h3_ijP1 * m_mblock.em_fields(i, j + 1, fld::ex3));
    m_mblock.em_fields(i, j, fld::bx2) += inv_sqrt_detH_iPj * coeff_x1 * (h3_iP1j * m_mblock.em_fields(i + 1, j, fld::ex3) - h3_ij * m_mblock.em_fields(i, j, fld::ex3));
    m_mblock.em_fields(i, j, fld::bx3) += inv_sqrt_detH_iPjP * (coeff_x2 * (h1_iPjP1 * m_mblock.em_fields(i, j + 1, fld::ex1) - h1_iPj * m_mblock.em_fields(i, j, fld::ex1)) + coeff_x1 * (h2_ijP * m_mblock.em_fields(i, j, fld::ex2) - h2_iP1jP * m_mblock.em_fields(i + 1, j, fld::ex2)));
  }

  template <>
  Inline void FaradayCurvilinear<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
    UNUSED(i);
    UNUSED(j);
    UNUSED(k);
    throw std::logic_error("# 3d curvilinear faraday NOT IMPLEMENTED.");
    // real_t x1 {m_mblock.convert_iTOx1(i)};
    // real_t x2 {m_mblock.convert_jTOx2(j)};
    // real_t x3 {m_mblock.convert_kTOx3(k)};
    // real_t dx1 {(m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])};
    // real_t dx2 {(m_mblock.m_extent[3] - m_mblock.m_extent[2]) / static_cast<real_t>(m_mblock.m_resolution[1])};
    // real_t dx3 {(m_mblock.m_extent[5] - m_mblock.m_extent[4]) / static_cast<real_t>(m_mblock.m_resolution[2])};
    // real_t inv_sqrt_detH_ijPkP {ONE / m_mblock.m_coord_system->sqrt_det_h(x1, x2 + 0.5 * dx2, x3 + 0.5 * dx3)};
    // real_t inv_sqrt_detH_iPjkP {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx1, x2, x3 + 0.5 * dx3)};
    // real_t inv_sqrt_detH_iPjPk {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx1, x2 + 0.5 * dx2, x3)};
    //
    //
    // m_mblock.em_fields(i, j, k, fld::bx1) += inv_sqrt_detH_ijPkP * (coeff_x3 * (m_mblock.em_fields(i, j, k + 1, fld::ex2) - m_mblock.em_fields(i, j, k, fld::ex2)) + coeff_x2 * (m_mblock.em_fields(i, j, k, fld::ex3) - m_mblock.em_fields(i, j + 1, k, fld::ex3)));
    // m_mblock.em_fields(i, j, k, fld::bx2) += inv_sqrt_detH_iPjkP * (coeff_x1 * (m_mblock.em_fields(i + 1, j, k, fld::ex3) - m_mblock.em_fields(i, j, k, fld::ex3)) + coeff_x3 * (m_mblock.em_fields(i, j, k, fld::ex1) - m_mblock.em_fields(i, j, k + 1, fld::ex1)));
    // m_mblock.em_fields(i, j, k, fld::bx3) += inv_sqrt_detH_iPjPk * (coeff_x2 * (m_mblock.em_fields(i, j + 1, k, fld::ex1) - m_mblock.em_fields(i, j, k, fld::ex1)) + coeff_x1 * (m_mblock.em_fields(i, j, k, fld::ex2) - m_mblock.em_fields(i + 1, j, k, fld::ex2)));
  }
} // namespace ntt

template class ntt::FaradayCurvilinear<ntt::ONE_D>;
template class ntt::FaradayCurvilinear<ntt::TWO_D>;
template class ntt::FaradayCurvilinear<ntt::THREE_D>;

#endif
