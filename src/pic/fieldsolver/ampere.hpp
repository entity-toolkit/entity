#ifndef PIC_FIELDSOLVER_AMPERE_H
#define PIC_FIELDSOLVER_AMPERE_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

// * * * * Ampere's law * * * * * * * * * * * * * * * *
template <Dimension D>
class Ampere : public FieldSolver<D> {
  using index_t = typename RealFieldND<D, 1>::size_type;
  real_t coeff_x1, coeff_x2, coeff_x3;

public:
  Ampere(
      const Meshblock<D>& m_mblock_,
      const real_t& coeff_x1_,
      const real_t& coeff_x2_,
      const real_t& coeff_x3_)
      : FieldSolver<D> {m_mblock_}, coeff_x1(coeff_x1_), coeff_x2(coeff_x2_), coeff_x3(coeff_x3_) {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

// clang-format off

#ifndef CURVILINEAR_COORDS
// flat cartesian system

template <>
Inline void Ampere<ONE_D>::operator()(const index_t i) const {
  m_mblock.em_fields(i, fld::ex2) += coeff_x1 * (
                m_mblock.em_fields(i - 1, fld::bx3) -
                m_mblock.em_fields(i, fld::bx3)
              );
  m_mblock.em_fields(i, fld::ex3) += coeff_x1 * (
                m_mblock.em_fields(i, fld::bx2) -
                m_mblock.em_fields(i - 1, fld::bx2)
              );
}

template <>
Inline void Ampere<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.em_fields(i, j, fld::ex1) += coeff_x2 * (
                m_mblock.em_fields(i, j, fld::bx3) -
                m_mblock.em_fields(i, j - 1, fld::bx3)
              );
  m_mblock.em_fields(i, j, fld::ex2) += coeff_x1 * (
                m_mblock.em_fields(i - 1, j, fld::bx3) -
                m_mblock.em_fields(i, j, fld::bx3)
              );
  m_mblock.em_fields(i, j, fld::ex3) += coeff_x2 * (
                m_mblock.em_fields(i, j - 1, fld::bx1)
                - m_mblock.em_fields(i, j, fld::bx1)
              ) + coeff_x1 * (
                m_mblock.em_fields(i, j, fld::bx2) -
                m_mblock.em_fields(i - 1, j, fld::bx2)
              );
}

template <>
Inline void Ampere<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  m_mblock.em_fields(i, j, k, fld::ex1) += coeff_x3 * (
                m_mblock.em_fields(i, j, k - 1, fld::bx2) -
                m_mblock.em_fields(i, j, k, fld::bx2)
              ) + coeff_x2 * (
                m_mblock.em_fields(i, j, k, fld::bx3) -
                m_mblock.em_fields(i, j - 1, k, fld::bx3)
              );
  m_mblock.em_fields(i, j, k, fld::ex2) += coeff_x1 * (
                m_mblock.em_fields(i - 1, j, k, fld::bx3) -
                m_mblock.em_fields(i, j, k, fld::bx3)
              ) + coeff_x3 * (
                m_mblock.em_fields(i, j, k, fld::bx1) -
                m_mblock.em_fields(i, j, k - 1, fld::bx1)
              );
  m_mblock.em_fields(i, j, k, fld::ex3) += coeff_x2 * (
                m_mblock.em_fields(i, j - 1, k, fld::bx1) -
                m_mblock.em_fields(i, j, k, fld::bx1)
              ) + coeff_x1 * (
                m_mblock.em_fields(i, j, k, fld::bx2) -
                m_mblock.em_fields(i - 1, j, k, fld::bx2)
              );
}

#else
// curvilinear coordinate system

template <>
Inline void Ampere<ONE_D>::operator()(const index_t i) const {
  real_t x1 {m_mblock.convert_iTOx1(i)};
  real_t dx1 {(m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0])};

  real_t hx1_i {m_mblock.Jacobian_h1(x1)};
  real_t hx2_i {m_mblock.Jacobian_h2(x1)};
  real_t hx3_i {m_mblock.Jacobian_h3(x1)};
  real_t hx3_iP {m_mblock.Jacobian_h3(x1 + 0.5 * dx1)};
  real_t hx3_iM {m_mblock.Jacobian_h3(x1 - 0.5 * dx1)};
  real_t hx2_iP {m_mblock.Jacobian_h2(x1 + 0.5 * dx1)};
  real_t hx2_iM {m_mblock.Jacobian_h2(x1 - 0.5 * dx1)};

  m_mblock.em_fields(i, fld::ex2) += coeff_x1 * (
                hx3_iM * m_mblock.em_fields(i - 1, fld::bx3) -
                hx3_iP * m_mblock.em_fields(i, fld::bx3)
              ) / (hx1_i * hx3_i);
  m_mblock.em_fields(i, fld::ex3) += coeff_x1 * (
                hx2_iP * m_mblock.em_fields(i, fld::bx2) -
                hx2_iM * m_mblock.em_fields(i - 1, fld::bx2)
              ) / (hx1_i * hx2_i);
}

template <>
Inline void Ampere<TWO_D>::operator()(const index_t i, const index_t j) const {
  real_t x1 {m_mblock.convert_iTOx1(i)};
  real_t x2 {m_mblock.convert_jTOx2(j)};

  real_t dx1 {(m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0])};
  real_t dx2 {(m_extent[3] - m_extent[2]) / static_cast<real_t>(m_resolution[1])};

  real_t hx2_iPj {m_mblock.Jacobian_h2(x1 + 0.5 * dx1, x2)};
  real_t hx3_iPj {m_mblock.Jacobian_h3(x1 + 0.5 * dx1, x2)};
  real_t hx3_iPjP {m_mblock.Jacobian_h3(x1 + 0.5 * dx1, x2 + 0.5 * dx2)};
  real_t hx3_iPjM {m_mblock.Jacobian_h3(x1 + 0.5 * dx1, x2 - 0.5 * dx2)};

  real_t hx1_ijP {m_mblock.Jacobian_h1(x1, x2 + 0.5 * dx2)};
  real_t hx3_ijP {m_mblock.Jacobian_h3(x1, x2 + 0.5 * dx2)};
  real_t hx3_iMjP {m_mblock.Jacobian_h3(x1 - 0.5 * dx1, x2 + 0.5 * dx2)};

  real_t hx1_ij {m_mblock.Jacobian_h1(x1, x2)};
  real_t hx2_ij {m_mblock.Jacobian_h2(x1, x2)};
  real_t hx1_ijP {m_mblock.Jacobian_h1(x1, x2 + 0.5 * dx2)};
  real_t hx1_ijM {m_mblock.Jacobian_h1(x1, x2 - 0.5 * dx2)};
  real_t hx2_iPj {m_mblock.Jacobian_h2(x1 + 0.5 * dx1, x2)};
  real_t hx2_iMj {m_mblock.Jacobian_h2(x1 - 0.5 * dx1, x2)};

  m_mblock.em_fields(i, j, fld::ex1) += coeff_x2 * (
                hx3_iPjP * m_mblock.em_fields(i, j, fld::bx3) -
                hx3_iPjM * m_mblock.em_fields(i, j - 1, fld::bx3)
              ) / (hx2_iPj * hx3_iPj);

  m_mblock.em_fields(i, j, fld::ex2) += coeff_x1 * (
                hx3_iMjP * m_mblock.em_fields(i - 1, j, fld::bx3) -
                hx3_iPjP * m_mblock.em_fields(i, j, fld::bx3)
              ) / (hx1_ijP * hx3_ijP);

  m_mblock.em_fields(i, j, fld::ex3) += (coeff_x2 * (
                hx1_ijM * m_mblock.em_fields(i, j - 1, fld::bx1) -
                hx1_ijP * m_mblock.em_fields(i, j, fld::bx1)
              ) + coeff_x1 * (
                hx2_iPj * m_mblock.em_fields(i, j, fld::bx2) -
                hx2_iMj * m_mblock.em_fields(i - 1, j, fld::bx2))
            ) / (hx1_ij * hx2_ij);
}

template <>
Inline void Ampere<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  real_t x1 {m_mblock.convert_iTOx1(i)};
  real_t x2 {m_mblock.convert_jTOx2(j)};
  real_t x3 {m_mblock.convert_kTOx3(k)};

  real_t dx1 {(m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0])};
  real_t dx2 {(m_extent[3] - m_extent[2]) / static_cast<real_t>(m_resolution[1])};
  real_t dx3 {(m_extent[5] - m_extent[4]) / static_cast<real_t>(m_resolution[2])};

  real_t hx2_iPjk {m_mblock.Jacobian_h2(x1 + 0.5 * dx1, x2, x3)};
  real_t hx3_iPjk {m_mblock.Jacobian_h3(x1 + 0.5 * dx1, x2, x3)};
  real_t hx2_iPjkM {m_mblock.Jacobian_h2(x1 + 0.5 * dx1, x2, x3 - 0.5 * dx3)};
  real_t hx2_iPjkP {m_mblock.Jacobian_h2(x1 + 0.5 * dx1, x2, x3 + 0.5 * dx3)};
  real_t hx3_iPjPk {m_mblock.Jacobian_h3(x1 + 0.5 * dx1, x2 + 0.5 * dx2, x3)};
  real_t hx3_iPjMk {m_mblock.Jacobian_h3(x1 + 0.5 * dx1, x2 - 0.5 * dx2, x3)};

  real_t hx1_ijPk  {m_mblock.Jacobian_h1(x1, x2 + 0.5 * dx2, x3)};
  real_t hx3_ijPk  {m_mblock.Jacobian_h3(x1, x2 + 0.5 * dx2, x3)};
  real_t hx3_iMjPk {m_mblock.Jacobian_h3(x1 - 0.5 * dx1, x2 + 0.5 * dx2, x3)};
  real_t hx3_iPjPk {m_mblock.Jacobian_h3(x1 + 0.5 * dx1, x2 + 0.5 * dx2, x3)};
  real_t hx1_ijPkP {m_mblock.Jacobian_h1(x1, x2 + 0.5 * dx2, x3 + 0.5 * dx3)};
  real_t hx1_ijPkM {m_mblock.Jacobian_h1(x1, x2 + 0.5 * dx2, x3 - 0.5 * dx3)};

  real_t hx1_ijkP  {m_mblock.Jacobian_h1(x1, x2, x3 + 0.5 * dx3)};
  real_t hx2_ijkP  {m_mblock.Jacobian_h2(x1, x2, x3 + 0.5 * dx3)};
  real_t hx1_ijMkP {m_mblock.Jacobian_h1(x1, x2 - 0.5 * dx2, x3 + 0.5 * dx3)};
  real_t hx1_ijPkP {m_mblock.Jacobian_h1(x1, x2 + 0.5 * dx2, x3 + 0.5 * dx3)};
  real_t hx2_iPjkP {m_mblock.Jacobian_h2(x1 + 0.5 * dx1, x2, x3 + 0.5 * dx3)};
  real_t hx2_iMjkP {m_mblock.Jacobian_h2(x1 - 0.5 * dx1, x2, x3 + 0.5 * dx3)};

  m_mblock.em_fields(i, j, k, fld::ex1) += coeff_x3 * (
                hx2_iPjkM * m_mblock.em_fields(i, j, k - 1, fld::bx2) -
                hx2_iPjkP * m_mblock.em_fields(i, j, k, fld::bx2)
              ) + coeff_x2 * (
                hx3_iPjPk * m_mblock.em_fields(i, j, k, fld::bx3) -
                hx3_iPjMk * m_mblock.em_fields(i, j - 1, k, fld::bx3)
            ) / (hx2_iPjk * hx3_iPjk);
  m_mblock.em_fields(i, j, k, fld::ex2) += coeff_x1 * (
                hx3_iMjPk * m_mblock.em_fields(i - 1, j, k, fld::bx3) -
                hx3_iPjPk * m_mblock.em_fields(i, j, k, fld::bx3)
              ) + coeff_x3 * (
                hx1_ijPkP * m_mblock.em_fields(i, j, k, fld::bx1) -
                hx1_ijPkM * m_mblock.em_fields(i, j, k - 1, fld::bx1)
            ) / (hx1_ijPk * hx3_ijPk);
  m_mblock.em_fields(i, j, k, fld::ex3) += coeff_x2 * (
                hx1_ijMkP * m_mblock.em_fields(i, j - 1, k, fld::bx1) -
                hx1_ijPkP * m_mblock.em_fields(i, j, k, fld::bx1)
              ) + coeff_x1 * (
                hx2_iPjkP * m_mblock.em_fields(i, j, k, fld::bx2) -
                hx2_iMjkP * m_mblock.em_fields(i - 1, j, k, fld::bx2)
            ) / (hx1_ijkP * hx2_ijkP);
}

#endif

// clang-format on

} // namespace ntt

template class ntt::Ampere<ntt::ONE_D>;
template class ntt::Ampere<ntt::TWO_D>;
template class ntt::Ampere<ntt::THREE_D>;

#endif
