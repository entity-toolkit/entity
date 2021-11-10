#ifndef PIC_FIELDSOLVER_FARADAY_H
#define PIC_FIELDSOLVER_FARADAY_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

// * * * * Faraday's law * * * * * * * * * * * * * * *
template <Dimension D>
class Faraday : public FieldSolver<D> {
  using index_t = typename RealFieldND<D, 1>::size_type;
  real_t coeff_x1, coeff_x2, coeff_x3;

public:
  Faraday(
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

#ifdef HARDCODE_FLAT_COORDS
// flat cartesian system

template <>
Inline void Faraday<ONE_D>::operator()(const index_t i) const {
  m_mblock.em_fields(i, fld::bx2) += coeff_x1 * (
                m_mblock.em_fields(i + 1, fld::ex3) -
                m_mblock.em_fields(i, fld::ex3)
              );
  m_mblock.em_fields(i, fld::bx3) += coeff_x1 * (
                m_mblock.em_fields(i, fld::ex2) -
                m_mblock.em_fields(i + 1, fld::ex2)
              );
}

template <>
Inline void Faraday<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.em_fields(i, j, fld::bx1) += coeff_x2 * (
                m_mblock.em_fields(i, j, fld::ex3) -
                m_mblock.em_fields(i, j + 1, fld::ex3)
              );
  m_mblock.em_fields(i, j, fld::bx2) += coeff_x1 * (
                m_mblock.em_fields(i + 1, j, fld::ex3) -
                m_mblock.em_fields(i, j, fld::ex3)
              );
  m_mblock.em_fields(i, j, fld::bx3) += coeff_x2 * (
                m_mblock.em_fields(i, j + 1, fld::ex1) -
                m_mblock.em_fields(i, j, fld::ex1)
              ) + coeff_x1 * (
                m_mblock.em_fields(i, j, fld::ex2) -
                m_mblock.em_fields(i + 1, j, fld::ex2)
              );
}

template <>
Inline void Faraday<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  m_mblock.em_fields(i, j, k, fld::bx1) += coeff_x3 * (
                m_mblock.em_fields(i, j, k + 1, fld::ex2) -
                m_mblock.em_fields(i, j, k, fld::ex2)
              ) + coeff_x2 * (
                m_mblock.em_fields(i, j, k, fld::ex3) -
                m_mblock.em_fields(i, j + 1, k, fld::ex3)
              );
  m_mblock.em_fields(i, j, k, fld::bx2) += coeff_x1 * (
                m_mblock.em_fields(i + 1, j, k, fld::ex3) -
                m_mblock.em_fields(i, j, k, fld::ex3)
              ) + coeff_x3 * (
                m_mblock.em_fields(i, j, k, fld::ex1) -
                m_mblock.em_fields(i, j, k + 1, fld::ex1)
              );
  m_mblock.em_fields(i, j, k, fld::bx3) += coeff_x2 * (
                m_mblock.em_fields(i, j + 1, k, fld::ex1) -
                m_mblock.em_fields(i, j, k, fld::ex1)
              ) + coeff_x1 * (
                m_mblock.em_fields(i, j, k, fld::ex2) -
                m_mblock.em_fields(i + 1, j, k, fld::ex2)
              );
}

#else
// curvilinear coordinate system

template <>
Inline void Faraday<ONE_D>::operator()(const index_t i) const {
  real_t x1 {m_mblock.convert_iTOx1(i)};
  real_t dx1 {(m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0])};

  real_t hx1_iP {{m_mblock.Jacobian_h1(x1 + 0.5 * dx1)}};
  real_t hx2_iP {{m_mblock.Jacobian_h2(x1 + 0.5 * dx1)}};
  real_t hx3_iP {{m_mblock.Jacobian_h3(x1 + 0.5 * dx1)}};
  real_t hx2_i {{m_mblock.Jacobian_h2(x1)}};
  real_t hx2_iP1 {{m_mblock.Jacobian_h2(x1 + dx1)}};
  real_t hx3_i {{m_mblock.Jacobian_h3(x1)}};
  real_t hx3_iP1 {{m_mblock.Jacobian_h3(x1 + dx1)}};

  m_mblock.em_fields(i, fld::bx2) += coeff_x1 * (
                hx3_iP1 * m_mblock.em_fields(i + 1, fld::ex3) -
                hx3_i * m_mblock.em_fields(i, fld::ex3)
              ) / (hx1_iP * hx3_iP);
  m_mblock.em_fields(i, fld::bx3) += coeff_x1 * (
                hx2_i * m_mblock.em_fields(i, fld::ex2) -
                hx2_iP1 * m_mblock.em_fields(i + 1, fld::ex2)
              ) / (hx1_iP * hx2_iP);
}

template <>
Inline void Faraday<TWO_D>::operator()(const index_t i, const index_t j) const {
  real_t x1 {m_mblock.convert_iTOx1(i)};
  real_t x2 {m_mblock.convert_jTOx2(j)};

  real_t dx1 {(m_extent[1] - m_extent[0]) / static_cast<real_t>(m_resolution[0])};
  real_t dx2 {(m_extent[3] - m_extent[2]) / static_cast<real_t>(m_resolution[1])};

  real_t hx2_ijP {m_mblock.Jacobian_h2(x1, x2 + 0.5 * dx2)};
  real_t hx3_ijP {m_mblock.Jacobian_h3(x1, x2 + 0.5 * dx2)};
  real_t hx1_iPj {m_mblock.Jacobian_h1(x1 + 0.5 * dx1, x2)};
  real_t hx3_iPj {m_mblock.Jacobian_h3(x1 + 0.5 * dx1, x2)};
  real_t hx1_iPjP {m_mblock.Jacobian_h1(x1 + 0.5 * dx1, x2 + 0.5 * dx2)};
  real_t hx2_iPjP {m_mblock.Jacobian_h2(x1 + 0.5 * dx1, x2 + 0.5 * dx2)};

  real_t hx3_ij {m_mblock.Jacobian_h3(x1, x2)};
  real_t hx3_ijP1 {m_mblock.Jacobian_h3(x1, x2 + dx2)};
  real_t hx3_iP1j {m_mblock.Jacobian_h3(x1 + dx1, x2)};
  real_t hx1_iPjP1 {m_mblock.Jacobian_h1(x1 + 0.5 * dx1, x2 + dx2)};

  real_t hx2_iP1jP {m_mblock.Jacobian_h2(x1 + dx1, x2 + 0.5 * dx2)};
  real_t hx2_ijP {m_mblock.Jacobian_h2(x1, x2 + 0.5 * dx2)}

  m_mblock.em_fields(i, j, fld::bx1) += coeff_x2 * (
                hx3_ij * m_mblock.em_fields(i, j, fld::ex3) -
                hx3_ijP1 * m_mblock.em_fields(i, j + 1, fld::ex3)
              ) / (hx2_ijP * hx3_ijP);
  m_mblock.em_fields(i, j, fld::bx2) += coeff_x1 * (
                hx3_iP1j * m_mblock.em_fields(i + 1, j, fld::ex3) -
                hx3_ij * m_mblock.em_fields(i, j, fld::ex3)
              ) / (hx1_iPj * hx3_iPj);
  m_mblock.em_fields(i, j, fld::bx3) += (coeff_x2 * (
                hx1_iPjP1 * m_mblock.em_fields(i, j + 1, fld::ex1) -
                hx1_iPj * m_mblock.em_fields(i, j, fld::ex1)
              ) + coeff_x1 * (
                hx2_iP1jP * m_mblock.em_fields(i, j, fld::ex2) -
                hx2_ijP * hx2 * m_mblock.em_fields(i + 1, j, fld::ex2))
            ) / (hx1_iPjP * hx2_iPjP);
}

template <>
Inline void Faraday<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  throw std::logic_error("# NOT IMPLEMENTED.");
  m_mblock.em_fields(i, j, k, fld::bx1) += coeff_x3 * (
                m_mblock.em_fields(i, j, k + 1, fld::ex2) -
                m_mblock.em_fields(i, j, k, fld::ex2)
              ) + coeff_x2 * (
                m_mblock.em_fields(i, j, k, fld::ex3) -
                m_mblock.em_fields(i, j + 1, k, fld::ex3)
              );
  m_mblock.em_fields(i, j, k, fld::bx2) += coeff_x1 * (
                m_mblock.em_fields(i + 1, j, k, fld::ex3) -
                m_mblock.em_fields(i, j, k, fld::ex3)
              ) + coeff_x3 * (
                m_mblock.em_fields(i, j, k, fld::ex1) -
                m_mblock.em_fields(i, j, k + 1, fld::ex1)
              );
  m_mblock.em_fields(i, j, k, fld::bx3) += coeff_x2 * (
                m_mblock.em_fields(i, j + 1, k, fld::ex1) -
                m_mblock.em_fields(i, j, k, fld::ex1)
              ) + coeff_x1 * (
                m_mblock.em_fields(i, j, k, fld::ex2) -
                m_mblock.em_fields(i + 1, j, k, fld::ex2)
              );
}


#endif

// clang-format on

} // namespace ntt

template class ntt::Faraday<ntt::ONE_D>;
template class ntt::Faraday<ntt::TWO_D>;
template class ntt::Faraday<ntt::THREE_D>;

#endif
