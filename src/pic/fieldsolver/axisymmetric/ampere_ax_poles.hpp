#ifndef PIC_FIELDSOLVER_AMPERE_AXISYMMETRIC_POLES_H
#define PIC_FIELDSOLVER_AMPERE_AXISYMMETRIC_POLES_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

  // * * * * Ampere's law for E_r and E_theta near axes * * * * * * * * * * * * * * * *
  template <Dimension D>
  class AmpereAxisymmetricPoles : public FieldSolver<D> {
    using index_t = typename RealFieldND<D, 1>::size_type;
    real_t coeff;

  public:
    AmpereAxisymmetricPoles(const Meshblock<D>& m_mblock_, const real_t& coeff_)
      : FieldSolver<D> {m_mblock_}, coeff(coeff_) {}
    Inline void operator()(const index_t) const;
  };

  template <>
  Inline void AmpereAxisymmetricPoles<TWO_D>::operator()(const index_t i) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t i_half {static_cast<real_t>(i - N_GHOSTS) + HALF};
    real_t i_Mhalf {static_cast<real_t>(i - N_GHOSTS) - HALF};
    real_t j_max_half {static_cast<real_t>(m_mblock.Nj) + HALF};

    real_t inv_polar_area_iPj {ONE / m_mblock.m_coord_system->polar_area(i_half, HALF)};
    real_t h3_min_iPjP {m_mblock.m_coord_system->h33(i_half, HALF)};
    real_t h3_max_iPjP {m_mblock.m_coord_system->h33(i_half, j_max_half)};

    real_t inv_sqrt_detH_ijP {ONE / m_mblock.m_coord_system->sqrt_det_h(i_, HALF)};
    real_t h3_min_iMjP {m_mblock.m_coord_system->h33(i_Mhalf, HALF)};

    index_t j_min {N_GHOSTS};
    index_t j_max {m_mblock.Nj + N_GHOSTS - 1};

    // theta = 0
    m_mblock.em_fields(i, j_min, fld::ex1) += inv_polar_area_iPj * coeff * (h3_min_iPjP * m_mblock.em_fields(i, j_min, fld::bx3));
    // theta = pi
    m_mblock.em_fields(i, j_max + 1, fld::ex1) -= inv_polar_area_iPj * coeff * (h3_max_iPjP * m_mblock.em_fields(i, j_max, fld::bx3));

    // j = jmin + 1/2
    m_mblock.em_fields(i, j_min, fld::ex2) += inv_sqrt_detH_ijP * coeff * (
                                                    h3_min_iMjP * m_mblock.em_fields(i - 1, j_min, fld::bx3) - h3_min_iPjP * m_mblock.em_fields(i, j_min, fld::bx3)
                                                );
  }

} // namespace ntt

template class ntt::AmpereAxisymmetricPoles<ntt::TWO_D>;

#endif
