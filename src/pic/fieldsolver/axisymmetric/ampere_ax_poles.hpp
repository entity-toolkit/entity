// #ifndef PIC_FIELDSOLVER_AMPERE_AXISYMMETRIC_POLES_H
// #define PIC_FIELDSOLVER_AMPERE_AXISYMMETRIC_POLES_H

// #include "global.h"
// #include "meshblock.h"
// #include "fieldsolver.h"

// namespace ntt {

//   // * * * * Ampere's law for E_r and E_theta near axes * * * * * * * * * * * * * * * *
//   template <Dimension D>
//   class AmpereAxisymmetricPoles : public FieldSolver<D> {
//     using index_t = typename RealFieldND<D, 1>::size_type;
//     real_t coeff, coeff_x1;

//   public:
//     AmpereAxisymmetricPoles(
//         const Meshblock<D>& m_mblock_,
//         const real_t& coeff_,
//         const real_t& coeff_x1_)
//         : FieldSolver<D> {m_mblock_},
//           coeff(coeff_),
//           coeff_x1(coeff_x1_)
//         {}
//     Inline void operator()(const index_t) const;
//   };

//   template <>
//   Inline void AmpereAxisymmetricPoles<TWO_D>::operator()(const index_t i) const {

//     index_t j_min {N_GHOSTS};
//     index_t j_max {m_mblock.m_resolution[1] + N_GHOSTS - 1};

//     real_t x1 {m_mblock.convert_iTOx1(i)};
//     real_t x2_min {m_mblock.convert_jTOx2(j_min)};
//     real_t x2_max {m_mblock.convert_jTOx2(j_max)};

//     real_t dx1 {(m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])};
//     real_t dx2 {(m_mblock.m_extent[3] - m_mblock.m_extent[2]) / static_cast<real_t>(m_mblock.m_resolution[1])};

//     real_t inv_polar_area_iPj {ONE / m_mblock.m_coord_system->polar_area(x1 + 0.5 * dx1, dx2)};
//     real_t h3_min_iPjP {m_mblock.m_coord_system->hx3(x1 + 0.5 * dx1, x2_min + 0.5 * dx2)};
//     real_t h3_max_iPjP {m_mblock.m_coord_system->hx3(x1 + 0.5 * dx1, x2_max + 0.5 * dx2)};

//     real_t inv_sqrt_detH_ijP {ONE / m_mblock.m_coord_system->sqrt_det_h(x1, x2_min + 0.5 * dx2)};
//     real_t h3_min_iMjP {m_mblock.m_coord_system->hx3(x1 - 0.5 * dx1, x2_min + 0.5 * dx2)};

//     // theta = 0
//     m_mblock.em_fields(i, j_min, fld::ex1) += inv_polar_area_iPj * coeff * (h3_min_iPjP * m_mblock.em_fields(i, j_min, fld::bx3));
//     // theta = pi
//     m_mblock.em_fields(i, j_max + 1, fld::ex1) -= inv_polar_area_iPj * coeff * (h3_max_iPjP * m_mblock.em_fields(i, j_max, fld::bx3));

//     // j = jmin + 1/2
//     m_mblock.em_fields(i, j_min, fld::ex2) += inv_sqrt_detH_ijP * coeff_x1 * (h3_min_iMjP * m_mblock.em_fields(i - 1, j_min, fld::bx3) - h3_min_iPjP * m_mblock.em_fields(i, j_min, fld::bx3));
//   }

// } // namespace ntt

// template class ntt::AmpereAxisymmetricPoles<ntt::TWO_D>;

// #endif
