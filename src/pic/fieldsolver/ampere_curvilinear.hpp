// #ifndef PIC_FIELDSOLVER_AMPERE_CURVILINEAR_H
// #define PIC_FIELDSOLVER_AMPERE_CURVILINEAR_H

// #include "global.h"
// #include "meshblock.h"
// #include "fieldsolver.h"

// namespace ntt {

//   // * * * * Curvilinear Ampere's law * * * * * * * * * * * * * * * *
//   template <Dimension D>
//   class AmpereCurvilinear : public FieldSolver<D> {
//     using index_t = typename RealFieldND<D, 1>::size_type;
//     real_t coeff_x1, coeff_x2, coeff_x3;

//   public:
//     AmpereCurvilinear(
//         const Meshblock<D>& m_mblock_,
//         const real_t& coeff_x1_,
//         const real_t& coeff_x2_,
//         const real_t& coeff_x3_)
//         : FieldSolver<D> {m_mblock_},
//           coeff_x1(coeff_x1_),
//           coeff_x2(coeff_x2_),
//           coeff_x3(coeff_x3_)
//         {}
//     Inline void operator()(const index_t) const;
//     Inline void operator()(const index_t, const index_t) const;
//     Inline void operator()(const index_t, const index_t, const index_t) const;
//   };

//   template <>
//   Inline void AmpereCurvilinear<ONE_D>::operator()(const index_t i) const {
//     real_t x1 {
//       m_mblock.convert_iTOx1(i)
//     };
//     real_t dx1 {
//       (m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])
//     };

//     real_t inv_sqrt_detH_i {
//       ONE / m_mblock.m_coord_system->sqrt_det_h(x1)
//     };
//     real_t h2_iP {
//       m_mblock.m_coord_system->hx2(x1 + 0.5 * dx1)
//     };
//     real_t h2_iM {
//       m_mblock.m_coord_system->hx2(x1 - 0.5 * dx1)
//     };
//     real_t h3_iP {
//       m_mblock.m_coord_system->hx3(x1 + 0.5 * dx1)
//     };
//     real_t h3_iM {
//       m_mblock.m_coord_system->hx3(x1 - 0.5 * dx1)
//     };

//     m_mblock.em_fields(i, fld::ex2) += inv_sqrt_detH_i * coeff_x1 * (
//                                           h3_iM * m_mblock.em_fields(i - 1, fld::bx3) - h3_iP * m_mblock.em_fields(i, fld::bx3)
//                                         );
//     m_mblock.em_fields(i, fld::ex3) += inv_sqrt_detH_i * coeff_x1 * (
//                                           h2_iP * m_mblock.em_fields(i, fld::bx2) - h2_iM * m_mblock.em_fields(i - 1, fld::bx2)
//                                         );
//   }

//   template <>
//   Inline void AmpereCurvilinear<TWO_D>::operator()(const index_t i, const index_t j) const {
//     real_t x1 {
//       m_mblock.convert_iTOx1(i)
//     };
//     real_t x2 {
//       m_mblock.convert_jTOx2(j)
//     };
//     real_t dx1 {
//       (m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])
//     };
//     real_t dx2 {
//       (m_mblock.m_extent[3] - m_mblock.m_extent[2]) / static_cast<real_t>(m_mblock.m_resolution[1])
//     };

//     real_t inv_sqrt_detH_ij {
//       ONE / m_mblock.m_coord_system->sqrt_det_h(x1, x2)
//     };
//     real_t inv_sqrt_detH_iPj {
//       ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx1, x2)
//     };
//     real_t inv_sqrt_detH_ijP {
//       ONE / m_mblock.m_coord_system->sqrt_det_h(x1, x2 + 0.5 * dx2)
//     };
//     real_t h1_ijM {
//       m_mblock.m_coord_system->hx1(x1, x2 - 0.5 * dx2)
//     };
//     real_t h1_ijP {
//       m_mblock.m_coord_system->hx1(x1, x2 + 0.5 * dx2)
//     };
//     real_t h2_iPj {
//       m_mblock.m_coord_system->hx2(x1 + 0.5 * dx1, x2)
//     };
//     real_t h2_iMj {
//       m_mblock.m_coord_system->hx2(x1 - 0.5 * dx1, x2)
//     };
//     real_t h3_iMjP {
//       m_mblock.m_coord_system->hx3(x1 - 0.5 * dx1, x2 + 0.5 * dx2)
//     };
//     real_t h3_iPjM {
//       m_mblock.m_coord_system->hx3(x1 + 0.5 * dx1, x2 - 0.5 * dx2)
//     };
//     real_t h3_iPjP {
//       m_mblock.m_coord_system->hx3(x1 + 0.5 * dx1, x2 + 0.5 * dx2)
//     };

//     m_mblock.em_fields(i, j, fld::ex1) += inv_sqrt_detH_iPj * coeff_x2 * (
//                                               h3_iPjP * m_mblock.em_fields(i, j, fld::bx3) - h3_iPjM * m_mblock.em_fields(i, j - 1, fld::bx3)
//                                             );
//     m_mblock.em_fields(i, j, fld::ex2) += inv_sqrt_detH_ijP * coeff_x1 * (
//                                               h3_iMjP * m_mblock.em_fields(i - 1, j, fld::bx3) - h3_iPjP * m_mblock.em_fields(i, j, fld::bx3)
//                                             );
//     m_mblock.em_fields(i, j, fld::ex3) += inv_sqrt_detH_ij * (coeff_x2 * (
//                                               h1_ijM * m_mblock.em_fields(i, j - 1, fld::bx1) - h1_ijP * m_mblock.em_fields(i, j, fld::bx1)
//                                             ) + coeff_x1 * (
//                                               h2_iPj * m_mblock.em_fields(i, j, fld::bx2) - h2_iMj * m_mblock.em_fields(i - 1, j, fld::bx2)
//                                             )
//                                           );
//   }

//   template <>
//   Inline void AmpereCurvilinear<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
//     UNUSED(i);
//     UNUSED(j);
//     UNUSED(k);
//     throw std::logic_error("# 3d curvilinear ampere NOT IMPLEMENTED.");
//     // real_t x1 {m_mblock.convert_iTOx1(i)};
//     // real_t x2 {m_mblock.convert_jTOx2(j)};
//     // real_t x3 {m_mblock.convert_kTOx3(k)};
//     // real_t dx1 {(m_mblock.m_extent[1] - m_mblock.m_extent[0]) / static_cast<real_t>(m_mblock.m_resolution[0])};
//     // real_t dx2 {(m_mblock.m_extent[3] - m_mblock.m_extent[2]) / static_cast<real_t>(m_mblock.m_resolution[1])};
//     // real_t dx3 {(m_mblock.m_extent[5] - m_mblock.m_extent[4]) / static_cast<real_t>(m_mblock.m_resolution[2])};
//     // real_t inv_sqrt_detH_iPjk {ONE / m_mblock.m_coord_system->sqrt_det_h(x1 + 0.5 * dx1, x2, x3)};
//     // real_t inv_sqrt_detH_ijPk {ONE / m_mblock.m_coord_system->sqrt_det_h(x1, x2 + 0.5 * dx2, x3)};
//     // real_t inv_sqrt_detH_ijkP {ONE / m_mblock.m_coord_system->sqrt_det_h(x1, x2, x3 + 0.5 * dx3)};
//     //
//     // m_mblock.em_fields(i, j, k, fld::ex1) += inv_sqrt_detH_iPjk * (coeff_x3 * (m_mblock.em_fields(i, j, k - 1, fld::bx2) - m_mblock.em_fields(i, j, k, fld::bx2)) + coeff_x2 * (m_mblock.em_fields(i, j, k, fld::bx3) - m_mblock.em_fields(i, j - 1, k, fld::bx3)));
//     // m_mblock.em_fields(i, j, k, fld::ex2) += inv_sqrt_detH_ijPk * (coeff_x1 * (m_mblock.em_fields(i - 1, j, k, fld::bx3) - m_mblock.em_fields(i, j, k, fld::bx3)) + coeff_x3 * (m_mblock.em_fields(i, j, k, fld::bx1) - m_mblock.em_fields(i, j, k - 1, fld::bx1)));
//     // m_mblock.em_fields(i, j, k, fld::ex3) += inv_sqrt_detH_ijkP * (coeff_x2 * (m_mblock.em_fields(i, j - 1, k, fld::bx1) - m_mblock.em_fields(i, j, k, fld::bx1)) + coeff_x1 * (m_mblock.em_fields(i, j, k, fld::bx2) - m_mblock.em_fields(i - 1, j, k, fld::bx2)));
//   }

// } // namespace ntt

// template class ntt::AmpereCurvilinear<ntt::ONE_D>;
// template class ntt::AmpereCurvilinear<ntt::TWO_D>;
// template class ntt::AmpereCurvilinear<ntt::THREE_D>;

// #endif
