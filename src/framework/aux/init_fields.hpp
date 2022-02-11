#ifndef GRPIC_INIT_FIELDS
#define GRPIC_INIT_FIELDS

#include "global.h"
#include "fields.h"
#include "meshblock.h"

#include <stdexcept>

namespace ntt {

  /**
   * Computes D and B from Aphi, Ar, At 
   *
   * @tparam D Dimension.
   */

  template <Dimension D>
    class init_fields_potential {
    using index_t = typename RealFieldND<D, 6>::size_type;
    Meshblock<D, SimulationType::GRPIC> m_mblock;
    real_t m_eps;
    real_t (*m_aphi)(const Meshblock<D, SimulationType::GRPIC>&, const coord_t<D>& x);
    real_t (*m_ar)(const Meshblock<D, SimulationType::GRPIC>&, const coord_t<D>& x);
    real_t (*m_at)(const Meshblock<D, SimulationType::GRPIC>&, const coord_t<D>& x);

  public:
    init_fields_potential(const Meshblock<D, SimulationType::GRPIC>& mblock,
    real_t eps,
    real_t (*aphi)(const Meshblock<D, SimulationType::GRPIC>&, const coord_t<D>& x),
    real_t (*ar)(const Meshblock<D, SimulationType::GRPIC>&, const coord_t<D>& x),
    real_t (*at)(const Meshblock<D, SimulationType::GRPIC>&, const coord_t<D>& x))
    : m_mblock(mblock), m_eps(eps), m_aphi(aphi), m_ar(ar), m_at(at) {}

    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void init_fields_potential<Dimension::TWO_D>::operator()(const index_t i, const index_t j) const {
    real_t i_ {static_cast<real_t>(i - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(j - N_GHOSTS)};
    coord_t<Dimension::TWO_D> x0m, x0p;

    real_t inv_sqrt_detH_ij   {ONE / m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t inv_sqrt_detH_ijP  {ONE / m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t inv_sqrt_detH_iPj  {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_})};
    real_t inv_sqrt_detH_iPjP {ONE / m_mblock.metric.sqrt_det_h({i_ + HALF, j_ + HALF})};
    real_t sqrt_detH_ij  {m_mblock.metric.sqrt_det_h({i_, j_})};
    real_t sqrt_detH_ijP {m_mblock.metric.sqrt_det_h({i_, j_ + HALF})};
    real_t alpha_ij   {m_mblock.metric.alpha({i_, j_})};
    real_t alpha_ijP  {m_mblock.metric.alpha({i_, j_ + HALF})};
    real_t alpha_iPj  {m_mblock.metric.alpha({i_ + HALF, j_})};
    real_t betar_ij   {m_mblock.metric.betar({i_, j_})};
    real_t betar_ijP  {m_mblock.metric.betar({i_, j_ + HALF})};
    real_t h11_inv_iPj {m_mblock.metric.h_11_inv({i_ + HALF, j_})};
    real_t h13_inv_iPj {m_mblock.metric.h_13_inv({i_ + HALF, j_})};
    real_t h22_inv_ijP {m_mblock.metric.h_22_inv({i_, j_ + HALF})};
    real_t h33_inv_ij  {m_mblock.metric.h_33_inv({i_, j_})};
    real_t h13_inv_ij  {m_mblock.metric.h_13_inv({i_, j_})};

    x0m[0] = i_, x0m[1] = j_ + HALF - m_eps;
    x0p[0] = i_, x0p[1] = j_ + HALF + m_eps;
    real_t Bru     =   (m_aphi(m_mblock, x0p) -m_aphi(m_mblock, x0m)) * inv_sqrt_detH_ijP / m_eps;
    real_t Ethd    =   (m_at(m_mblock, x0p) - m_at(m_mblock, x0m)) * inv_sqrt_detH_ijP / m_eps;
    real_t Bph_aux = - (m_ar(m_mblock, x0p) - m_ar(m_mblock, x0m)) * inv_sqrt_detH_ijP / m_eps;

    x0m[0] = i_ + HALF - m_eps, x0m[1] = j_;
    x0p[0] = i_ + HALF + m_eps, x0p[1] = j_;
    real_t Bthu    = - (m_aphi(m_mblock, x0p) -m_aphi(m_mblock, x0m)) * inv_sqrt_detH_iPj / m_eps;
    real_t Erd     =   (m_at(m_mblock, x0p) - m_at(m_mblock, x0m)) * inv_sqrt_detH_iPj / m_eps;

    x0m[0] = i_ + HALF, x0m[1] = j_ + HALF - m_eps;
    x0p[0] = i_ + HALF, x0p[1] = j_ + HALF + m_eps;
    real_t Bphu    = - (m_ar(m_mblock, x0p) - m_ar(m_mblock, x0m)) * inv_sqrt_detH_iPjP / m_eps;

    x0m[0] = i_ - m_eps, x0m[1] = j_;
    x0p[0] = i_ + m_eps, x0p[1] = j_;
    real_t Bth_aux = - (m_aphi(m_mblock, x0p) -m_aphi(m_mblock, x0m)) * inv_sqrt_detH_ij / m_eps;
 
    real_t Drd  {Erd / alpha_iPj};
    real_t Dthd {Ethd / alpha_ijP + sqrt_detH_ijP * betar_ijP * Bph_aux / alpha_ijP};
    real_t Dphd {- sqrt_detH_ij * betar_ij * Bth_aux / alpha_ij};

    // Contravariant D to covariant D
    real_t Dru  {h11_inv_iPj * Drd + h13_inv_iPj * Dphd};
    real_t Dthu {h22_inv_ijP * Dthd};
    real_t Dphu {h33_inv_ij * Dphd + h13_inv_ij * Drd};

    m_mblock.em0(i, j, em::bx1) = Bru;
    m_mblock.em0(i, j, em::bx2) = Bthu;
    m_mblock.em0(i, j, em::bx3) = Bphu;
    m_mblock.em0(i, j, em::ex1) = Dru;
    m_mblock.em0(i, j, em::ex2) = Dthu;
    m_mblock.em0(i, j, em::ex3) = Dphu;
    m_mblock.aux(i, j, em::ex1) = Erd;
    m_mblock.aux(i, j, em::ex2) = Ethd;
    m_mblock.aux(i, j, em::ex3) = ZERO;
  }

  template <>
  Inline void
  init_fields_potential<Dimension::THREE_D>::operator()(const index_t, const index_t, const index_t) const {
    // No initialization with Aphi in 3D
  }

} // namespace ntt

#endif