#ifndef PIC_CURRENTS_BC_H
#define PIC_CURRENTS_BC_H

#include "wrapper.h"

#include "field_macros.h"
#include "pic.h"

#include PGEN_HEADER

namespace ntt {
  // /**
  //  * @brief Algorithms for PIC current boundary conditions.
  //  * @tparam D Dimension.
  //  */
  // template <Dimension D>
  // class AbsorbCurrents_kernel {
  //   Meshblock<D, PICEngine>        m_mblock;
  //   ProblemGenerator<D, PICEngine> m_pgen;
  //   real_t                         m_rabsorb;
  //   real_t                         m_rmax;

  // public:
  //   /**
  //    * @brief Constructor.
  //    * @param mblock Meshblock.
  //    * @param pgen Problem generator.
  //    * @param rabsorb Absorbing radius.
  //    * @param rmax Maximum radius.
  //    */
  //   AbsorbCurrents_kernel(const Meshblock<D, PICEngine>&        mblock,
  //                         const ProblemGenerator<D, PICEngine>& pgen,
  //                         real_t                                r_absorb,
  //                         real_t                                r_max) :
  //     m_mblock { mblock },
  //     m_pgen { pgen },
  //     m_rabsorb { r_absorb },
  //     m_rmax { r_max } {}

  //   /**
  //    * @brief 2D implementation of the algorithm.
  //    * @param i1 index.
  //    * @param i2 index.
  //    */
  //   Inline void operator()(index_t, index_t) const;
  // };

  // template <>
  // Inline void AbsorbCurrents_kernel<Dim2>::operator()(index_t i, index_t j) const {
  //   const real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
  //   const real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

  //   const real_t i1
  //   // vec_t<PrtlCoordD> rth_;
  //   // m_mblock.metric.x_Code2Sph({ i_, j_, ZERO }, rth_);
  //   real_t delta_r1 { (rth_[0] - m_rabsorb) / (m_rmax - m_rabsorb) };
  //   real_t sigma_r1 { HEAVISIDE(delta_r1) * delta_r1 * delta_r1 * delta_r1 };

  //   JX1(i, j) = (ONE - sigma_r1) * JX1(i, j);
  //   JX2(i, j) = (ONE - sigma_r1) * JX2(i, j);
  //   JX3(i, j) = (ONE - sigma_r1) * JX3(i, j);
  // }
} // namespace ntt

#endif
