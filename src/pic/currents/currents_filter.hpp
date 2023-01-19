#ifndef PIC_CURRENTS_FILTER_H
#define PIC_CURRENTS_FILTER_H

#include "wrapper.h"

#include "pic.h"

namespace ntt {
  /**
   * @brief Spatial filtering of all the deposited currents.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class CurrentsFilter_kernel {
    Meshblock<D, PICEngine> m_mblock;
    tuple_t<std::size_t, D> m_size;

  public:
    /**
     * @brief Constructor.
     * @param mblock Meshblock object.
     */
    CurrentsFilter_kernel(const Meshblock<D, PICEngine>& mblock) : m_mblock { mblock } {
      for (short d = 0; d < (short)D; ++d) {
        m_size[d] = m_mblock.Ni(d);
      }
    }

    /**
     * @brief 1D implementation of the algorithm.
     * @param i1 index.
     */
    Inline void operator()(index_t) const;
    /**
     * @brief 2D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t, index_t) const;
    /**
     * @brief 3D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     * @param i3 index.
     */
    Inline void operator()(index_t, index_t, index_t) const;
  };

#ifdef MINKOWSKI_METRIC
  template <>
  Inline void CurrentsFilter_kernel<Dim1>::operator()(index_t i) const {
    for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
      m_mblock.cur(i, comp)
        = INV_2 * m_mblock.buff(i, comp)
          + INV_4 * (m_mblock.buff(i - 1, comp) + m_mblock.buff(i + 1, comp));
    }
  }

  template <>
  Inline void CurrentsFilter_kernel<Dim2>::operator()(index_t i, index_t j) const {
    for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
      m_mblock.cur(i, j, comp)
        = INV_4 * m_mblock.buff(i, j, comp)
          + INV_8
              * (m_mblock.buff(i - 1, j, comp) + m_mblock.buff(i + 1, j, comp)
                 + m_mblock.buff(i, j - 1, comp) + m_mblock.buff(i, j + 1, comp))
          + INV_16
              * (m_mblock.buff(i - 1, j - 1, comp) + m_mblock.buff(i + 1, j + 1, comp)
                 + m_mblock.buff(i - 1, j + 1, comp) + m_mblock.buff(i + 1, j - 1, comp));
    }
  }

  template <>
  Inline void CurrentsFilter_kernel<Dim3>::operator()(index_t i, index_t j, index_t k) const {
    for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
      m_mblock.cur(i, j, k, comp)
        = INV_8 * m_mblock.buff(i, j, k, comp)
          + INV_16
              * (m_mblock.buff(i - 1, j, k, comp) + m_mblock.buff(i + 1, j, k, comp)
                 + m_mblock.buff(i, j - 1, k, comp) + m_mblock.buff(i, j + 1, k, comp)
                 + m_mblock.buff(i, j, k - 1, comp) + m_mblock.buff(i, j, k + 1, comp))
          + INV_32
              * (m_mblock.buff(i - 1, j - 1, k, comp) + m_mblock.buff(i + 1, j + 1, k, comp)
                 + m_mblock.buff(i - 1, j + 1, k, comp) + m_mblock.buff(i + 1, j - 1, k, comp)
                 + m_mblock.buff(i, j - 1, k - 1, comp) + m_mblock.buff(i, j + 1, k + 1, comp)
                 + m_mblock.buff(i, j, k - 1, comp) + m_mblock.buff(i, j, k + 1, comp)
                 + m_mblock.buff(i - 1, j, k - 1, comp) + m_mblock.buff(i + 1, j, k + 1, comp)
                 + m_mblock.buff(i - 1, j, k + 1, comp) + m_mblock.buff(i + 1, j, k - 1, comp))
          + INV_64
              * (m_mblock.buff(i - 1, j - 1, k - 1, comp)
                 + m_mblock.buff(i + 1, j + 1, k + 1, comp)
                 + m_mblock.buff(i - 1, j + 1, k + 1, comp)
                 + m_mblock.buff(i + 1, j - 1, k - 1, comp)
                 + m_mblock.buff(i - 1, j - 1, k + 1, comp)
                 + m_mblock.buff(i + 1, j + 1, k - 1, comp)
                 + m_mblock.buff(i - 1, j + 1, k - 1, comp)
                 + m_mblock.buff(i + 1, j - 1, k + 1, comp));
    }
  }
#else
  template <>
  Inline void CurrentsFilter_kernel<Dim1>::operator()(index_t) const {}

#  define FILTER_IN_I1(ARR, COMP, I, J)                                                       \
    INV_2*(ARR)((I), (J), (COMP))                                                             \
      + INV_4*((ARR)((I)-1, (J), (COMP)) + (ARR)((I) + 1, (J), (COMP)))

  template <>
  Inline void CurrentsFilter_kernel<Dim2>::operator()(index_t i, index_t j) const {
    const std::size_t j_min = N_GHOSTS, j_min_p1 = j_min + 1;
    const std::size_t j_max = m_size[1] + N_GHOSTS, j_max_m1 = j_max - 1;
    real_t            cur_ij, cur_ijp1, cur_ijm1;
    if (j == j_min) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j);
      cur_ijp1                     = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j + 1);
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx1) = INV_2 * cur_ij + INV_4 * cur_ijp1;

      m_mblock.cur(i, j, cur::jx3) = ZERO;

      /* ---------------------------------- theta --------------------------------- */
      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx2, i, j);
      cur_ijp1                     = FILTER_IN_I1(m_mblock.buff, cur::jx2, i, j + 1);
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx2) = INV_4 * (cur_ij + cur_ijp1);
    } else if (j == j_min_p1) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j);
      cur_ijp1                     = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j + 1);
      cur_ijm1                     = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j - 1);
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx1) = INV_2 * (cur_ij + cur_ijm1) + INV_4 * cur_ijp1;

      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx3, i, j);
      cur_ijp1                     = FILTER_IN_I1(m_mblock.buff, cur::jx3, i, j + 1);
      cur_ijm1                     = ZERO;
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx3) = INV_2 * (cur_ij + cur_ijm1) + INV_4 * cur_ijp1;

      /* ---------------------------------- theta --------------------------------- */
      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx2, i, j);
      cur_ijp1                     = FILTER_IN_I1(m_mblock.buff, cur::jx2, i, j + 1);
      cur_ijm1                     = FILTER_IN_I1(m_mblock.buff, cur::jx2, i, j - 1);
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx2) = INV_2 * cur_ij + INV_4 * (cur_ijm1 + cur_ijp1);
    } else if (j == j_max_m1) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j);
      cur_ijp1                     = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j + 1);
      cur_ijm1                     = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j - 1);
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx1) = INV_2 * (cur_ij + cur_ijp1) + INV_4 * cur_ijm1;

      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx3, i, j);
      cur_ijp1                     = ZERO;
      cur_ijm1                     = FILTER_IN_I1(m_mblock.buff, cur::jx3, i, j - 1);
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx3) = INV_2 * (cur_ij + cur_ijp1) + INV_4 * cur_ijm1;

      /* ---------------------------------- theta --------------------------------- */
      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx2, i, j);
      cur_ijm1                     = FILTER_IN_I1(m_mblock.buff, cur::jx2, i, j - 1);
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx2) = INV_4 * (cur_ij + cur_ijm1);
    } else if (j == j_max) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij                       = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j);
      cur_ijm1                     = FILTER_IN_I1(m_mblock.buff, cur::jx1, i, j - 1);
      // ... filter in theta
      m_mblock.cur(i, j, cur::jx1) = INV_2 * cur_ij + INV_4 * cur_ijm1;

      m_mblock.cur(i, j, cur::jx3) = ZERO;
    } else {
      for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
        m_mblock.cur(i, j, comp)
          = INV_4 * m_mblock.buff(i, j, comp)
            + INV_8
                * (m_mblock.buff(i - 1, j, comp) + m_mblock.buff(i + 1, j, comp)
                   + m_mblock.buff(i, j - 1, comp) + m_mblock.buff(i, j + 1, comp))
            + INV_16
                * (m_mblock.buff(i - 1, j - 1, comp) + m_mblock.buff(i + 1, j + 1, comp)
                   + m_mblock.buff(i - 1, j + 1, comp) + m_mblock.buff(i + 1, j - 1, comp));
      }
    }
  }

#  undef FILTER_IN_I1

  template <>
  Inline void CurrentsFilter_kernel<Dim3>::operator()(index_t, index_t, index_t) const {}

#endif

}    // namespace ntt

#endif    // PIC_CURRENTS_FILTER_H