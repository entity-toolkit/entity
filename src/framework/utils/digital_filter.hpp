#ifndef UTILS_DIGITAL_FILTER_H
#define UTILS_DIGITAL_FILTER_H

#include "global.h"
#include "meshblock.h"

namespace ntt {
  /**
   * @brief Digital filtering routine.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class DigitalFilter {
    ndfield_t<D, 3> m_cur;
    ndfield_t<D, 3> m_cur_b;
    Mesh<D>         m_mesh;
    unsigned short  m_npasses;
    tuple_t<std::size_t, D> m_size;

  public:
    /**
     * @brief Constructor.
     * @param cur Current field.
     * @param cur0 Backup current field.
     * @param npasses Number of filter passes.
     */
    DigitalFilter(const ndfield_t<D, 3>& cur,
                  const ndfield_t<D, 3>& cur_b,
                  const Mesh<D>&         mesh,
                  const unsigned short&  npasses)
      : m_cur(cur), m_cur_b(cur_b), m_mesh(mesh), m_npasses(npasses) {
      for (short d = 0; d < (short)D; ++d) {
        m_size[d] = m_mesh.Ni(d);
      }
    }

    void apply() {
      for (unsigned short i = 0; i < m_npasses; ++i) {
        synchronizeGhostZones();
        Kokkos::deep_copy(m_cur_b, m_cur);
        filterPass();
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

  private:
    void filterPass() {
      auto range {m_mesh.rangeActiveCells()};
      Kokkos::parallel_for("filter_pass", range, *this);
    }
    void synchronizeGhostZones() const;
  };

#ifdef MINKOWSKI_METRIC
  template <>
  Inline void DigitalFilter<Dim1>::operator()(index_t i) const {
    for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
      m_cur(i, comp)
        = INV_2 * m_cur_b(i, comp) + INV_4 * (m_cur_b(i - 1, comp) + m_cur_b(i + 1, comp));
    }
  }

  template <>
  Inline void DigitalFilter<Dim2>::operator()(index_t i, index_t j) const {
    for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
      m_cur(i, j, comp) = INV_4 * m_cur_b(i, j, comp)
                          + INV_8
                              * (m_cur_b(i - 1, j, comp) + m_cur_b(i + 1, j, comp)
                                 + m_cur_b(i, j - 1, comp) + m_cur_b(i, j + 1, comp))
                          + INV_16
                              * (m_cur_b(i - 1, j - 1, comp) + m_cur_b(i + 1, j + 1, comp)
                                 + m_cur_b(i - 1, j + 1, comp) + m_cur_b(i + 1, j - 1, comp));
    }
  }

  template <>
  Inline void DigitalFilter<Dim3>::operator()(index_t i, index_t j, index_t k) const {
    for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
      m_cur(i, j, k, comp)
        = INV_8 * m_cur_b(i, j, k, comp)
          + INV_16
              * (m_cur_b(i - 1, j, k, comp) + m_cur_b(i + 1, j, k, comp)
                 + m_cur_b(i, j - 1, k, comp) + m_cur_b(i, j + 1, k, comp)
                 + m_cur_b(i, j, k - 1, comp) + m_cur_b(i, j, k + 1, comp))
          + INV_32
              * (m_cur_b(i - 1, j - 1, k, comp) + m_cur_b(i + 1, j + 1, k, comp)
                 + m_cur_b(i - 1, j + 1, k, comp) + m_cur_b(i + 1, j - 1, k, comp)
                 + m_cur_b(i, j - 1, k - 1, comp) + m_cur_b(i, j + 1, k + 1, comp)
                 + m_cur_b(i, j, k - 1, comp) + m_cur_b(i, j, k + 1, comp)
                 + m_cur_b(i - 1, j, k - 1, comp) + m_cur_b(i + 1, j, k + 1, comp)
                 + m_cur_b(i - 1, j, k + 1, comp) + m_cur_b(i + 1, j, k - 1, comp))
          + INV_64
              * (m_cur_b(i - 1, j - 1, k - 1, comp) + m_cur_b(i + 1, j + 1, k + 1, comp)
                 + m_cur_b(i - 1, j + 1, k + 1, comp) + m_cur_b(i + 1, j - 1, k - 1, comp)
                 + m_cur_b(i - 1, j - 1, k + 1, comp) + m_cur_b(i + 1, j + 1, k - 1, comp)
                 + m_cur_b(i - 1, j + 1, k - 1, comp) + m_cur_b(i + 1, j - 1, k + 1, comp));
    }
  }
#else
  template <>
  Inline void DigitalFilter<Dim1>::operator()(index_t) const {}

#  define FILTER_IN_I1(ARR, COMP, I, J)                                                       \
    (ARR)((I), (J), (COMP)) = static_cast<real_t>(0.5) * (ARR)((I), (J), (COMP))              \
                              + static_cast<real_t>(0.25)                                     \
                                  * ((ARR)((I)-1, (J), (COMP)) + (ARR)((I) + 1, (J), (COMP)))

  template <>
  Inline void DigitalFilter<Dim2>::operator()(index_t i, index_t j) const {
    const std::size_t j_min = N_GHOSTS, j_min_p1 = j_min + 1;
    const std::size_t j_max = m_size[1] - N_GHOSTS - 1, j_max_m1 = j_max - 1;
    real_t    cur_ij, cur_ijp1, cur_ijm1;
    if (j == j_min) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx1, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx1, i, j + 1);
      // ... filter in theta
      m_cur(i, j, cur::jx1) = INV_2 * cur_ij + INV_4 * cur_ijp1;

      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx3, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx3, i, j + 1);
      // ... filter in theta
      m_cur(i, j, cur::jx3) = INV_2 * cur_ij + INV_4 * cur_ijp1;

      /* ---------------------------------- theta --------------------------------- */
      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx2, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx2, i, j + 1);
      // ... filter in theta
      m_cur(i, j, cur::jx2) = INV_4 * cur_ij + INV_4 * cur_ijp1;
    } else if (j == j_min_p1) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx1, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx1, i, j + 1);
      cur_ijm1 = FILTER_IN_I1(m_cur_b, cur::jx1, i, j - 1);
      // ... filter in theta
      m_cur(i, j, cur::jx1) = INV_2 * (cur_ij + cur_ijm1) + INV_4 * cur_ijp1;

      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx3, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx3, i, j + 1);
      cur_ijm1 = FILTER_IN_I1(m_cur_b, cur::jx3, i, j - 1);
      // ... filter in theta
      m_cur(i, j, cur::jx3) = INV_2 * (cur_ij + cur_ijm1) + INV_4 * cur_ijp1;
    } else if (j == j_max_m1) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx1, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx1, i, j + 1);
      cur_ijm1 = FILTER_IN_I1(m_cur_b, cur::jx1, i, j - 1);
      // ... filter in theta
      m_cur(i, j, cur::jx1) = INV_2 * (cur_ij + cur_ijp1) + INV_4 * cur_ijm1;

      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx3, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx3, i, j + 1);
      cur_ijm1 = FILTER_IN_I1(m_cur_b, cur::jx3, i, j - 1);
      // ... filter in theta
      m_cur(i, j, cur::jx3) = INV_2 * (cur_ij + cur_ijp1) + INV_4 * cur_ijm1;

      /* ---------------------------------- theta --------------------------------- */
      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx2, i, j);
      cur_ijm1 = FILTER_IN_I1(m_cur_b, cur::jx2, i, j - 1);
      // ... filter in theta
      m_cur(i, j, cur::jx2) = INV_4 * cur_ij + INV_4 * cur_ijm1;
    } else if (j == j_max) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx1, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx1, i, j + 1);
      cur_ijm1 = FILTER_IN_I1(m_cur_b, cur::jx1, i, j - 1);
      // ... filter in theta
      m_cur(i, j, cur::jx1)
        = INV_2 * m_cur_b(i, j, cur::jx1) + INV_4 * m_cur_b(i, j - 1, cur::jx1);

      // ... filter in r
      cur_ij   = FILTER_IN_I1(m_cur_b, cur::jx3, i, j);
      cur_ijp1 = FILTER_IN_I1(m_cur_b, cur::jx3, i, j + 1);
      cur_ijm1 = FILTER_IN_I1(m_cur_b, cur::jx3, i, j - 1);
      // ... filter in theta
      m_cur(i, j, cur::jx3)
        = INV_2 * m_cur_b(i, j, cur::jx3) + INV_4 * m_cur_b(i, j - 1, cur::jx3);
    } else {
      for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
        m_cur(i, j, comp)
          = INV_4 * m_cur_b(i, j, comp)
            + INV_8
                * (m_cur_b(i - 1, j, comp) + m_cur_b(i + 1, j, comp) + m_cur_b(i, j - 1, comp)
                   + m_cur_b(i, j + 1, comp))
            + INV_16
                * (m_cur_b(i - 1, j - 1, comp) + m_cur_b(i + 1, j + 1, comp)
                   + m_cur_b(i - 1, j + 1, comp) + m_cur_b(i + 1, j - 1, comp));
      }
    }
  }

#  undef FILTER_IN_I1

  template <>
  Inline void DigitalFilter<Dim3>::operator()(index_t, index_t, index_t) const {}

#endif

} // namespace ntt

#endif