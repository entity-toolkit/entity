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
      : m_cur(cur), m_cur_b(cur_b), m_mesh(mesh), m_npasses(npasses) {}

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
      m_cur(i, comp) = (real_t)(0.5) * m_cur_b(i, comp)
                       + (real_t)(0.25) * (m_cur_b(i - 1, comp) + m_cur_b(i + 1, comp));
    }
  }

  template <>
  Inline void DigitalFilter<Dim2>::operator()(index_t i, index_t j) const {
    for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
      m_cur(i, j, comp) = (real_t)(0.25) * m_cur_b(i, j, comp)
                          + (real_t)(0.125)
                              * (m_cur_b(i - 1, j, comp) + m_cur_b(i + 1, j, comp)
                                 + m_cur_b(i, j - 1, comp) + m_cur_b(i, j + 1, comp))
                          + (real_t)(0.0625)
                              * (m_cur_b(i - 1, j - 1, comp) + m_cur_b(i + 1, j + 1, comp)
                                 + m_cur_b(i - 1, j + 1, comp) + m_cur_b(i + 1, j - 1, comp));
    }
  }

  template <>
  Inline void DigitalFilter<Dim3>::operator()(index_t i, index_t j, index_t k) const {
    for (auto& comp : {cur::jx1, cur::jx2, cur::jx3}) {
      m_cur(i, j, k, comp)
        = (real_t)(0.125) * m_cur_b(i, j, k, comp)
          + (real_t)(0.0625)
              * (m_cur_b(i - 1, j, k, comp) + m_cur_b(i + 1, j, k, comp)
                 + m_cur_b(i, j - 1, k, comp) + m_cur_b(i, j + 1, k, comp)
                 + m_cur_b(i, j, k - 1, comp) + m_cur_b(i, j, k + 1, comp))
          + (real_t)(0.03125)
              * (m_cur_b(i - 1, j - 1, k, comp) + m_cur_b(i + 1, j + 1, k, comp)
                 + m_cur_b(i - 1, j + 1, k, comp) + m_cur_b(i + 1, j - 1, k, comp)
                 + m_cur_b(i, j - 1, k - 1, comp) + m_cur_b(i, j + 1, k + 1, comp)
                 + m_cur_b(i, j, k - 1, comp) + m_cur_b(i, j, k + 1, comp)
                 + m_cur_b(i - 1, j, k - 1, comp) + m_cur_b(i + 1, j, k + 1, comp)
                 + m_cur_b(i - 1, j, k + 1, comp) + m_cur_b(i + 1, j, k - 1, comp))
          + (real_t)(0.015625)
              * (m_cur_b(i - 1, j - 1, k - 1, comp) + m_cur_b(i + 1, j + 1, k + 1, comp)
                 + m_cur_b(i - 1, j + 1, k + 1, comp) + m_cur_b(i + 1, j - 1, k - 1, comp)
                 + m_cur_b(i - 1, j - 1, k + 1, comp) + m_cur_b(i + 1, j + 1, k - 1, comp)
                 + m_cur_b(i - 1, j + 1, k - 1, comp) + m_cur_b(i + 1, j - 1, k + 1, comp));
    }
  }
#endif

} // namespace ntt

#endif