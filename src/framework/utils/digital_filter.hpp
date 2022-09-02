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
    Mesh<D>         m_mesh;
    ndfield_t<D, 3> m_cur;
    ndfield_t<D, 3> m_cur_b;
    unsigned short  m_npasses;

  public:
    /**
     * @brief Constructor.
     * @param cur Current field.
     * @param cur0 Backup current field.
     * @param npasses Number of filter passes.
     */
    Digital(const ndfield_t<D, 3>& cur,
            const ndfiend_t<D, 3>& cur_b,
            const Mesh<D>&         mesh,
            const unsigned short&  npasses)
      : m_cur(cur), m_cur_b(cur_b), m_mesh(mesh), m_npasses(npasses) {}

    void apply() const {
      for (unsigned short i = 0; i < m_npasses; ++i) {
        // backup the field
        Kokkos::deep_copy(cur_b, cur);
        // filter the current
        filterPass();
        // exchange ghost zones and/or apply boundary conditions
      }
    }

  private:
    void filterPass() const {}
    void synchronizeGhostZones() const;
  };

  template <>
  void DigitalFilter<Dim1>::synchronizeGhostZones() const {
    auto ni {m_mesh.Ni1()};
    Kokkos::parallel_for(
      "1d_gh_x1m",
      m_mesh.rangeCells({CellLayer::minGhostLayer}),
      Lambda(index_t i, index_t j) {
        m_cur.cur(i, cur::jx1) += m_cur.cur(i + ni, cur::jx1);
        m_cur.cur(i, cur::jx2) += m_cur.cur(i + ni, cur::jx2);
        m_cur.cur(i, cur::jx3) += m_cur.cur(i + ni, cur::jx3);
      });
    Kokkos::parallel_for(
      "1d_gh_x1p",
      m_mesh.rangeCells({CellLayer::maxGhostLayer}),
      Lambda(index_t i, index_t j) {
        m_cur.cur(i, cur::jx1) += m_cur.cur(i - ni, cur::jx1);
        m_cur.cur(i, cur::jx2) += m_cur.cur(i - ni, cur::jx2);
        m_cur.cur(i, cur::jx3) += m_cur.cur(i - ni, cur::jx3);
      });
  }

  template <>
  void DigitalFilter<Dim2>::synchronizeGhostZones() const {}

  template <>
  void DigitalFilter<Dim3>::synchronizeGhostZones() const {}

#endif