#ifndef UTILS_DIGITAL_FILTER_H
#define UTILS_DIGITAL_FILTER_H

#include "global.h"

namespace ntt {
  /**
   * @brief Digital filtering routine.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class DigitalFilter {
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
            const unsigned short&  npasses)
      : m_cur(cur), m_cur_b(cur_b), m_npasses(npasses) {}

    void apply() const {
      for (unsigned short i = 0; i < m_npasses; ++i) {
        // backup the field
        Kokkos::deep_copy(cur_b, cur);
        // filter the current
        filter();
        // exchange ghost zones and/or apply boundary conditions
      }
    }
  };
} // namespace ntt

#endif