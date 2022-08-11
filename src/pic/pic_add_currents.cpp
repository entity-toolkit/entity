#include "global.h"
#include "pic.h"

#if (METRIC == MINKOWSKI_METRIC)
#  include "pic_add_currents_minkowski.hpp"
#else
#  include "pic_add_currents_curvilinear.hpp"
#endif

namespace ntt {
  /**
   * @brief add currents to the E-field
   *
   */
  template <Dimension D>
  void PIC<D>::addCurrentsSubstep(const real_t&) {
    const auto dt {this->m_mblock.timestep()};
    const auto rho0 {this->m_sim_params.larmor0()};
    const auto de0 {this->m_sim_params.skindepth0()};
    const auto n0 {this->m_sim_params.ppc0()};
    const auto dx {(this->m_mblock.metric.x1_max - this->m_mblock.metric.x1_min)
                   / this->m_mblock.metric.nx1};
    const auto coeff {-dt * rho0 / (n0 * SQR(de0))};
#if (METRIC == MINKOWSKI_METRIC)
    Kokkos::parallel_for("add_currents",
                         this->m_mblock.rangeActiveCells(),
                         AddCurrentsMinkowski<D>(this->m_mblock, coeff / CUBE(dx)));
#else
    Kokkos::parallel_for("add_currents",
                         this->m_mblock.rangeActiveCells(),
                         AddCurrentsCurvilinear<D>(this->m_mblock, coeff));
#endif
  }
} // namespace ntt

template class ntt::PIC<ntt::Dimension::ONE_D>;
template class ntt::PIC<ntt::Dimension::TWO_D>;
template class ntt::PIC<ntt::Dimension::THREE_D>;
