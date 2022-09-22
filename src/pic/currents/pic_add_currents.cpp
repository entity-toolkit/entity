#include "global.h"
#include "pic.h"

#ifdef MINKOWSKI_METRIC
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
    const auto dt    = this->m_mblock.timestep();
    const auto rho0  = this->m_sim_params.larmor0();
    const auto de0   = this->m_sim_params.skindepth0();
    const auto n0    = this->m_sim_params.ppc0();
    const auto coeff = -dt * rho0 / (n0 * SQR(de0));
#ifdef MINKOWSKI_METRIC
    const auto dx = (this->m_mblock.metric.x1_max - this->m_mblock.metric.x1_min)
                    / this->m_mblock.metric.nx1;
    Kokkos::parallel_for("add_currents",
                         this->m_mblock.rangeActiveCells(),
                         AddCurrentsMinkowski<D>(this->m_mblock, coeff / CUBE(dx)));
#else
    tuple_t<tuple_t<short, Dim2>, D> range;
    // skip the axis
    if constexpr (D == Dim1) {
      range[0][0] = 0;
      range[0][1] = 0;
    } else if constexpr (D == Dim2) {
      range[0][0] = 0;
      range[0][1] = 0;
      range[1][0] = 1;
      range[1][1] = -1;
    } else if constexpr (D == Dim3) {
      range[0][0] = 0;
      range[0][1] = 0;
      range[1][0] = 1;
      range[1][1] = -1;
      range[2][0] = 0;
      range[2][1] = 0;
    }
    Kokkos::parallel_for("add_currents",
                         this->m_mblock.rangeCells(range),
                         AddCurrentsCurvilinear<D>(this->m_mblock, coeff));
#endif
  }
} // namespace ntt

template class ntt::PIC<ntt::Dim1>;
template class ntt::PIC<ntt::Dim2>;
template class ntt::PIC<ntt::Dim3>;
