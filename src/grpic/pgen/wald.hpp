#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "generate_fields.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct WaldPotential : public VectorPotential<D, S> {
    WaldPotential(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : VectorPotential<D, S>(params, mblock) {}
    Inline auto A_x0(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    Inline auto A_x1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    Inline auto A_x2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
    Inline auto A_x3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
  };

  template <>
  Inline auto WaldPotential<Dim2, GRPICEngine>::A_x0(const coord_t<Dim2>& x_cu) const
    -> real_t {
    real_t g00 { -(this->m_mblock).metric.alpha(x_cu) * (this->m_mblock).metric.alpha(x_cu)
                 + (this->m_mblock).metric.h_11(x_cu) * (this->m_mblock).metric.beta1u(x_cu)
                     * (this->m_mblock).metric.beta1u(x_cu) };
    return HALF
           * ((this->m_mblock).metric.h_13(x_cu) * (this->m_mblock).metric.beta1u(x_cu)
              + TWO * (this->m_mblock).metric.spin() * g00);
  }

  template <>
  Inline auto WaldPotential<Dim2, GRPICEngine>::A_x1(const coord_t<Dim2>& x_cu) const
    -> real_t {
    return HALF
           * ((this->m_mblock).metric.h_13(x_cu)
              + TWO * (this->m_mblock).metric.spin() * (this->m_mblock).metric.h_11(x_cu)
                  * (this->m_mblock).metric.beta1u(x_cu));
  }

  template <>
  Inline auto WaldPotential<Dim2, GRPICEngine>::A_x3(const coord_t<Dim2>& x_cu) const
    -> real_t {
    return HALF
           * ((this->m_mblock).metric.h_33(x_cu)
              + TWO * (this->m_mblock).metric.spin() * (this->m_mblock).metric.h_13(x_cu)
                  * (this->m_mblock).metric.beta1u(x_cu));
  }

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {}
  };

  template <>
  inline void ProblemGenerator<Dim2, GRPICEngine>::UserInitFields(
    const SimulationParams& params, Meshblock<Dim2, GRPICEngine>& mblock) {
    Kokkos::parallel_for(
      "UserInitFields",
      CreateRangePolicy<Dim2>({ mblock.i1_min() - 1, mblock.i2_min() },
                              { mblock.i1_max(), mblock.i2_max() + 1 }),
      Generate2DGRFromVectorPotential_kernel<WaldPotential>(params, mblock, ONE));
  }
}    // namespace ntt

#endif
