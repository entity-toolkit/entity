#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "qmath.h"
#include "sim_params.h"

#include "archetypes.hpp"
#include "generate_fields.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct VerticalPotential : public VectorPotential<D, S> {
    VerticalPotential(const SimulationParams& params, const Meshblock<D, S>& mblock)
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
  Inline auto VerticalPotential<Dim2, GRPICEngine>::A_x3(const coord_t<Dim2>& x_cu) const
    -> real_t {
    coord_t<Dim2> rth_;
    (this->m_mblock).metric.x_Code2Sph(x_cu, rth_);
    return HALF * SQR(math::sin(rth_[1])) * SQR(rth_[0]);
  }

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : TargetFields<D, S>(params, mblock), _epsilon { ONE }, v_pot { params, mblock } {}
    Inline auto operator()(const em&, const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t            _epsilon;
    VerticalPotential<D, S> v_pot;
  };

  template <>
  Inline auto PgenTargetFields<Dim2, GRPICEngine>::operator()(const em&            comp,
                                                              const coord_t<Dim2>& xi) const
    -> real_t {
    if (comp == em::bx1) {
      coord_t<Dim2> x0m { ZERO }, x0p { ZERO };
      real_t        inv_sqrt_detH_ijP { ONE / m_mblock.metric.sqrt_det_h(xi) };
      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF * _epsilon;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF * _epsilon;
      return (v_pot.A_x3(x0p) - v_pot.A_x3(x0m)) * inv_sqrt_detH_ijP / _epsilon;
    } else if (comp == em::bx2) {
      coord_t<Dim2> x0m { ZERO }, x0p { ZERO };
      real_t        inv_sqrt_detH_iPj { ONE / m_mblock.metric.sqrt_det_h(xi) };
      x0m[0] = xi[0] + HALF - HALF * _epsilon;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF + HALF * _epsilon;
      x0p[1] = xi[1];
      if (AlmostEqual(xi[1], ZERO)) {
        return ZERO;
      } else {
        return -(v_pot.A_x3(x0p) - v_pot.A_x3(x0m)) * inv_sqrt_detH_iPj / _epsilon;
      }
    } else {
      return ZERO;
    }
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
      Generate2DGRFromVectorPotential_kernel<VerticalPotential>(params, mblock, ONE));
  }
}    // namespace ntt

#endif
