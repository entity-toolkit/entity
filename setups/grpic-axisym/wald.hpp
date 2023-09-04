#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "sim_params.h"

#include "meshblock/meshblock.h"
#include "utils/qmath.h"

#include "utils/archetypes.hpp"
#include "utils/generate_fields.hpp"
#include "utils/injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct WaldPotential : public VectorPotential<D, S> {
    WaldPotential(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      VectorPotential<D, S>(params, mblock) {}

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

  template <Dimension D, SimulationEngine S>
  struct VerticalPotential : public VectorPotential<D, S> {
    VerticalPotential(const SimulationParams& params,
                      const Meshblock<D, S>&  mblock) :
      VectorPotential<D, S>(params, mblock) {}

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
    real_t g00 { -(this->m_mblock).metric.alpha(x_cu) *
                   (this->m_mblock).metric.alpha(x_cu) +
                 (this->m_mblock).metric.h_11(x_cu) *
                   (this->m_mblock).metric.beta1(x_cu) *
                   (this->m_mblock).metric.beta1(x_cu) };
    return HALF * ((this->m_mblock).metric.h_13(x_cu) *
                     (this->m_mblock).metric.beta1(x_cu) +
                   TWO * (this->m_mblock).metric.spin() * g00);
  }

  template <>
  Inline auto WaldPotential<Dim2, GRPICEngine>::A_x1(const coord_t<Dim2>& x_cu) const
    -> real_t {
    return HALF * ((this->m_mblock).metric.h_13(x_cu) +
                   TWO * (this->m_mblock).metric.spin() *
                     (this->m_mblock).metric.h_11(x_cu) *
                     (this->m_mblock).metric.beta1(x_cu));
  }

  template <>
  Inline auto WaldPotential<Dim2, GRPICEngine>::A_x3(const coord_t<Dim2>& x_cu) const
    -> real_t {
    return HALF * ((this->m_mblock).metric.h_33(x_cu) +
                   TWO * (this->m_mblock).metric.spin() *
                     (this->m_mblock).metric.h_13(x_cu) *
                     (this->m_mblock).metric.beta1(x_cu));
  }

  template <>
  Inline auto VerticalPotential<Dim2, GRPICEngine>::A_x3(
    const coord_t<Dim2>& x_cu) const -> real_t {
    coord_t<Dim2> x_ph;
    (this->m_mblock).metric.x_Code2Sph(x_cu, x_ph);
    return HALF * SQR(x_ph[0]) * SQR(math::sin(x_ph[1]));
  }

  template <Dimension D, SimulationEngine S>
  struct PgenTargetFields : public TargetFields<D, S> {
    PgenTargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      TargetFields<D, S>(params, mblock),
      _epsilon { ONE },
      v_pot { params, mblock } {}

    Inline auto operator()(const em&, const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t        _epsilon;
    WaldPotential<D, S> v_pot;
  };

  template <>
  Inline auto PgenTargetFields<Dim2, GRPICEngine>::operator()(
    const em&            comp,
    const coord_t<Dim2>& xi) const -> real_t {
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
    inline ProblemGenerator(const SimulationParams& params) :
      inj_fraction { params.get<real_t>("problem", "inj_fraction", (real_t)(0.1)) },
      inj_rmax { params.get<real_t>("problem", "inj_rmax", (real_t)(2.0)) } {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
    }

    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

  private:
    const real_t inj_fraction, inj_rmax;
  };

  template <>
  inline void ProblemGenerator<Dim2, GRPICEngine>::UserInitFields(
    const SimulationParams&       params,
    Meshblock<Dim2, GRPICEngine>& mblock) {
    Kokkos::parallel_for(
      "UserInitFields",
      CreateRangePolicy<Dim2>({ mblock.i1_min() - 1, mblock.i2_min() },
                              { mblock.i1_max(), mblock.i2_max() + 1 }),
      Generate2DGRFromVectorPotential_kernel<WaldPotential>(params, mblock, ONE));

    Kokkos::parallel_for(
      "UserInitFields",
      mblock.rangeAllCells(),
      Lambda(index_t i1, index_t i2) {
        mblock.em(i1, i2, em::dx1) = ZERO;
        mblock.em(i1, i2, em::dx2) = ZERO;
        mblock.em(i1, i2, em::dx3) = ZERO;
      });
  }

  template <Dimension D, SimulationEngine S>
  struct RadialKick : public EnergyDistribution<D, S> {
    RadialKick(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      EnergyDistribution<D, S>(params, mblock),
      u_kick { params.get<real_t>("problem", "u_kick", ZERO) } {}

    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v, const int&) const override {
      v[0] = u_kick;
    }

  private:
    const real_t u_kick;
  };

  template <Dimension D, SimulationEngine S>
  struct InjectionShell : public SpatialDistribution<D, S> {
    explicit InjectionShell(const SimulationParams& params,
                            Meshblock<D, S>&        mblock) :
      SpatialDistribution<D, S>(params, mblock),
      _inj_rmin { params.get<real_t>("problem", "r_surf", (real_t)(1.0)) },
      _inj_rmax { params.get<real_t>("problem", "inj_rmax", (real_t)(1.5)) } {
      NTTHostErrorIf(_inj_rmin >= _inj_rmax, "inj_rmin >= inj_rmax");
    }

    Inline real_t operator()(const coord_t<D>& x_ph) const {
      return ((x_ph[0] <= _inj_rmax) && (x_ph[0] > _inj_rmin)) ? ONE : ZERO;
    }

  private:
    const real_t _inj_rmin, _inj_rmax;
  };

  template <Dimension D, SimulationEngine S>
  struct MaxDensCrit : public InjectionCriterion<D, S> {
    explicit MaxDensCrit(const SimulationParams& params, Meshblock<D, S>& mblock) :
      InjectionCriterion<D, S>(params, mblock),
      _inj_maxdens { params.get<real_t>("problem", "inj_maxdens", (real_t)(5.0)) } {
    }

    Inline bool operator()(const coord_t<D>&) const {
      return true;
    }

  private:
    const real_t _inj_maxdens;
  };

  template <>
  Inline bool MaxDensCrit<Dim2, GRPICEngine>::operator()(
    const coord_t<Dim2>& xph) const {
    coord_t<Dim2> xi { ZERO };
    (this->m_mblock).metric.x_Sph2Code(xph, xi);
    std::size_t i1 = (std::size_t)(xi[0] + N_GHOSTS);
    std::size_t i2 = (std::size_t)(xi[1] + N_GHOSTS);
    return (this->m_mblock).buff(i1, i2, 0) < _inj_maxdens;
  }
} // namespace ntt

#endif
