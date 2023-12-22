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

    // Define two instances of the vector potential class: Wald and Vertical

    // Initialize 
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

    // Define
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
    // coord_t<Dim2> x_ph;
    // (this->m_mblock).metric.x_Code2Sph(x_cu, x_ph);
    // return HALF * SQR(x_ph[0]) * SQR(math::sin(x_ph[1]));
    const auto r { (this->m_mblock).metric.x1_Code2Sph(x_cu[0]) };
    const auto th { (this->m_mblock).metric.x2_Code2Sph(x_cu[1]) };
    return HALF * SQR(r) * SQR(math::sin(th));
  }

    // Define the target field in outer radial boundary

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

    // Define problem generator. Two functors will be overriden: initialize fields and drive particles

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) :
      inj_rmin { params.get<real_t>("problem", "inj_rmin", (real_t)(2.0)) },
      inj_rmax { params.get<real_t>("problem", "inj_rmax", (real_t)(3.0)) },
      inj_thmin { params.get<real_t>("problem", "inj_thmin", (real_t)(0.0)) },
      inj_thmax { params.get<real_t>("problem", "inj_thmax", (real_t)(ntt::constant::PI)) } {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
    }

    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

    // inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}

  private:
    ndarray_t<(short)(D)> m_ppc_per_spec;
    real_t inj_rmin, inj_rmax, inj_thmin, inj_thmax;
  };

    // Initialize fields override

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

    // Injection criterion

  template <Dimension D, SimulationEngine S>
  struct SigmaCrit : public InjectionCriterion<D, S> {
    explicit SigmaCrit(const SimulationParams& params, Meshblock<D, S>& mblock) :
      InjectionCriterion<D, S>(params, mblock),
      _sigma_max { params.get<real_t>("problem", "sigma_max", (real_t)(1.0)) },
      _density_min { params.get<real_t>("problem", "multiplicity", (real_t)(1.0)) } {
    }

    Inline bool operator()(const coord_t<D>&) const {
      return true;
    }

  private:
    const real_t _sigma_max;
    const real_t _density_min;
    const real_t nGJ { (this->m_mblock).metric.spin() * (this->m_params).B0() * SQR((this->m_params).skindepth0()) };;
    const real_t sigma0 { SQR( (this->m_params).skindepth0() / (this->m_params).larmor0() ) };;
  };

  template <>
  Inline bool SigmaCrit<Dim2, GRPICEngine>::operator()(
    const coord_t<Dim2>& xph) const {
    coord_t<Dim2> xi { ZERO };

    (this->m_mblock).metric.x_Sph2Code(xph, xi);
    std::size_t i1 = (std::size_t)(xi[0] + N_GHOSTS);
    std::size_t i2 = (std::size_t)(xi[1] + N_GHOSTS);

    vec_t<Dim3> b_cntrv { (this->m_mblock).em(i1, i2, em::bx1), (this->m_mblock).em(i1, i2, em::bx2), (this->m_mblock).em(i1, i2, em::bx3) };
    vec_t<Dim3> b_cov;

    (this->m_mblock).metric.v3_Cntrv2Cov(xi, b_cntrv, b_cov);
    real_t sigma { DOT(b_cov[0], b_cov[1], b_cov[2], b_cntrv[0], b_cntrv[1], b_cntrv[2]) / ((this->m_mblock).buff(i1, i2, 0)) };

    return (sigma > _sigma_max / sigma0) || ((this->m_mblock).buff(i1, i2, 0) < _density_min * nGJ);
    // return ((this->m_mblock).buff(i1, i2, 0) < _density_min * nGJ);
  }

//     // Initial particles override

//   template <>
//   inline void ProblemGenerator<Dim2, GRPICEngine>::UserInitParticles(
//     const SimulationParams& params,
//     Meshblock<Dim2, GRPICEngine>&        mblock) {

//     m_ppc_per_spec = ndarray_t<2>("ppc_per_spec", mblock.Ni1(), mblock.Ni2());
//     const auto ppc0  = params.ppc0();

//     Kokkos::parallel_for(
//         "mppc_per_spec",
//         mblock.rangeActiveCells(),
//         ClassLambda(index_t i1, index_t i2) {
//         const auto j1_ = static_cast<int>(i1) - N_GHOSTS;
//         const auto j2_ = static_cast<int>(i2) - N_GHOSTS;
//         m_ppc_per_spec(j1_, j2_) = ppc0;
//         });

//   InjectNonUniform<Dim2, GRPICEngine>(
//     params, 
//     mblock, 
//     { 1, 2 }, 
//     m_ppc_per_spec,n_inject
//     { inj_rmin, inj_rmax, mblock.metric.x2_min, mblock.metric.x2_max });

// //   InjectNonUniform<Dim2, GRPICEngine>(
// //     params, 
// //     mblock, 
// //     { 1, 2 }, 
// //     m_ppc_per_spec,
// //     { inj_rmin, inj_rmax, 0.1, ntt::constant::PI - 0.1});

//   }

    // Drive particles override

template <>
inline void ProblemGenerator<Dim2, GRPICEngine>::UserDriveParticles(
  const real_t&               time,
  const SimulationParams&     params,
  Meshblock<Dim2, GRPICEngine>& mblock) {
//   auto nppc_per_spec = (real_t)(params.ppc0()) * 0.1; 
  m_ppc_per_spec = ndarray_t<2>("ppc_per_spec", mblock.Ni1(), mblock.Ni2());
  const auto ppc0  = params.ppc0();
  const short smooth = 1;

//   mblock.ComputeMoments(params, FieldID::N, {}, { 1, 2 }, 0, smooth);
  mblock.ComputeDensity(params, { 1, 2 }, 0, smooth);
//   WaitAndSynchronize();

      Kokkos::parallel_for(
        "mppc_per_spec",
        mblock.rangeActiveCells(),
        ClassLambda(index_t i1, index_t i2) {
          const auto j1_ = static_cast<int>(i1) - N_GHOSTS;
          const auto j2_ = static_cast<int>(i2) - N_GHOSTS;
          m_ppc_per_spec(j1_, j2_) = HALF * ppc0;
        });

  InjectNonUniform<Dim2, GRPICEngine, ColdDist, SigmaCrit>( 
    params, 
    mblock, 
    { 1, 2 }, 
    m_ppc_per_spec, 
    { inj_rmin, inj_rmax, inj_thmin, inj_thmax });

}

} // namespace ntt

#endif
