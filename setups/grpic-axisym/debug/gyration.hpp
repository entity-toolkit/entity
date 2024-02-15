#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "particle_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"
#include "utils/qmath.h"

#include "utils/archetypes.hpp"
#include "utils/generate_fields.hpp"

namespace ntt {

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
  Inline auto VerticalPotential<Dim2, GRPICEngine>::A_x3(
    const coord_t<Dim2>& x_cu) const -> real_t {
    coord_t<Dim2> rth_;
    (this->m_mblock).metric.x_Code2Sph(x_cu, rth_);
    return HALF * SQR(math::sin(rth_[1])) * SQR(rth_[0]);
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
    const real_t            _epsilon;
    VerticalPotential<D, S> v_pot;
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
      r0 { params.get<std::vector<real_t>>("problem", "r0") },
      th0 { params.get<std::vector<real_t>>("problem", "th0") },
      ph0 { params.get<std::vector<real_t>>("problem", "ph0") },
      ur0 { params.get<std::vector<real_t>>("problem", "ur0") },
      uth0 { params.get<std::vector<real_t>>("problem", "uth0") },
      uph0 { params.get<std::vector<real_t>>("problem", "uph0") } {}

    inline void UserInitParticles(const SimulationParams&,
                                  Meshblock<D, S>&) override {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
    }

  private:
    const std::vector<real_t> r0, th0, ph0, ur0, uth0, uph0;
  };

  template <>
  inline void ProblemGenerator<Dim2, GRPICEngine>::UserInitFields(
    const SimulationParams&       params,
    Meshblock<Dim2, GRPICEngine>& mblock) {
    Kokkos::parallel_for(
      "UserInitFields",
      CreateRangePolicy<Dim2>({ mblock.i1_min() - 1, mblock.i2_min() },
                              { mblock.i1_max(), mblock.i2_max() + 1 }),
      Generate2DGRFromVectorPotential_kernel<VerticalPotential>(params, mblock, ONE));
  }

  template <>
  inline void ProblemGenerator<Dim2, GRPICEngine>::UserInitParticles(
    const SimulationParams&,
    Meshblock<Dim2, GRPICEngine>& mblock) {
    NTTHostErrorIf(r0.size() != th0.size() || r0.size() != ur0.size() ||
                     r0.size() != uth0.size() || r0.size() != uph0.size(),
                   "particle initial condition lenghts should be equal");
    auto&      species = mblock.particles[0];
    const auto npart   = r0.size();
    array_t<real_t*> r0_d("r0", npart), th0_d("th0", npart), ph0_d("ph0", npart),
      ur0_d("ur0", npart), uth0_d("uth0", npart), uph0_d("uph0", npart);
    auto r0_h   = Kokkos::create_mirror_view(r0_d);
    auto th0_h  = Kokkos::create_mirror_view(th0_d);
    auto ph0_h  = Kokkos::create_mirror_view(ph0_d);
    auto ur0_h  = Kokkos::create_mirror_view(ur0_d);
    auto uth0_h = Kokkos::create_mirror_view(uth0_d);
    auto uph0_h = Kokkos::create_mirror_view(uph0_d);
    for (auto i { 0 }; i < npart; ++i) {
      r0_h(i)   = r0[i];
      th0_h(i)  = th0[i];
      ph0_h(i)  = ph0[i];
      ur0_h(i)  = ur0[i];
      uth0_h(i) = uth0[i];
      uph0_h(i) = uph0[i];
    }
    Kokkos::deep_copy(r0_d, r0_h);
    Kokkos::deep_copy(th0_d, th0_h);
    Kokkos::deep_copy(ph0_d, ph0_h);
    Kokkos::deep_copy(ur0_d, ur0_h);
    Kokkos::deep_copy(uth0_d, uth0_h);
    Kokkos::deep_copy(uph0_d, uph0_h);
    Kokkos::parallel_for(
      "UserInitParticles",
      npart,
      ClassLambda(index_t p) {
        init_prtl_2d_covariant(mblock,
                               species,
                               p,
                               r0_d(p),
                               th0_d(p),
                               ur0_d(p),
                               uth0_d(p),
                               uph0_d(p),
                               ONE);
        species.phi(p) = ph0_d(p);
      });
    species.setNpart(npart);
  }
} // namespace ntt

#endif
