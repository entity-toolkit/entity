#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "archetypes.h"
#include "sim_params.h"
#include "meshblock.h"
#include "particle_macros.h"

#ifdef NTTINY_ENABLED
#  include "nttiny/api.h"
#endif

#include <map>

namespace ntt {

  struct Params {
    real_t N_ppc {-1.0};
    real_t T {-1.0};
  };

  enum InjectorFlags_ {
    InjectorFlags_None           = 0,
    InjectorFlags_UseMaxwellian  = 1 << 0,
    InjectorFlags_UseDensityDist = 1 << 2,
    InjectorFlags_UseCriterion   = 1 << 1,
    // InjectorFlags_... = 1 << 3,
    // InjectorFlags_... = 1 << 4,
    // InjectorFlags_... = 1 << 5,
    // InjectorFlags_... = 1 << 6,
    // InjectorFlags_... = 1 << 7,
  };
  typedef int InjectorFlags;

  template <Dimension D, SimulationType S>
  struct CoshDistribution : public SpatialDistribution<D, S> {
    explicit CoshDistribution(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline real_t operator()(real_t x_ph) const { return 1.0 / x_ph; }
  };

  template <Dimension      D,
            SimulationType S,
            template <Dimension, SimulationType>
            class EnergyDist,
            template <Dimension, SimulationType>
            class SpatialDist,
            template <Dimension, SimulationType>
            class InjCriterion>
  class Injector_kernel {
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
    std::vector<int> m_species;
    real_t           m_ppc_per_spec;

  public:
    Injector_kernel(const std::vector<int>& species,
                    const real_t&           ppc_per_spec,
                    const SimulationParams& params,
                    const Meshblock<D, S>&  mblock)
      : m_species(species), m_ppc_per_spec(ppc_per_spec), m_params(params), m_mblock(mblock) {}
    inline void inject() {
      EnergyDist<D, S>   energy_dist(m_params, m_mblock);
      SpatialDist<D, S>  spatial_dist(m_params, m_mblock);
      InjCriterion<D, S> inj_criterion(m_params, m_mblock);

      auto ncells         = (std::size_t)(m_mblock.Ni1() * m_mblock.Ni2() * m_mblock.Ni3());
      auto npart_per_spec = (std::size_t)((double)(ncells * m_ppc_per_spec));
      for (auto& s : m_species) {
        m_mblock.particles[s - 1].setNpart(npart_per_spec);
      }
      Kokkos::parallel_for(
        "dada", CreateRangePolicy<Dim1>({0}, {1}), Lambda(index_t p) {
          coord_t<D>  x {ZERO};
          vec_t<Dim3> v {ZERO};
          energy_dist(x, v);
        });
    }
  };
  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams& params, Meshblock<D, S>& mblock) {
      auto n_per_spec = (real_t)(params.ppc0()) * HALF;

      Injector_kernel<D, S, ColdDist, UniformDist, NoCriterion> ik1(
        {1, 2}, n_per_spec, params, mblock);
      // Injector_kernel<D, S, UpstreamMaxwellian, UniformDist, NoCriterion> ik2(
      //   {3, 4}, {overdensity * ppc0 * HALF, overdensity * ppc0 * HALF}, params, mblock);

      // Injector_kernel<D, S, ColdDist, UniformDist, NoCriterion> ik1(
      //   {1, 2}, {params.ppc0() / 2.0f, params.ppc0() / 2.0f}, params, mblock);
      // Injector_kernel<D, S, UpstreamMaxwellian, UniformDist, NoCriterion> ik1(params,
      // mblock);

      ik1.inject();

      // if constexpr (D == Dim2) {
      //   auto&                electrons   = mblock.particles[0];
      //   auto                 random_pool = *(mblock.random_pool_ptr);
      //   array_t<std::size_t> nprtl("Nprtl");
      //   Kokkos::parallel_for(
      //     "test_prtl_inject", mblock.rangeActiveCells(), Lambda(index_t i1, index_t i2) {
      //       const real_t i1_ {static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS)};
      //       const real_t i2_ {static_cast<real_t>(static_cast<int>(i2) - N_GHOSTS)};

      //       // if (i1 % 2 == 0) {
      //       typename RandomNumberPool_t::generator_type rand_gen =
      //       random_pool.get_state(); std::size_t p = Kokkos::atomic_fetch_add(&nprtl(),
      //       1);

      //       vec_t<Dim2> xmin_Cart {ZERO}, xmax_Cart {ZERO};
      //       mblock.metric.x_Code2Cart({i1_, i2_}, xmin_Cart);
      //       mblock.metric.x_Code2Cart({i1_ + ONE, i2_ + ONE}, xmax_Cart);
      //       real_t rx = rand_gen.frand(xmin_Cart[0], xmax_Cart[0]);
      //       real_t ry = rand_gen.frand(xmin_Cart[1], xmax_Cart[1]);
      //       init_prtl_2d(mblock, electrons, p, rx, ry, 0.0, 0.0, 0.0);
      //       random_pool.free_state(rand_gen);
      //       // }
      //     });
      //   auto nprtl_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nprtl);
      //   electrons.setNpart(nprtl_h());
      // }
    }
  }; // struct ProblemGenerator

} // namespace ntt

#endif
