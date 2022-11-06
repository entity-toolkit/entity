#ifndef FRAMEWORK_INJECTOR_H
#define FRAMEWORK_INJECTOR_H

#include "wrapper.h"
#include "sim_params.h"
#include "meshblock.h"
#include "particles.h"
#include "archetypes.hpp"
#include "particle_macros.h"

#include <vector>

namespace ntt {

  /*
   * @brief 1D particle-vectorized injection kernel
   */
  template <SimulationType S, template <Dimension, SimulationType> class EnDist>
  struct Injector1d_kernel {
    Injector1d_kernel(const SimulationParams&   pr,
                      const Meshblock<Dim1, S>& mb,
                      const Particles<Dim1, S>& sp,
                      const std::size_t&        ofs,
                      const list_t<real_t, 2>&  box,
                      const real_t&             time)
      : params {pr},
        mblock {mb},
        species {sp},
        offset {ofs},
        region {box[0], box[1]},
        energy_dist {params, mblock},
        pool {(uint64_t)(1e6 * (time / mb.timestep()))} {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim1> x {ZERO};
      vec_t<Dim3>   v {ZERO};
      energy_dist(x, v);
      init_prtl_1d(mblock, species, p + offset, x[0], v[0], v[1], v[2]);
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams   params;
    Meshblock<Dim1, S> mblock;
    Particles<Dim1, S> species;
    const std::size_t  offset;
    EnDist<Dim1, S>    energy_dist;
    list_t<real_t, 2>  region;
    RandomNumberPool_t pool {constant::RandomSeed};
  };

  /*
   * @brief 2D particle-vectorized injection kernel
   */
  template <SimulationType S, template <Dimension, SimulationType> class EnDist>
  struct Injector2d_kernel {
    Injector2d_kernel(const SimulationParams&   pr,
                      const Meshblock<Dim2, S>& mb,
                      const Particles<Dim2, S>& sp,
                      const std::size_t&        ofs,
                      const list_t<real_t, 4>&  box,
                      const real_t&             time)
      : params {pr},
        mblock {mb},
        species {sp},
        offset {ofs},
        region {box[0], box[1], box[2], box[3]},
        energy_dist {params, mblock},
        pool {(uint64_t)(1e6 * (time / mb.timestep()))} {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim2> x {ZERO};
      vec_t<Dim3>   v {ZERO};
      x[0] = rand_gen.frand(region[0], region[1]);
      x[1] = rand_gen.frand(region[2], region[3]);
      energy_dist(x, v);
      init_prtl_2d(mblock, species, p + offset, x[0], x[1], v[0], v[1], v[2]);
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams   params;
    Meshblock<Dim2, S> mblock;
    Particles<Dim2, S> species;
    const std::size_t  offset;
    EnDist<Dim2, S>    energy_dist;
    list_t<real_t, 4>  region;
    RandomNumberPool_t pool;
  };

  /*
   * @brief 3D particle-vectorized injection kernel
   */
  template <SimulationType S, template <Dimension, SimulationType> class EnDist>
  struct Injector3d_kernel {
    Injector3d_kernel(const SimulationParams&   pr,
                      const Meshblock<Dim3, S>& mb,
                      const Particles<Dim3, S>& sp,
                      const std::size_t&        ofs,
                      const list_t<real_t, 6>&  box,
                      const real_t&             time)
      : params {pr},
        mblock {mb},
        species {sp},
        offset {ofs},
        region {box[0], box[1], box[2], box[3], box[4], box[5]},
        energy_dist {params, mblock},
        pool {(uint64_t)(1e6 * (time / mb.timestep()))} {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim3> x {ZERO};
      vec_t<Dim3>   v {ZERO};
      energy_dist(x, v);
      init_prtl_3d(mblock, species, p + offset, x[0], x[1], x[2], v[0], v[1], v[2]);
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams   params;
    Meshblock<Dim3, S> mblock;
    Particles<Dim3, S> species;
    const std::size_t  offset;
    EnDist<Dim3, S>    energy_dist;
    list_t<real_t, 6>  region;
    RandomNumberPool_t pool;
  };

  template <Dimension      D,
            SimulationType S,
            template <Dimension, SimulationType> class EnDist = ColdDist>
  inline void InjectUniform(const SimulationParams& params,
                            Meshblock<D, S>&        mblock,
                            const std::vector<int>& species,
                            const real_t&           ppc_per_spec,
                            std::vector<real_t>     region = {},
                            const real_t&           time   = ZERO) {
    auto         ncells         = (std::size_t)(mblock.Ni1() * mblock.Ni2() * mblock.Ni3());
    auto         npart_per_spec = (std::size_t)((double)(ncells * ppc_per_spec));
    EnDist<D, S> energy_dist(params, mblock);
    if (region.size() == 0) {
      region = mblock.extent();
    }
    for (auto& s : species) {
      auto& sp           = mblock.particles[s - 1];
      auto  npart_before = sp.npart();
      sp.setNpart(npart_before + npart_per_spec);
      if constexpr (D == Dim1) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
          Injector1d_kernel<S, EnDist>(
            params, mblock, sp, npart_before, {region[0], region[1]}, time));
      } else if constexpr (D == Dim2) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
          Injector2d_kernel<S, EnDist>(params,
                                       mblock,
                                       sp,
                                       npart_before,
                                       {region[0], region[1], region[2], region[3]},
                                       time));
      } else if constexpr (D == Dim3) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
          Injector3d_kernel<S, EnDist>(
            params,
            mblock,
            sp,
            npart_before,
            {region[0], region[1], region[2], region[3], region[4], region[5]},
            time));
      }
    }
  }

} // namespace ntt

#endif // FRAMEWORK_INJECTOR_H