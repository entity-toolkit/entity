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

  /* -------------------------------------------------------------------------- */
  /*                   Uniform injection kernels and routines                   */
  /* -------------------------------------------------------------------------- */

  /*
   * @brief 1D particle-vectorized injection kernel
   */
  template <SimulationType S, template <Dimension, SimulationType> class EnDist>
  struct InjectorUniform1d_kernel {
    InjectorUniform1d_kernel(const SimulationParams&   pr,
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
      x[0] = rand_gen.frand(region[0], region[1]);
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
  struct InjectorUniform2d_kernel {
    InjectorUniform2d_kernel(const SimulationParams&   pr,
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
  struct InjectorUniform3d_kernel {
    InjectorUniform3d_kernel(const SimulationParams&   pr,
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
      x[0] = rand_gen.frand(region[0], region[1]);
      x[1] = rand_gen.frand(region[2], region[3]);
      x[2] = rand_gen.frand(region[4], region[5]);
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
    auto   ncells = (std::size_t)(mblock.Ni1() * mblock.Ni2() * mblock.Ni3());
    real_t delta_V, full_V;
    if (region.size() == 0) {
      region = mblock.extent();
    }
    if constexpr (D == Dim1) {
      delta_V = (region[1] - region[0]);
      full_V  = (mblock.extent()[1] - mblock.extent()[0]);
    } else if constexpr (D == Dim2) {
      delta_V = (region[1] - region[0]) * (region[3] - region[2]);
      full_V  = (mblock.extent()[1] - mblock.extent()[0])
               * (mblock.extent()[3] - mblock.extent()[2]);
    } else if constexpr (D == Dim3) {
      delta_V = (region[1] - region[0]) * (region[3] - region[2]) * (region[5] - region[4]);
      full_V  = (mblock.extent()[1] - mblock.extent()[0])
               * (mblock.extent()[3] - mblock.extent()[2])
               * (mblock.extent()[5] - mblock.extent()[4]);
    }
    ncells = (std::size_t)((real_t)ncells * delta_V / full_V);

    auto npart_per_spec = (std::size_t)((double)(ncells * ppc_per_spec));
    list_t<real_t, 2 * static_cast<short>(D)> box {ZERO};
    for (auto i {0}; i < 2 * static_cast<short>(D); ++i) {
      box[i] = region[i];
    }

    for (auto& s : species) {
      auto& sp           = mblock.particles[s - 1];
      auto  npart_before = sp.npart();
      sp.setNpart(npart_before + npart_per_spec);
      if constexpr (D == Dim1) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
          InjectorUniform1d_kernel<S, EnDist>(params, mblock, sp, npart_before, box, time));
      } else if constexpr (D == Dim2) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
          InjectorUniform2d_kernel<S, EnDist>(params, mblock, sp, npart_before, box, time));
      } else if constexpr (D == Dim3) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
          InjectorUniform3d_kernel<S, EnDist>(params, mblock, sp, npart_before, box, time));
      }
    }
  }

  /* -------------------------------------------------------------------------- */
  /*                    Volume injection kernels and routines                   */
  /* -------------------------------------------------------------------------- */

  template <SimulationType S,
            template <Dimension, SimulationType>
            class EnDist,
            template <Dimension, SimulationType>
            class SpDist,
            template <Dimension, SimulationType>
            class InjCrit>
  struct VolumeInjector1d_kernel {
    VolumeInjector1d_kernel(const SimulationParams&     pr,
                            const Meshblock<Dim2, S>&   mb,
                            const Particles<Dim2, S>&   sp,
                            const std::size_t&          ppc,
                            const array_t<std::size_t>& nprt,
                            const real_t&               time)
      : params {pr},
        mblock {mb},
        species {sp},
        nppc {(real_t)ppc},
        npart {nprt},
        energy_dist {params, mblock},
        spatial_dist {params, mblock},
        inj_criterion {params, mblock},
        pool {(uint64_t)(1e6 * (time / mb.timestep()))} {}
    Inline void operator()(index_t i1) const {
//       // cell node
//       coord_t<Dim1> xi {static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS)};
//       // cell center
//       coord_t<Dim1> xc {xi[0] + HALF};
//       // physical coordinate
//       coord_t<Dim1> xph {ZERO};

// #ifdef MINKOWSKI_METRIC
//       mblock.metric.x_Code2Cart(xc, xph);
// #else
//       mblock.metric.x_Code2Sph(xc, xph);
// #endif

//       if (inj_criterion(xph)) {
//         typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

//         real_t ninject = nppc * spatial_dist(xph);
//         while (ninject > ZERO) {
//           real_t random = rand_gen.frand();
//           if (random < ninject) {
//             vec_t<Dim3> v {ZERO};
//             energy_dist(xph, v);

            real_t dx1 = rand_gen.frand();

            auto p         = Kokkos::atomic_fetch_add(&npart(), 1);
            species.i1(p)  = static_cast<int>(i1) - N_GHOSTS;
            species.dx1(p) = dx1;
            species.ux1(p) = v[0];
            species.ux2(p) = v[1];
            species.ux3(p) = v[2];
          }
          ninject -= ONE;
        }
        pool.free_state(rand_gen);
      }
    }

  private:
    SimulationParams     params;
    Meshblock<Dim1, S>   mblock;
    Particles<Dim1, S>   species;
    const real_t         nppc;
    array_t<std::size_t> npart;
    EnDist<Dim1, S>      energy_dist;
    SpDist<Dim1, S>      spatial_dist;
    InjCrit<Dim1, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  template <Dimension      D,
            SimulationType S,
            template <Dimension, SimulationType> class EnDist  = ColdDist,
            template <Dimension, SimulationType> class SpDist  = UniformDist,
            template <Dimension, SimulationType> class InjCrit = NoCriterion>
  inline void InjectInVolume(const SimulationParams& params,
                             Meshblock<D, S>&        mblock,
                             const std::vector<int>& species,
                             const real_t&           ppc_per_spec,
                             std::vector<real_t>     region = {},
                             const real_t&           time   = ZERO) {
    EnDist<D, S>  energy_dist(params, mblock);
    SpDist<D, S>  spatial_dist(params, mblock);
    InjCrit<D, S> inj_criterion(params, mblock);
    if (region.size() == 0) {
      region = mblock.extent();
    }
    for (auto& s : species) {
      auto&                sp = mblock.particles[s - 1];
      array_t<std::size_t> npart("npart_sp");
      auto                 npart_h = Kokkos::create_mirror(npart);
      npart_h()                    = sp.npart();
      std::cout << "npart before: " << npart_h() << std::endl;
      Kokkos::deep_copy(npart, npart_h);

      // Kokkos::parallel_for(
      //   "inject",
      //   CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
      //   InjectorUniform1d_kernel<S, EnDist>(params, mblock, sp, npart_before, box, time));

      Kokkos::deep_copy(npart_h, npart);
      std::cout << "npart after: " << npart_h() << std::endl;
    }
    //   if constexpr (D == Dim1) {
    //     Kokkos::parallel_for(
    //       "inject",
    //       CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
    //       Injector1d_kernel<S, EnDist>(
    //         params, mblock, sp, npart_before, {region[0], region[1]}, time));
    //   } else if constexpr (D == Dim2) {
    //     Kokkos::parallel_for(
    //       "inject",
    //       CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
    //       Injector2d_kernel<S, EnDist>(params,
    //                                    mblock,
    //                                    sp,
    //                                    npart_before,
    //                                    {region[0], region[1], region[2], region[3]},
    //                                    time));
    //   } else if constexpr (D == Dim3) {
    //     Kokkos::parallel_for(
    //       "inject",
    //       CreateRangePolicy<Dim1>({0}, {npart_per_spec}),
    //       Injector3d_kernel<S, EnDist>(
    //         params,
    //         mblock,
    //         sp,
    //         npart_before,
    //         {region[0], region[1], region[2], region[3], region[4], region[5]},
    //         time));
    //   }
    // }
  }

} // namespace ntt

#endif // FRAMEWORK_INJECTOR_H