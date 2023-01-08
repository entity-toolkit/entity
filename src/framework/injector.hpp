#ifndef FRAMEWORK_INJECTOR_H
#define FRAMEWORK_INJECTOR_H

#include "wrapper.h"

#include "meshblock.h"
#include "particle_macros.h"
#include "particles.h"
#include "sim_params.h"

#include "archetypes.hpp"

#include <vector>

namespace ntt {

  /* -------------------------------------------------------------------------- */
  /*                   Uniform injection kernels and routines                   */
  /* -------------------------------------------------------------------------- */

  /**
   * @brief 1D particle-vectorized injection kernel
   */
  template <SimulationType S, template <Dimension, SimulationType> class EnDist>
  struct UniformInjector1d_kernel {
    UniformInjector1d_kernel(const SimulationParams&   pr,
                             const Meshblock<Dim1, S>& mb,
                             const Particles<Dim1, S>& sp,
                             const std::size_t&        ofs,
                             const list_t<real_t, 2>&  box,
                             const real_t&             time)
      : params { pr },
        mblock { mb },
        species { sp },
        offset { ofs },
        region { box[0], box[1] },
        energy_dist { params, mblock },
        pool { (uint64_t)(1e6 * (time / mb.timestep())) } {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim1>                               x { ZERO };
      vec_t<Dim3>                                 v { ZERO };
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
    RandomNumberPool_t pool { constant::RandomSeed };
  };

  /**
   * @brief 2D particle-vectorized injection kernel
   */
  template <SimulationType S, template <Dimension, SimulationType> class EnDist>
  struct UniformInjector2d_kernel {
    UniformInjector2d_kernel(const SimulationParams&   pr,
                             const Meshblock<Dim2, S>& mb,
                             const Particles<Dim2, S>& sp,
                             const std::size_t&        ofs,
                             const list_t<real_t, 4>&  box,
                             const real_t&             time)
      : params { pr },
        mblock { mb },
        species { sp },
        offset { ofs },
        region { box[0], box[1], box[2], box[3] },
        energy_dist { params, mblock },
        pool { *(mb.random_pool_ptr) } {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim2>                               x { ZERO };
      vec_t<Dim3>                                 v { ZERO };
      x[0] = rand_gen.frand(region[0], region[1]);
      x[1] = rand_gen.frand(region[2], region[3]);
      energy_dist(x, v);
      init_prtl_2d(mblock, species, p + offset, x[0], x[1], v[0], v[1], v[2]);
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams    params;
    Meshblock<Dim2, S>  mblock;
    Particles<Dim2, S>  species;
    const std::size_t   offset;
    EnDist<Dim2, S>     energy_dist;
    list_t<real_t, 4>   region;
    RandomNumberPool_t& pool;
  };

  /**
   * @brief 3D particle-vectorized injection kernel
   */
  template <SimulationType S, template <Dimension, SimulationType> class EnDist>
  struct UniformInjector3d_kernel {
    UniformInjector3d_kernel(const SimulationParams&   pr,
                             const Meshblock<Dim3, S>& mb,
                             const Particles<Dim3, S>& sp,
                             const std::size_t&        ofs,
                             const list_t<real_t, 6>&  box,
                             const real_t&             time)
      : params { pr },
        mblock { mb },
        species { sp },
        offset { ofs },
        region { box[0], box[1], box[2], box[3], box[4], box[5] },
        energy_dist { params, mblock },
        pool { (uint64_t)(1e6 * (time / mb.timestep())) } {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim3>                               x { ZERO };
      vec_t<Dim3>                                 v { ZERO };
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

  /**
   * @brief Volumetrically uniform particle injector parallelized over particles.
   * @tparam D dimension.
   * @tparam S simulation type.
   * @tparam EnDist energy distribution [default = ColdDist].
   *
   * @param params simulation parameters.
   * @param mblock meshblock.
   * @param species species to inject as a list.
   * @param ppc_per_spec fiducial number of particles per cell per species.
   * @param region region to inject particles as a list of coordinates.
   * @param time current time.
   */
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
#ifdef MINKOWSKI_METRIC
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
#else
    if constexpr (D == Dim2) {
      delta_V = (SQR(region[1]) - SQR(region[0])) * (region[3] - region[2]);
      full_V  = (SQR(mblock.extent()[1]) - SQR(mblock.extent()[0])) * constant::PI * HALF;
    } else if constexpr (D == Dim3) {
      // !TODO: need to be a bit more careful
      delta_V = (CUBE(region[1]) - CUBE(region[0])) * (region[3] - region[2])
                * (region[5] - region[4]);
      full_V
        = (CUBE(mblock.extent()[1]) - CUBE(mblock.extent()[0])) * (4.0 / 3.0) * constant::PI;
    }
#endif
    ncells              = (std::size_t)((real_t)ncells * delta_V / full_V);

    auto npart_per_spec = (std::size_t)((double)(ncells * ppc_per_spec));
    list_t<real_t, 2 * static_cast<short>(D)> box { ZERO };
    for (auto i { 0 }; i < 2 * static_cast<short>(D); ++i) {
      box[i] = region[i];
    }

    for (auto& s : species) {
      auto& sp           = mblock.particles[s - 1];
      auto  npart_before = sp.npart();
      sp.setNpart(npart_before + npart_per_spec);
      if constexpr (D == Dim1) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({ 0 }, { npart_per_spec }),
          UniformInjector1d_kernel<S, EnDist>(params, mblock, sp, npart_before, box, time));
      } else if constexpr (D == Dim2) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({ 0 }, { npart_per_spec }),
          UniformInjector2d_kernel<S, EnDist>(params, mblock, sp, npart_before, box, time));
      } else if constexpr (D == Dim3) {
        Kokkos::parallel_for(
          "inject",
          CreateRangePolicy<Dim1>({ 0 }, { npart_per_spec }),
          UniformInjector3d_kernel<S, EnDist>(params, mblock, sp, npart_before, box, time));
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
                            const Meshblock<Dim1, S>&   mb,
                            const Particles<Dim1, S>&   sp1,
                            const Particles<Dim1, S>&   sp2,
                            const array_t<std::size_t>& ind,
                            const real_t&               ppc,
                            const real_t&               time)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        index { ind },
        nppc { ppc },
        energy_dist { params, mblock },
        spatial_dist { params, mblock },
        inj_criterion { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t i1) const {
      // cell node
      coord_t<Dim1> xi { static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS) };
      // cell center
      coord_t<Dim1> xc { xi[0] + HALF };
      // physical coordinate
      coord_t<Dim1> xph { ZERO };

#ifdef MINKOWSKI_METRIC
      mblock.metric.x_Code2Cart(xc, xph);
#else
      mblock.metric.x_Code2Sph(xc, xph);
#endif

      if (inj_criterion(xph)) {
        typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();
        real_t                                      ninject  = nppc * spatial_dist(xph);
        while (ninject > ZERO) {
          real_t random = rand_gen.frand();
          if (random < ninject) {
            vec_t<Dim3> v { ZERO };
            energy_dist(xph, v);

            real_t dx1                = rand_gen.frand();
            real_t dx2                = rand_gen.frand();

            auto   p                  = Kokkos::atomic_fetch_add(&index(), 1);
            species1.i1(offset1 + p)  = static_cast<int>(i1) - N_GHOSTS;
            species1.dx1(offset1 + p) = dx1;
            species1.ux1(offset1 + p) = v[0];
            species1.ux2(offset1 + p) = v[1];
            species1.ux3(offset1 + p) = v[2];

            species2.i1(offset2 + p)  = static_cast<int>(i1) - N_GHOSTS;
            species2.dx1(offset2 + p) = dx1;
            species2.ux1(offset2 + p) = v[0];
            species2.ux2(offset2 + p) = v[1];
            species2.ux3(offset2 + p) = v[2];
          }
          ninject -= ONE;
        }
        pool.free_state(rand_gen);
      }
    }

  private:
    SimulationParams     params;
    Meshblock<Dim1, S>   mblock;
    Particles<Dim1, S>   species1, species2;
    const std::size_t    offset1, offset2;
    array_t<std::size_t> index;
    const real_t         nppc;
    EnDist<Dim1, S>      energy_dist;
    SpDist<Dim1, S>      spatial_dist;
    InjCrit<Dim1, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  template <SimulationType S,
            template <Dimension, SimulationType>
            class EnDist,
            template <Dimension, SimulationType>
            class SpDist,
            template <Dimension, SimulationType>
            class InjCrit>
  struct VolumeInjector2d_kernel {
    VolumeInjector2d_kernel(const SimulationParams&     pr,
                            const Meshblock<Dim2, S>&   mb,
                            const Particles<Dim2, S>&   sp1,
                            const Particles<Dim2, S>&   sp2,
                            const array_t<std::size_t>& ind,
                            const real_t&               ppc,
                            const real_t&               time)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        index { ind },
        nppc { ppc },
        energy_dist { params, mblock },
        spatial_dist { params, mblock },
        inj_criterion { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t i1, index_t i2) const {
      // cell node
      coord_t<Dim2> xi { static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS),
                         static_cast<real_t>(static_cast<int>(i2) - N_GHOSTS) };
      // cell center
      coord_t<Dim2> xc { xi[0] + HALF, xi[1] + HALF };
      // physical coordinate
      coord_t<Dim2> xph { ZERO };

#ifdef MINKOWSKI_METRIC
      mblock.metric.x_Code2Cart(xc, xph);
#else
      mblock.metric.x_Code2Sph(xc, xph);
#endif

      if (inj_criterion(xph)) {
        typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();
        real_t                                      ninject  = nppc * spatial_dist(xph);
        while (ninject > ZERO) {
          real_t random = rand_gen.frand();
          if (random < ninject) {
            vec_t<Dim3> v { ZERO }, v_cart { ZERO };
            energy_dist(xph, v);
#ifdef MINKOWSKI_METRIC
            v_cart[0] = v[0];
            v_cart[1] = v[1];
            v_cart[2] = v[2];
#else
            mblock.metric.v_Hat2Cart({ xc[0], xc[1], ZERO }, v, v_cart);
#endif
            real_t dx1                = rand_gen.frand();
            real_t dx2                = rand_gen.frand();

            auto   p                  = Kokkos::atomic_fetch_add(&index(), 1);
            species1.i1(offset1 + p)  = static_cast<int>(i1) - N_GHOSTS;
            species1.dx1(offset1 + p) = dx1;
            species1.i2(offset1 + p)  = static_cast<int>(i2) - N_GHOSTS;
            species1.dx2(offset1 + p) = dx2;
            species1.ux1(offset1 + p) = v_cart[0];
            species1.ux2(offset1 + p) = v_cart[1];
            species1.ux3(offset1 + p) = v_cart[2];

            species2.i1(offset2 + p)  = static_cast<int>(i1) - N_GHOSTS;
            species2.dx1(offset2 + p) = dx1;
            species2.i2(offset2 + p)  = static_cast<int>(i2) - N_GHOSTS;
            species2.dx2(offset2 + p) = dx2;
            species2.ux1(offset2 + p) = v_cart[0];
            species2.ux2(offset2 + p) = v_cart[1];
            species2.ux3(offset2 + p) = v_cart[2];
          }
          ninject -= ONE;
        }
        pool.free_state(rand_gen);
      }
    }

  private:
    SimulationParams     params;
    Meshblock<Dim2, S>   mblock;
    Particles<Dim2, S>   species1, species2;
    const std::size_t    offset1, offset2;
    array_t<std::size_t> index;
    const real_t         nppc;
    EnDist<Dim2, S>      energy_dist;
    SpDist<Dim2, S>      spatial_dist;
    InjCrit<Dim2, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  template <SimulationType S,
            template <Dimension, SimulationType>
            class EnDist,
            template <Dimension, SimulationType>
            class SpDist,
            template <Dimension, SimulationType>
            class InjCrit>
  struct VolumeInjector3d_kernel {
    VolumeInjector3d_kernel(const SimulationParams&     pr,
                            const Meshblock<Dim3, S>&   mb,
                            const Particles<Dim3, S>&   sp1,
                            const Particles<Dim3, S>&   sp2,
                            const array_t<std::size_t>& ind,
                            const real_t&               ppc,
                            const real_t&               time)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        index { ind },
        nppc { ppc },
        energy_dist { params, mblock },
        spatial_dist { params, mblock },
        inj_criterion { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      // cell node
      coord_t<Dim3> xi { static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS),
                         static_cast<real_t>(static_cast<int>(i2) - N_GHOSTS),
                         static_cast<real_t>(static_cast<int>(i3) - N_GHOSTS) };
      // cell center
      coord_t<Dim3> xc { xi[0] + HALF, xi[1] + HALF, xi[2] + HALF };
      // physical coordinate
      coord_t<Dim3> xph { ZERO };

#ifdef MINKOWSKI_METRIC
      mblock.metric.x_Code2Cart(xc, xph);
#else
      mblock.metric.x_Code2Sph(xc, xph);
#endif

      if (inj_criterion(xph)) {
        typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

        real_t                                      ninject  = nppc * spatial_dist(xph);
        while (ninject > ZERO) {
          real_t random = rand_gen.frand();
          if (random < ninject) {
            vec_t<Dim3> v { ZERO }, v_cart { ZERO };
            energy_dist(xph, v);
#ifdef MINKOWSKI_METRIC
            v_cart[0] = v[0];
            v_cart[1] = v[1];
            v_cart[2] = v[2];
#else
            mblock.metric.v_Hat2Cart({ xc[0], xc[1], ZERO }, v, v_cart);
#endif

            real_t dx1                = rand_gen.frand();
            real_t dx2                = rand_gen.frand();
            real_t dx3                = rand_gen.frand();

            auto   p                  = Kokkos::atomic_fetch_add(&index(), 1);
            species1.i1(offset1 + p)  = static_cast<int>(i1) - N_GHOSTS;
            species1.dx1(offset1 + p) = dx1;
            species1.i2(offset1 + p)  = static_cast<int>(i2) - N_GHOSTS;
            species1.dx2(offset1 + p) = dx2;
            species1.i3(offset1 + p)  = static_cast<int>(i3) - N_GHOSTS;
            species1.dx3(offset1 + p) = dx3;
            species1.ux1(offset1 + p) = v_cart[0];
            species1.ux2(offset1 + p) = v_cart[1];
            species1.ux3(offset1 + p) = v_cart[2];

            species2.i1(offset2 + p)  = static_cast<int>(i1) - N_GHOSTS;
            species2.dx1(offset2 + p) = dx1;
            species2.i2(offset2 + p)  = static_cast<int>(i2) - N_GHOSTS;
            species2.dx2(offset2 + p) = dx2;
            species2.i3(offset2 + p)  = static_cast<int>(i3) - N_GHOSTS;
            species2.dx3(offset2 + p) = dx3;
            species2.ux1(offset2 + p) = v_cart[0];
            species2.ux2(offset2 + p) = v_cart[1];
            species2.ux3(offset2 + p) = v_cart[2];
          }
          ninject -= ONE;
        }
        pool.free_state(rand_gen);
      }
    }

  private:
    SimulationParams     params;
    Meshblock<Dim3, S>   mblock;
    Particles<Dim3, S>   species1, species2;
    const std::size_t    offset1, offset2;
    array_t<std::size_t> index;
    const real_t         nppc;
    EnDist<Dim3, S>      energy_dist;
    SpDist<Dim3, S>      spatial_dist;
    InjCrit<Dim3, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  /**
   * @brief Particle injector parallelized by cells in a volume.
   * @tparam D dimension.
   * @tparam S simulation type.
   * @tparam EnDist energy distribution [default = ColdDist].
   * @tparam SpDist spatial distribution [default = UniformDist].
   * @tparam InjCrit injection criterion [default = NoCriterion].
   *
   * @param params simulation parameters.
   * @param mblock meshblock.
   * @param species species to inject as a list.
   * @param ppc_per_spec fiducial number of particles per cell per species.
   * @param region region to inject particles as a list of coordinates.
   * @param time current time.
   */
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
    range_t<D>    range_policy;
    if (region.size() == 0) {
      range_policy = mblock.rangeActiveCells();
    } else {
      range_policy = mblock.rangeActiveCells();
      NTTHostError("Non-trivial region not implemented yet");
    }

    if (species.size() != 2) {
      NTTHostError("Exactly two species can be injected at the same time");
    }
    auto& sp1 = mblock.particles[species[0] - 1];
    auto& sp2 = mblock.particles[species[1] - 1];
    if (sp1.charge() != -sp2.charge()) {
      NTTHostError("Injected species must have the same but opposite charge: q1 = -q2");
    }
    array_t<std::size_t> ind("ind_inj");
    if constexpr (D == Dim1) {
      Kokkos::parallel_for("inject",
                           range_policy,
                           VolumeInjector1d_kernel<S, EnDist, SpDist, InjCrit>(
                             params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for("inject",
                           range_policy,
                           VolumeInjector2d_kernel<S, EnDist, SpDist, InjCrit>(
                             params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    } else if constexpr (D == Dim3) {
      Kokkos::parallel_for("inject",
                           range_policy,
                           VolumeInjector3d_kernel<S, EnDist, SpDist, InjCrit>(
                             params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    }

    auto ind_h = Kokkos::create_mirror(ind);
    Kokkos::deep_copy(ind_h, ind);
    sp1.setNpart(sp1.npart() + ind_h());
    sp2.setNpart(sp2.npart() + ind_h());
  }

}    // namespace ntt

#endif    // FRAMEWORK_INJECTOR_H