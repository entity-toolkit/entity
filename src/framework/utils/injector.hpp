#ifndef FRAMEWORK_INJECTOR_H
#define FRAMEWORK_INJECTOR_H

#include "wrapper.h"

#include "particle_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"
#include "meshblock/particles.h"

#include "utils/archetypes.hpp"

#include <vector>

namespace ntt {

  /* -------------------------------------------------------------------------- */
  /*                   Uniform injection kernels and routines                   */
  /* -------------------------------------------------------------------------- */

  /**
   * @brief 1D particle-vectorized injection kernel
   */
  template <SimulationEngine S, template <Dimension, SimulationEngine> class EnDist>
  struct UniformInjector1d_kernel {
    UniformInjector1d_kernel(const SimulationParams&   pr,
                             const Meshblock<Dim1, S>& mb,
                             const Particles<Dim1, S>& sp1,
                             const Particles<Dim1, S>& sp2,
                             const list_t<real_t, 2>&  box,
                             const real_t&)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        species_index1 { sp1.index() },
        species_index2 { sp2.index() },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        region { box[0], box[1] },
        energy_dist { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim1>                               x { ZERO };
      vec_t<Dim3>                                 v { ZERO };
      x[0] = rand_gen.frand(region[0], region[1]);
      energy_dist(x, v, species_index1);
      init_prtl_1d(mblock, species1, p + offset1, x[0], v[0], v[1], v[2], ONE);
      energy_dist(x, v, species_index2);
      init_prtl_1d(mblock, species2, p + offset2, x[0], v[0], v[1], v[2], ONE);
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams   params;
    Meshblock<Dim1, S> mblock;
    Particles<Dim1, S> species1, species2;
    const int          species_index1, species_index2;
    const std::size_t  offset1, offset2;
    EnDist<Dim1, S>    energy_dist;
    list_t<real_t, 2>  region;
    RandomNumberPool_t pool;
  };

  /**
   * @brief 2D particle-vectorized injection kernel
   */
  template <SimulationEngine S, template <Dimension, SimulationEngine> class EnDist>
  struct UniformInjector2d_kernel {
    UniformInjector2d_kernel(const SimulationParams&   pr,
                             const Meshblock<Dim2, S>& mb,
                             const Particles<Dim2, S>& sp1,
                             const Particles<Dim2, S>& sp2,
                             const list_t<real_t, 4>&  box,
                             const real_t&)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        species_index1 { sp1.index() },
        species_index2 { sp2.index() },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        region { box[0], box[1], box[2], box[3] },
        energy_dist { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim2>                               x { ZERO };
      vec_t<Dim3>                                 v { ZERO };
      x[0] = rand_gen.frand(region[0], region[1]);
      x[1] = rand_gen.frand(region[2], region[3]);
      energy_dist(x, v, species_index1);
      init_prtl_2d(mblock, species1, p + offset1, x[0], x[1], v[0], v[1], v[2], ONE);
      energy_dist(x, v, species_index2);
      init_prtl_2d(mblock, species2, p + offset2, x[0], x[1], v[0], v[1], v[2], ONE);
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams   params;
    Meshblock<Dim2, S> mblock;
    Particles<Dim2, S> species1, species2;
    const int          species_index1, species_index2;
    const std::size_t  offset1, offset2;
    EnDist<Dim2, S>    energy_dist;
    list_t<real_t, 4>  region;
    RandomNumberPool_t pool;
  };

  /**
   * @brief 3D particle-vectorized injection kernel
   */
  template <SimulationEngine S, template <Dimension, SimulationEngine> class EnDist>
  struct UniformInjector3d_kernel {
    UniformInjector3d_kernel(const SimulationParams&   pr,
                             const Meshblock<Dim3, S>& mb,
                             const Particles<Dim3, S>& sp1,
                             const Particles<Dim3, S>& sp2,
                             const list_t<real_t, 6>&  box,
                             const real_t&)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        species_index1 { sp1.index() },
        species_index2 { sp2.index() },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        region { box[0], box[1], box[2], box[3], box[4], box[5] },
        energy_dist { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t p) const {
      typename RandomNumberPool_t::generator_type rand_gen = pool.get_state();

      coord_t<Dim3>                               x { ZERO };
      vec_t<Dim3>                                 v { ZERO };
      x[0] = rand_gen.frand(region[0], region[1]);
      x[1] = rand_gen.frand(region[2], region[3]);
      x[2] = rand_gen.frand(region[4], region[5]);
      energy_dist(x, v, species_index1);
      init_prtl_3d(mblock, species1, p + offset1, x[0], x[1], x[2], v[0], v[1], v[2], ONE);
      energy_dist(x, v, species_index2);
      init_prtl_3d(mblock, species2, p + offset2, x[0], x[1], x[2], v[0], v[1], v[2], ONE);
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams   params;
    Meshblock<Dim3, S> mblock;
    Particles<Dim3, S> species1, species2;
    const int          species_index1, species_index2;
    const std::size_t  offset1, offset2;
    EnDist<Dim3, S>    energy_dist;
    list_t<real_t, 6>  region;
    RandomNumberPool_t pool;
  };

  /**
   * @brief Volumetrically uniform particle injector parallelized over particles.
   * @tparam D dimension.
   * @tparam S simulation engine.
   * @tparam EnDist energy distribution [default = ColdDist].
   *
   * @param params simulation parameters.
   * @param mblock meshblock.
   * @param species species to inject as a list.
   * @param ppc_per_spec fiducial number of particles per cell per species.
   * @param region region to inject particles as a list of coordinates [optional].
   * @param time current time [optional].
   */
  template <Dimension        D,
            SimulationEngine S,
            template <Dimension, SimulationEngine> class EnDist = ColdDist>
  inline void InjectUniform(const SimulationParams& params,
                            Meshblock<D, S>&        mblock,
                            const std::vector<int>& species,
                            const real_t&           ppc_per_spec,
                            std::vector<real_t>     region = {},
                            const real_t&           time   = ZERO) {
    NTTHostErrorIf(species.size() != 2, "Exactly two species can be injected at the same time");
    auto& sp1 = mblock.particles[species[0] - 1];
    auto& sp2 = mblock.particles[species[1] - 1];
    NTTHostErrorIf(sp1.charge() != -sp2.charge(),
                   "Injected species must have the same but opposite charge: q1 = -q2");
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

    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "InjectUniform",
        CreateRangePolicy<Dim1>({ 0 }, { npart_per_spec }),
        UniformInjector1d_kernel<S, EnDist>(params, mblock, sp1, sp2, box, time));
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for(
        "InjectUniform",
        CreateRangePolicy<Dim1>({ 0 }, { npart_per_spec }),
        UniformInjector2d_kernel<S, EnDist>(params, mblock, sp1, sp2, box, time));
    } else if constexpr (D == Dim3) {
      Kokkos::parallel_for(
        "InjectUniform",
        CreateRangePolicy<Dim1>({ 0 }, { npart_per_spec }),
        UniformInjector3d_kernel<S, EnDist>(params, mblock, sp1, sp2, box, time));
    }
    sp1.setNpart(sp1.npart() + npart_per_spec);
    sp2.setNpart(sp2.npart() + npart_per_spec);
  }

  /* -------------------------------------------------------------------------- */
  /*                    Volume injection kernels and routines                   */
  /* -------------------------------------------------------------------------- */
  template <SimulationEngine S,
            template <Dimension, SimulationEngine>
            class EnDist,
            template <Dimension, SimulationEngine>
            class SpDist,
            template <Dimension, SimulationEngine>
            class InjCrit>
  struct VolumeInjector1d_kernel {
    VolumeInjector1d_kernel(const SimulationParams&     pr,
                            const Meshblock<Dim1, S>&   mb,
                            const Particles<Dim1, S>&   sp1,
                            const Particles<Dim1, S>&   sp2,
                            const array_t<std::size_t>& ind,
                            const real_t&               ppc,
                            const real_t&)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        species_index1 { sp1.index() },
        species_index2 { sp2.index() },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        index { ind },
        nppc { ppc },
        use_weights { params.useWeights() },
        V0 { params.V0() },
        energy_dist { params, mblock },
        spatial_dist { params, mblock },
        inj_criterion { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t i1) const {
      // cell node
      coord_t<Dim1>     xi { static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS) };
      RandomGenerator_t rand_gen { pool.get_state() };
      real_t            n_inject { nppc };
      coord_t<Dim1>     xc { ZERO };
      coord_t<Dim1>     xph { ZERO };
      prtldx_t          dx1, dx2;
      vec_t<Dim3>       v { ZERO }, v_cart { ZERO };
      real_t            cell_vol;

      while (n_inject > ZERO) {
        dx1   = Random<prtldx_t>(rand_gen);
        xc[0] = xi[0] + dx1;
        mblock.metric.x_Code2Cart(xc, xph);
        if ((Random<real_t>(rand_gen) < n_inject) &&          // # of prtls
            inj_criterion(xph) &&                             // injection criterion
            (Random<real_t>(rand_gen) < spatial_dist(xph))    // spatial distribution
        ) {
          auto       p { Kokkos::atomic_fetch_add(&index(), 1) };
          const auto weight { use_weights ? (mblock.metric.sqrt_det_h(xc) / V0) : ONE };

          energy_dist(xph, v, species_index1);
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
          init_prtl_1d_i_di(species1,
                            offset1 + p,
                            static_cast<int>(i1) - N_GHOSTS,
                            dx1,
                            v_cart[0],
                            v_cart[1],
                            v_cart[2],
                            weight);

          energy_dist(xph, v, species_index2);
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
          init_prtl_1d_i_di(species2,
                            offset2 + p,
                            static_cast<int>(i1) - N_GHOSTS,
                            dx1,
                            v_cart[0],
                            v_cart[1],
                            v_cart[2],
                            weight);
        }
        n_inject -= ONE;
      }
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams     params;
    Meshblock<Dim1, S>   mblock;
    Particles<Dim1, S>   species1, species2;
    const int            species_index1, species_index2;
    const std::size_t    offset1, offset2;
    array_t<std::size_t> index;
    const real_t         nppc;
    const bool           use_weights;
    const real_t         V0;
    EnDist<Dim1, S>      energy_dist;
    SpDist<Dim1, S>      spatial_dist;
    InjCrit<Dim1, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  template <SimulationEngine S,
            template <Dimension, SimulationEngine>
            class EnDist,
            template <Dimension, SimulationEngine>
            class SpDist,
            template <Dimension, SimulationEngine>
            class InjCrit>
  struct VolumeInjector2d_kernel {
    VolumeInjector2d_kernel(const SimulationParams&     pr,
                            const Meshblock<Dim2, S>&   mb,
                            const Particles<Dim2, S>&   sp1,
                            const Particles<Dim2, S>&   sp2,
                            const array_t<std::size_t>& ind,
                            const real_t&               ppc,
                            const real_t&)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        species_index1 { sp1.index() },
        species_index2 { sp2.index() },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        index { ind },
        nppc { ppc },
        use_weights { params.useWeights() },
        V0 { params.V0() },
        energy_dist { params, mblock },
        spatial_dist { params, mblock },
        inj_criterion { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t i1, index_t i2) const {
      // cell node
      coord_t<Dim2>     xi { static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS),
                         static_cast<real_t>(static_cast<int>(i2) - N_GHOSTS) };
      RandomGenerator_t rand_gen { pool.get_state() };
      real_t            n_inject { nppc };
      coord_t<Dim2>     xc { ZERO };
      coord_t<Dim2>     xph { ZERO };
      prtldx_t          dx1, dx2;
      vec_t<Dim3>       v { ZERO }, v_cart { ZERO };
      real_t            cell_vol;

      while (n_inject > ZERO) {
        dx1   = Random<prtldx_t>(rand_gen);
        dx2   = Random<prtldx_t>(rand_gen);
        xc[0] = xi[0] + dx1;
        xc[1] = xi[1] + dx2;
#ifdef MINKOWSKI_METRIC
        mblock.metric.x_Code2Cart(xc, xph);
#else
        mblock.metric.x_Code2Sph(xc, xph);
#endif
        if ((Random<real_t>(rand_gen) < n_inject) &&          // # of prtls
            inj_criterion(xph) &&                             // injection criterion
            (Random<real_t>(rand_gen) < spatial_dist(xph))    // spatial distribution
        ) {
          auto       p { Kokkos::atomic_fetch_add(&index(), 1) };
          const auto weight { use_weights ? (mblock.metric.sqrt_det_h(xc) / V0) : ONE };

          energy_dist(xph, v, species_index1);
#ifdef MINKOWSKI_METRIC
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
#else
          mblock.metric.v3_Hat2Cart({ xc[0], xc[1], ZERO }, v, v_cart);
#endif
          init_prtl_2d_i_di(species1,
                            offset1 + p,
                            static_cast<int>(i1) - N_GHOSTS,
                            static_cast<int>(i2) - N_GHOSTS,
                            dx1,
                            dx2,
                            v_cart[0],
                            v_cart[1],
                            v_cart[2],
                            weight);

          energy_dist(xph, v, species_index2);
#ifdef MINKOWSKI_METRIC
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
#else
          mblock.metric.v3_Hat2Cart({ xc[0], xc[1], ZERO }, v, v_cart);
#endif
          init_prtl_2d_i_di(species2,
                            offset2 + p,
                            static_cast<int>(i1) - N_GHOSTS,
                            static_cast<int>(i2) - N_GHOSTS,
                            dx1,
                            dx2,
                            v_cart[0],
                            v_cart[1],
                            v_cart[2],
                            weight);
        }
        n_inject -= ONE;
      }
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams     params;
    Meshblock<Dim2, S>   mblock;
    Particles<Dim2, S>   species1, species2;
    const int            species_index1, species_index2;
    const std::size_t    offset1, offset2;
    array_t<std::size_t> index;
    const real_t         nppc;
    const bool           use_weights;
    const real_t         V0;
    EnDist<Dim2, S>      energy_dist;
    SpDist<Dim2, S>      spatial_dist;
    InjCrit<Dim2, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  template <SimulationEngine S,
            template <Dimension, SimulationEngine>
            class EnDist,
            template <Dimension, SimulationEngine>
            class SpDist,
            template <Dimension, SimulationEngine>
            class InjCrit>
  struct VolumeInjector3d_kernel {
    VolumeInjector3d_kernel(const SimulationParams&     pr,
                            const Meshblock<Dim3, S>&   mb,
                            const Particles<Dim3, S>&   sp1,
                            const Particles<Dim3, S>&   sp2,
                            const array_t<std::size_t>& ind,
                            const real_t&               ppc,
                            const real_t&)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        species_index1 { sp1.index() },
        species_index2 { sp2.index() },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        index { ind },
        nppc { ppc },
        use_weights { params.useWeights() },
        V0 { params.V0() },
        energy_dist { params, mblock },
        spatial_dist { params, mblock },
        inj_criterion { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      // cell node
      coord_t<Dim3>     xi { static_cast<real_t>(static_cast<int>(i1) - N_GHOSTS),
                         static_cast<real_t>(static_cast<int>(i2) - N_GHOSTS),
                         static_cast<real_t>(static_cast<int>(i3) - N_GHOSTS) };
      RandomGenerator_t rand_gen { pool.get_state() };
      real_t            n_inject { nppc };
      coord_t<Dim3>     xc { ZERO };
      coord_t<Dim3>     xph { ZERO };
      prtldx_t          dx1, dx2, dx3;
      vec_t<Dim3>       v { ZERO }, v_cart { ZERO };
      real_t            cell_vol;

      while (n_inject > ZERO) {
        dx1   = Random<prtldx_t>(rand_gen);
        dx2   = Random<prtldx_t>(rand_gen);
        dx3   = Random<prtldx_t>(rand_gen);
        xc[0] = xi[0] + dx1;
        xc[1] = xi[1] + dx2;
        xc[2] = xi[2] + dx3;
#ifdef MINKOWSKI_METRIC
        mblock.metric.x_Code2Cart(xc, xph);
#else
        mblock.metric.x_Code2Sph(xc, xph);
#endif
        if ((Random<real_t>(rand_gen) < n_inject) &&          // # of prtls
            inj_criterion(xph) &&                             // injection criterion
            (Random<real_t>(rand_gen) < spatial_dist(xph))    // spatial distribution
        ) {
          auto       p { Kokkos::atomic_fetch_add(&index(), 1) };
          const auto weight { use_weights ? (mblock.metric.sqrt_det_h(xc) / V0) : ONE };

          energy_dist(xph, v, species_index1);
#ifdef MINKOWSKI_METRIC
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
#else
          mblock.metric.v3_Hat2Cart({ xc[0], xc[1], xc[2] }, v, v_cart);
#endif
          init_prtl_3d_i_di(species1,
                            offset1 + p,
                            static_cast<int>(i1) - N_GHOSTS,
                            static_cast<int>(i2) - N_GHOSTS,
                            static_cast<int>(i3) - N_GHOSTS,
                            dx1,
                            dx2,
                            dx3,
                            v_cart[0],
                            v_cart[1],
                            v_cart[2],
                            weight);

          energy_dist(xph, v, species_index2);
#ifdef MINKOWSKI_METRIC
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
#else
          mblock.metric.v3_Hat2Cart({ xc[0], xc[1], xc[2] }, v, v_cart);
#endif
          init_prtl_3d_i_di(species2,
                            offset2 + p,
                            static_cast<int>(i1) - N_GHOSTS,
                            static_cast<int>(i2) - N_GHOSTS,
                            static_cast<int>(i3) - N_GHOSTS,
                            dx1,
                            dx2,
                            dx3,
                            v_cart[0],
                            v_cart[1],
                            v_cart[2],
                            weight);
        }
        n_inject -= ONE;
      }
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams     params;
    Meshblock<Dim3, S>   mblock;
    Particles<Dim3, S>   species1, species2;
    const int            species_index1, species_index2;
    const std::size_t    offset1, offset2;
    array_t<std::size_t> index;
    const real_t         nppc;
    const bool           use_weights;
    const real_t         V0;
    EnDist<Dim3, S>      energy_dist;
    SpDist<Dim3, S>      spatial_dist;
    InjCrit<Dim3, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  /**
   * @brief Particle injector parallelized by cells in a volume.
   * @tparam D dimension.
   * @tparam S simulation engine.
   * @tparam EnDist energy distribution [default = ColdDist].
   * @tparam SpDist spatial distribution [default = UniformDist].
   * @tparam InjCrit injection criterion [default = NoCriterion].
   *
   * @param params simulation parameters.
   * @param mblock meshblock.
   * @param species species to inject as a list.
   * @param ppc_per_spec fiducial number of particles per cell per species.
   * @param region region to inject particles as a list of coordinates [optional].
   * @param time current time [optional].
   */
  template <Dimension        D,
            SimulationEngine S,
            template <Dimension, SimulationEngine> class EnDist  = ColdDist,
            template <Dimension, SimulationEngine> class SpDist  = UniformDist,
            template <Dimension, SimulationEngine> class InjCrit = NoCriterion>
  inline void InjectInVolume(const SimulationParams& params,
                             Meshblock<D, S>&        mblock,
                             const std::vector<int>& species,
                             const real_t&           ppc_per_spec,
                             std::vector<real_t>     region = {},
                             const real_t&           time   = ZERO) {
    range_t<D> range_policy;
    if (region.size() == 0) {
      range_policy = mblock.rangeActiveCells();
    } else if (region.size() == 2 * static_cast<short>(D)) {
      tuple_t<std::size_t, D> region_min;
      tuple_t<std::size_t, D> region_max;
      coord_t<D>              xmin_ph { ZERO }, xmax_ph { ZERO };
      coord_t<D>              xmin_cu { ZERO }, xmax_cu { ZERO };
      for (short i = 0; i < static_cast<short>(D); ++i) {
        xmin_ph[i] = region[2 * i];
        xmax_ph[i] = region[2 * i + 1];
      }
#ifdef MINKOWSKI_METRIC
      mblock.metric.x_Cart2Code(xmin_ph, xmin_cu);
      mblock.metric.x_Cart2Code(xmax_ph, xmax_cu);
#else
      mblock.metric.x_Sph2Code(xmin_ph, xmin_cu);
      mblock.metric.x_Sph2Code(xmax_ph, xmax_cu);
#endif
      for (short i = 0; i < static_cast<short>(D); ++i) {
        region_min[i] = static_cast<std::size_t>(xmin_cu[i]);
        region_max[i] = static_cast<std::size_t>(xmax_cu[i]);
      }
      range_policy = CreateRangePolicy<D>(region_min, region_max);
    } else {
      NTTHostError("region must be empty or have 2 * D elements");
    }

    NTTHostErrorIf(species.size() != 2, "Exactly two species can be injected at the same time");
    auto& sp1 = mblock.particles[species[0] - 1];
    auto& sp2 = mblock.particles[species[1] - 1];
    NTTHostErrorIf(sp1.charge() != -sp2.charge(),
                   "Injected species must have the same but opposite charge: q1 = -q2");
    array_t<std::size_t> ind("ind_inj");
    if constexpr (D == Dim1) {
      Kokkos::parallel_for("InjectInVolume",
                           range_policy,
                           VolumeInjector1d_kernel<S, EnDist, SpDist, InjCrit>(
                             params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for("InjectInVolume",
                           range_policy,
                           VolumeInjector2d_kernel<S, EnDist, SpDist, InjCrit>(
                             params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    } else if constexpr (D == Dim3) {
      Kokkos::parallel_for("InjectInVolume",
                           range_policy,
                           VolumeInjector3d_kernel<S, EnDist, SpDist, InjCrit>(
                             params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    }

    auto ind_h = Kokkos::create_mirror(ind);
    Kokkos::deep_copy(ind_h, ind);
    sp1.setNpart(sp1.npart() + ind_h());
    sp2.setNpart(sp2.npart() + ind_h());
  }

  /* -------------------------------------------------------------------------- */

  template <SimulationEngine S,
            template <Dimension, SimulationEngine>
            class EnDist,
            template <Dimension, SimulationEngine>
            class InjCrit>
  struct FloorInjector1d_kernel {
    FloorInjector1d_kernel(const SimulationParams&     pr,
                           const Meshblock<Dim1, S>&   mb,
                           const Particles<Dim1, S>&   sp1,
                           const Particles<Dim1, S>&   sp2,
                           const array_t<std::size_t>& ind,
                           const ndarray_t<1>&         ppc,
                           const real_t&)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        species_index1 { sp1.index() },
        species_index2 { sp2.index() },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        index { ind },
        ppc_per_spec { ppc },
        use_weights { params.useWeights() },
        energy_dist { params, mblock },
        inj_criterion { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t i1) const {
      // cell node
      const auto        i1_ = static_cast<int>(i1) - N_GHOSTS;
      const auto        xi  = coord_t<Dim1> { static_cast<real_t>(i1_) };

      RandomGenerator_t rand_gen { pool.get_state() };
      real_t            n_inject { ppc_per_spec(i1_) };
      coord_t<Dim1>     xc { ZERO };
      coord_t<Dim1>     xph { ZERO };
      prtldx_t          dx1;
      vec_t<Dim3>       v { ZERO }, v_cart { ZERO };

      while (n_inject > ZERO) {
        dx1   = Random<prtldx_t>(rand_gen);
        xc[0] = xi[0] + dx1;
        mblock.metric.x_Code2Phys(xc, xph);
        if ((Random<real_t>(rand_gen) < n_inject) &&    // # of prtls
            inj_criterion(xph)                          // injection criterion
        ) {
          auto p { Kokkos::atomic_fetch_add(&index(), 1) };

          energy_dist(xph, v, species_index1);
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
          init_prtl_1d_i_di(
            species1, offset1 + p, i1_, dx1, v_cart[0], v_cart[1], v_cart[2], ONE);

          energy_dist(xph, v, species_index2);
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
          init_prtl_1d_i_di(
            species2, offset2 + p, i1_, dx1, v_cart[0], v_cart[1], v_cart[2], ONE);
        }
        n_inject -= ONE;
      }
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams     params;
    Meshblock<Dim1, S>   mblock;
    Particles<Dim1, S>   species1, species2;
    const int            species_index1, species_index2;
    const std::size_t    offset1, offset2;
    array_t<std::size_t> index;
    ndarray_t<1>         ppc_per_spec;
    const bool           use_weights;
    EnDist<Dim1, S>      energy_dist;
    InjCrit<Dim1, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  template <SimulationEngine S,
            template <Dimension, SimulationEngine>
            class EnDist,
            template <Dimension, SimulationEngine>
            class InjCrit>
  struct FloorInjector2d_kernel {
    FloorInjector2d_kernel(const SimulationParams&     pr,
                           const Meshblock<Dim2, S>&   mb,
                           const Particles<Dim2, S>&   sp1,
                           const Particles<Dim2, S>&   sp2,
                           const array_t<std::size_t>& ind,
                           const ndarray_t<2>&         ppc,
                           const real_t&)
      : params { pr },
        mblock { mb },
        species1 { sp1 },
        species2 { sp2 },
        species_index1 { sp1.index() },
        species_index2 { sp2.index() },
        offset1 { sp1.npart() },
        offset2 { sp2.npart() },
        index { ind },
        ppc_per_spec { ppc },
        use_weights { params.useWeights() },
        V0 { params.V0() },
        energy_dist { params, mblock },
        inj_criterion { params, mblock },
        pool { *(mblock.random_pool_ptr) } {}
    Inline void operator()(index_t i1, index_t i2) const {
      // cell node
      const auto i1_ = static_cast<int>(i1) - N_GHOSTS;
      const auto i2_ = static_cast<int>(i2) - N_GHOSTS;
      const auto xi  = coord_t<Dim2> { static_cast<real_t>(i1_), static_cast<real_t>(i2_) };
      const auto weight { use_weights
                            ? (mblock.metric.sqrt_det_h({ xi[0] + HALF, xi[1] + HALF }) / V0)
                            : ONE };

      RandomGenerator_t rand_gen { pool.get_state() };
      real_t            n_inject { ppc_per_spec(i1_, i2_) };
      coord_t<Dim2>     xc { ZERO };
      coord_t<Dim2>     xph { ZERO };
      prtldx_t          dx1, dx2;
      vec_t<Dim3>       v { ZERO }, v_cart { ZERO };

      while (n_inject > ZERO) {
        dx1   = Random<prtldx_t>(rand_gen);
        dx2   = Random<prtldx_t>(rand_gen);
        xc[0] = xi[0] + dx1;
        xc[1] = xi[1] + dx2;
        mblock.metric.x_Code2Phys(xc, xph);
        if ((Random<real_t>(rand_gen) < n_inject) &&    // # of prtls
            inj_criterion(xph)                          // injection criterion
        ) {
          auto p { Kokkos::atomic_fetch_add(&index(), 1) };

          energy_dist(xph, v, species_index1);
#ifdef MINKOWSKI_METRIC
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
#else
          mblock.metric.v3_Hat2Cart({ xc[0], xc[1], xc[2] }, v, v_cart);
#endif
          init_prtl_2d_i_di(
            species1, offset1 + p, i1_, i2_, dx1, dx2, v_cart[0], v_cart[1], v_cart[2], weight);

          energy_dist(xph, v, species_index2);
#ifdef MINKOWSKI_METRIC
          v_cart[0] = v[0];
          v_cart[1] = v[1];
          v_cart[2] = v[2];
#else
          mblock.metric.v3_Hat2Cart({ xc[0], xc[1], xc[2] }, v, v_cart);
#endif
          init_prtl_2d_i_di(
            species2, offset2 + p, i1_, i2_, dx1, dx2, v_cart[0], v_cart[1], v_cart[2], weight);
        }
        n_inject -= ONE;
      }
      pool.free_state(rand_gen);
    }

  private:
    SimulationParams     params;
    Meshblock<Dim2, S>   mblock;
    Particles<Dim2, S>   species1, species2;
    const int            species_index1, species_index2;
    const std::size_t    offset1, offset2;
    array_t<std::size_t> index;
    ndarray_t<2>         ppc_per_spec;
    const bool           use_weights;
    const real_t         V0;
    EnDist<Dim2, S>      energy_dist;
    InjCrit<Dim2, S>     inj_criterion;
    RandomNumberPool_t   pool;
  };

  /**
   * @brief Particle injector parallelized by cells in a volume ...
   * @brief ... up to certain number density.
   * @tparam D dimension.
   * @tparam S simulation engine.
   * @tparam EnDist energy distribution [default = ColdDist].
   * @tparam InjCrit injection criterion [default = NoCriterion].
   *
   * @param params simulation parameters.
   * @param mblock meshblock.
   * @param species species to inject as a list.
   * @param ppc_per_spec target injection ppc per species.
   * @param region region to inject particles as a list of coordinates [optional].
   * @param time current time [optional].
   */
  template <Dimension        D,
            SimulationEngine S,
            template <Dimension, SimulationEngine> class EnDist  = ColdDist,
            template <Dimension, SimulationEngine> class InjCrit = NoCriterion>
  inline void InjectNonUniform(const SimulationParams&      params,
                               Meshblock<D, S>&             mblock,
                               const std::vector<int>&      species,
                               const ndarray_t<(short)(D)>& ppc_per_spec,
                               std::vector<real_t>          region = {},
                               const real_t&                time   = ZERO) {
    EnDist<D, S>  energy_dist(params, mblock);
    InjCrit<D, S> inj_criterion(params, mblock);
    range_t<D>    range_policy;
    if (region.size() == 0) {
      range_policy = mblock.rangeActiveCells();
    } else if (region.size() == 2 * static_cast<short>(D)) {
      tuple_t<std::size_t, D> region_min;
      tuple_t<std::size_t, D> region_max;
      coord_t<D>              xmin_ph { ZERO }, xmax_ph { ZERO };
      coord_t<D>              xmin_cu { ZERO }, xmax_cu { ZERO };
      for (short i = 0; i < static_cast<short>(D); ++i) {
        xmin_ph[i] = region[2 * i];
        xmax_ph[i] = region[2 * i + 1];
      }
      mblock.metric.x_Phys2Code(xmin_ph, xmin_cu);
      mblock.metric.x_Phys2Code(xmax_ph, xmax_cu);
      for (short i = 0; i < static_cast<short>(D); ++i) {
        region_min[i] = static_cast<std::size_t>(xmin_cu[i]);
        region_max[i] = static_cast<std::size_t>(xmax_cu[i]);
      }
      range_policy = CreateRangePolicy<D>(region_min, region_max);
    } else {
      NTTHostError("region must be empty or have 2 * D elements");
    }

    NTTHostErrorIf(species.size() != 2, "Exactly two species can be injected at the same time");
    auto& sp1 = mblock.particles[species[0] - 1];
    auto& sp2 = mblock.particles[species[1] - 1];
    NTTHostErrorIf(sp1.charge() != -sp2.charge(),
                   "Injected species must have the same but opposite charge: q1 = -q2");
    array_t<std::size_t> ind("ind_inj");
    if constexpr (D == Dim1) {
      Kokkos::parallel_for("InjectNonUniform",
                           range_policy,
                           FloorInjector1d_kernel<S, EnDist, InjCrit>(
                             params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for("InjectNonUniform",
                           range_policy,
                           FloorInjector2d_kernel<S, EnDist, InjCrit>(
                             params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    } else if constexpr (D == Dim3) {
      // Kokkos::parallel_for("inject",
      //                      range_policy,
      //                      VolumeInjector3d_kernel<S, EnDist, SpDist, InjCrit>(
      //                        params, mblock, sp1, sp2, ind, ppc_per_spec, time));
    }

    auto ind_h = Kokkos::create_mirror(ind);
    Kokkos::deep_copy(ind_h, ind);
    sp1.setNpart(sp1.npart() + ind_h());
    sp2.setNpart(sp2.npart() + ind_h());
  }

  /* -------------------------------------------------------------------------- */

  // #ifdef MINKOWSKI_METRIC
  //   template <Dimension D, SimulationEngine S>
  //   auto InjectParticleGlobally(Meshblock<D, S>&  mblock,
  //                               short             species_id,
  //                               const coord_t<D>& x,
  //                               const vec_t<D>&   u,
  //                               real_t            w = ONE) -> bool {
  //     auto                 in_bounds = true;
  //     coord_t<D>           Xi { ZERO };
  //     tuple_t<int, D>      I { 0 };
  //     tuple_t<prtldx_t, D> dI { (prtldx_t)0 };
  //     mblock.metric.x_Phys2Code(x, Xi);
  //     for (short d { 0 }; d < (short)D; ++d) {
  //       from_Xi_to_i_di(Xi[d], I[d], dI[d]);
  //       in_bounds = (in_bounds) && (I[d] >= 0) && (I[d] < mblock.Ni(d));
  //     }
  //     if (!in_bounds) {
  //       return false;
  //     }
  //     auto&      species = mblock.particles[species_id];
  //     const auto p0      = species.npart();

  //     if constexpr (D == Dim1) {
  //       Kokkos::parallel_for(
  //         "InjectParticleGlobally", 1, Lambda(index_t p) {
  //           init_prtl_1d(mblock, sp, p0 + p, Xi[0], u[0], u[1], u[2], w);
  //         });
  //     } else if constexpr (D == Dim2) {
  //       Kokkos::parallel_for(
  //         "InjectParticleGlobally", 1, Lambda(index_t p) {
  //           init_prtl_2d(mblock, sp, p0 + p, Xi[0], Xi[1], u[0], u[1], u[2], w);
  //         });
  //     } else if constexpr (D == Dim3) {
  //       Kokkos::parallel_for(
  //         "InjectParticleGlobally", 1, Lambda(index_t p) {
  //           init_prtl_3d(mblock, sp, p0 + p, Xi[0], Xi[1], Xi[2], u[0], u[1], u[2], w);
  //         });
  //     }
  //     species.setNpart(p0 + 1);
  //     return true;
  //   }
  // #else    // not MINKOWSKI_METRIC
  //   template <Dimension D, SimulationEngine S>
  //   auto InjectParticleGlobally(Meshblock<D, S>& mblock,
  //                               short species_id,
  //                               const coord_t<Dim3>& x,
  //                               const vec_t<Dim3>& u,
  //                               real_t w = ONE) -> bool {
  //     coord_t<D> Xi { ZERO };
  //     coord_t<D> X_ph { ZERO };
  //     for (short d { 0 }; d < (short)D; ++d) {
  //       X_ph[d] = x[d];
  //     }
  //     tuple_t<int, D> I { 0 };
  //     mblock.metric.x_Phys2Code(X_ph, Xi);
  //     printf("particle coords CU %f %f\n", Xi[0], Xi[1]);
  //     const auto dtheta = (mblock.metric.x2_max - mblock.metric.x2_min) / mblock.Ni2();
  //     std::cout << "TEST : " << (X_ph[1] - mblock.metric.x2_min) / dtheta << " " << Xi[1]
  //               << "\n";
  //     for (short d { 0 }; d < (short)D; ++d) {
  //       // from_Xi_to_i(Xi[d], I[d]);
  //       I[d] = static_cast<int>(math::floor(Xi[d]));
  //       if (!((I[d] >= 0) && (I[d] < mblock.Ni(d)))) {
  //         return false;
  //       }
  //     }
  //     printf("  going to inject particle at %d %d %f %f\n", I[0], I[1], X_ph[0], X_ph[1]);
  //     printf("  bounds are %f %f %f %f %d %d\n",
  //            mblock.metric.x1_min,
  //            mblock.metric.x1_max,
  //            mblock.metric.x2_min,
  //            mblock.metric.x2_max,
  //            mblock.Ni1(),
  //            mblock.Ni2());
  //     auto& species = mblock.particles[species_id];
  //     const auto p0 = species.npart();

  //     if constexpr (D == Dim2) {
  //       Kokkos::parallel_for(
  //         "InjectParticleGlobally", 1, Lambda(index_t p) {
  //           init_prtl_2d(mblock, species, p0 + p, Xi[0], Xi[1], u[0], u[1], u[2], w);
  //           species.phi(p) = Xi[2];
  //         });
  //     } else if constexpr (D == Dim3) {
  //       NTTHostError("not implemented");
  //       return false;
  //     }
  //     species.setNpart(p0 + 1);
  //     return true;
  //   }
  // #endif

}    // namespace ntt

#endif    // FRAMEWORK_INJECTOR_H