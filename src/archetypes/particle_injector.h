/**
 * @file archetypes/particle_injector.h
 * @brief Particle injector routines and classes
 * @implements
 *   - arch::UniformInjector<>
 *   - arch::NonUniformInjector<>
 *   - arch::AtmosphereInjector<>
 *   - arch::InjectUniform<> -> void
 *   - arch::InjectGlobally<> -> void
 *   - arch::InjectNonUniform<> -> void
 * @namespaces:
 *   - arch::
 */

#ifndef ARCHETYPES_PARTICLE_INJECTOR_H
#define ARCHETYPES_PARTICLE_INJECTOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include "kernels/injectors.hpp"
#include "kernels/particle_moments.hpp"
#include "kernels/utils.hpp"

#include <Kokkos_Core.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace arch {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct BaseInjector {
    virtual auto DeduceRegion(const Domain<S, M>&         domain,
                              const boundaries_t<real_t>& box) const
      -> std::tuple<bool, array_t<real_t*>, array_t<real_t*>> {
      if (not domain.mesh.Intersects(box)) {
        return { false, array_t<real_t*> {}, array_t<real_t*> {} };
      }
      coord_t<M::Dim> xCorner_min_Ph { ZERO };
      coord_t<M::Dim> xCorner_max_Ph { ZERO };
      coord_t<M::Dim> xCorner_min_Cd { ZERO };
      coord_t<M::Dim> xCorner_max_Cd { ZERO };

      for (auto d { 0u }; d < M::Dim; ++d) {
        const auto local_xi_min = domain.mesh.extent(static_cast<in>(d)).first;
        const auto local_xi_max = domain.mesh.extent(static_cast<in>(d)).second;
        const auto extent_min   = std::min(std::max(local_xi_min, box[d].first),
                                         local_xi_max);
        const auto extent_max = std::max(std::min(local_xi_max, box[d].second),
                                         local_xi_min);
        xCorner_min_Ph[d]     = extent_min;
        xCorner_max_Ph[d]     = extent_max;
      }
      domain.mesh.metric.template convert<Crd::Ph, Crd::Cd>(xCorner_min_Ph,
                                                            xCorner_min_Cd);
      domain.mesh.metric.template convert<Crd::Ph, Crd::Cd>(xCorner_max_Ph,
                                                            xCorner_max_Cd);

      array_t<real_t*> xi_min { "xi_min", M::Dim }, xi_max { "xi_max", M::Dim };

      auto xi_min_h = Kokkos::create_mirror_view(xi_min);
      auto xi_max_h = Kokkos::create_mirror_view(xi_max);
      for (auto d { 0u }; d < M::Dim; ++d) {
        xi_min_h(d) = xCorner_min_Cd[d];
        xi_max_h(d) = xCorner_max_Cd[d];
      }
      Kokkos::deep_copy(xi_min, xi_min_h);
      Kokkos::deep_copy(xi_max, xi_max_h);

      return { true, xi_min, xi_max };
    }

    virtual auto ComputeNumInject(const SimulationParams&     params,
                                  const Domain<S, M>&         domain,
                                  real_t                      number_density,
                                  const boundaries_t<real_t>& box) const
      -> std::tuple<bool, npart_t, array_t<real_t*>, array_t<real_t*>> {
      const auto result = DeduceRegion(domain, box);
      if (not std::get<0>(result)) {
        return { false, (npart_t)0, array_t<real_t*> {}, array_t<real_t*> {} };
      }
      const auto xi_min   = std::get<1>(result);
      const auto xi_max   = std::get<2>(result);
      auto       xi_min_h = Kokkos::create_mirror_view(xi_min);
      auto       xi_max_h = Kokkos::create_mirror_view(xi_max);
      Kokkos::deep_copy(xi_min_h, xi_min);
      Kokkos::deep_copy(xi_max_h, xi_max);

      long double num_cells { 1.0 };
      for (auto d { 0u }; d < M::Dim; ++d) {
        num_cells *= static_cast<long double>(xi_max_h(d)) -
                     static_cast<long double>(xi_min_h(d));
      }

      const auto ppc0       = params.template get<real_t>("particles.ppc0");
      const auto nparticles = static_cast<npart_t>(
        (long double)(ppc0 * number_density * 0.5) * num_cells);

      return { true, nparticles, xi_min, xi_max };
    }
  };

  template <SimEngine::type S, class M, template <SimEngine::type, class> class ED>
  struct UniformInjector : BaseInjector<S, M> {
    using energy_dist_t = ED<S, M>;
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(energy_dist_t::is_energy_dist,
                  "E must be an energy distribution class");
    static constexpr bool      is_uniform_injector { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    const energy_dist_t               energy_dist;
    const std::pair<spidx_t, spidx_t> species;

    UniformInjector(const energy_dist_t&               energy_dist,
                    const std::pair<spidx_t, spidx_t>& species)
      : energy_dist { energy_dist }
      , species { species } {}

    ~UniformInjector() = default;
  };

  template <SimEngine::type S, class M, template <SimEngine::type, class> class ED>
  struct KeepConstantInjector : UniformInjector<S, M, ED> {
    using energy_dist_t = ED<S, M>;
    using UniformInjector<S, M, ED>::D;
    using UniformInjector<S, M, ED>::C;

    const idx_t          density_buff_idx;
    boundaries_t<real_t> probe_box;

    KeepConstantInjector(const energy_dist_t&               energy_dist,
                         const std::pair<spidx_t, spidx_t>& species,
                         idx_t                              density_buff_idx,
                         boundaries_t<real_t>               box = {})
      : UniformInjector<S, M, ED> { energy_dist, species }
      , density_buff_idx { density_buff_idx } {
      for (auto d { 0u }; d < M::Dim; ++d) {
        if (d < box.size()) {
          probe_box.push_back({ box[d].first, box[d].second });
        } else {
          probe_box.push_back(Range::All);
        }
      }
    }

    ~KeepConstantInjector() = default;

    auto ComputeAvgDensity(const SimulationParams& params,
                           const Domain<S, M>&     domain) const -> real_t {
      const auto result       = this->DeduceRegion(domain, probe_box);
      const auto should_probe = std::get<0>(result);
      if (not should_probe) {
        return ZERO;
      }
      const auto xi_min_arr = std::get<1>(result);
      const auto xi_max_arr = std::get<2>(result);

      tuple_t<ncells_t, M::Dim> i_min { 0 };
      tuple_t<ncells_t, M::Dim> i_max { 0 };

      auto xi_min_h = Kokkos::create_mirror_view(xi_min_arr);
      auto xi_max_h = Kokkos::create_mirror_view(xi_max_arr);
      Kokkos::deep_copy(xi_min_h, xi_min_arr);
      Kokkos::deep_copy(xi_max_h, xi_max_arr);

      ncells_t num_cells = 1u;
      for (auto d { 0u }; d < M::Dim; ++d) {
        i_min[d]   = std::floor(xi_min_h(d)) + N_GHOSTS;
        i_max[d]   = std::ceil(xi_max_h(d)) + N_GHOSTS;
        num_cells *= (i_max[d] - i_min[d]);
      }

      real_t dens { ZERO };
      if (should_probe) {
        Kokkos::parallel_reduce(
          "AvgDensity",
          CreateRangePolicy<M::Dim>(i_min, i_max),
          kernel::ComputeSum_kernel<M::Dim, 3>(domain.fields.buff, density_buff_idx),
          dens);
      }
#if defined(MPI_ENABLED)
      real_t   tot_dens { ZERO };
      ncells_t tot_num_cells { 0 };
      MPI_Allreduce(&dens, &tot_dens, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&num_cells,
                    &tot_num_cells,
                    1,
                    mpi::get_type<ncells_t>(),
                    MPI_SUM,
                    MPI_COMM_WORLD);
      dens      = tot_dens;
      num_cells = tot_num_cells;
#endif
      if (num_cells > 0) {
        return dens / (real_t)(num_cells);
      } else {
        return ZERO;
      }
    }

    auto ComputeNumInject(const SimulationParams&     params,
                          const Domain<S, M>&         domain,
                          real_t                      number_density,
                          const boundaries_t<real_t>& box) const
      -> std::tuple<bool, npart_t, array_t<real_t*>, array_t<real_t*>> override {
      const auto computed_avg_density = ComputeAvgDensity(params, domain);

      const auto result = this->DeduceRegion(domain, box);
      if (not std::get<0>(result)) {
        return { false, (npart_t)0, array_t<real_t*> {}, array_t<real_t*> {} };
      }

      const auto xi_min   = std::get<1>(result);
      const auto xi_max   = std::get<2>(result);
      auto       xi_min_h = Kokkos::create_mirror_view(xi_min);
      auto       xi_max_h = Kokkos::create_mirror_view(xi_max);
      Kokkos::deep_copy(xi_min_h, xi_min);
      Kokkos::deep_copy(xi_max_h, xi_max);

      long double num_cells { 1.0 };
      for (auto d { 0u }; d < M::Dim; ++d) {
        num_cells *= static_cast<long double>(xi_max_h(d)) -
                     static_cast<long double>(xi_min_h(d));
      }

      const auto ppc0 = params.template get<real_t>("particles.ppc0");
      npart_t    nparticles { 0u };
      if (number_density > computed_avg_density) {
        nparticles = static_cast<npart_t>(
          (long double)(ppc0 * (number_density - computed_avg_density) * 0.5) *
          num_cells);
      }

      return { nparticles != 0u, nparticles, xi_min, xi_max };
    }
  };

  template <SimEngine::type S,
            class M,
            template <SimEngine::type, class> class ED,
            template <SimEngine::type, class> class SD>
  struct NonUniformInjector {
    using energy_dist_t  = ED<S, M>;
    using spatial_dist_t = SD<S, M>;
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(energy_dist_t::is_energy_dist,
                  "E must be an energy distribution class");
    static_assert(spatial_dist_t::is_spatial_dist,
                  "SD must be a spatial distribution class");
    static constexpr bool      is_nonuniform_injector { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    const energy_dist_t               energy_dist;
    const spatial_dist_t              spatial_dist;
    const std::pair<spidx_t, spidx_t> species;

    NonUniformInjector(const energy_dist_t&               energy_dist,
                       const spatial_dist_t&              spatial_dist,
                       const std::pair<spidx_t, spidx_t>& species)
      : energy_dist { energy_dist }
      , spatial_dist { spatial_dist }
      , species { species } {}

    ~NonUniformInjector() = default;
  };

  template <SimEngine::type S, class M, bool P, in O>
  struct AtmosphereInjector {
    struct TargetDensityProfile {
      const real_t nmax, height, xsurf, ds;

      TargetDensityProfile(real_t nmax, real_t height, real_t xsurf, real_t ds)
        : nmax { nmax }
        , height { height }
        , xsurf { xsurf }
        , ds { ds } {}

      Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
        if constexpr ((O == in::x1) or
                      (O == in::x2 and (M::Dim == Dim::_2D or M::Dim == Dim::_3D)) or
                      (O == in::x3 and M::Dim == Dim::_3D)) {
          const auto xi = x_Ph[static_cast<dim_t>(O)];
          if constexpr (P) {
            // + direction
            if (xi < xsurf - ds or xi >= xsurf) {
              return ZERO;
            } else {
              if constexpr (M::CoordType == Coord::Cart) {
                return nmax * math::exp(-(xsurf - xi) / height);
              } else {
                raise::KernelError(
                  HERE,
                  "Atmosphere in +x cannot be applied for non-cartesian");
                return ZERO;
              }
            }
          } else {
            // - direction
            if (xi < xsurf or xi >= xsurf + ds) {
              return ZERO;
            } else {
              if constexpr (M::CoordType == Coord::Cart) {
                return nmax * math::exp(-(xi - xsurf) / height);
              } else {
                return nmax * math::exp(-(xsurf / height) * (ONE - (xsurf / xi)));
              }
            }
          }
        } else {
          raise::KernelError(HERE, "Wrong direction");
          return ZERO;
        }
      }
    };

    using energy_dist_t  = Maxwellian<S, M>;
    using spatial_dist_t = Replenish<S, M, TargetDensityProfile>;
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr bool      is_nonuniform_injector { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    const energy_dist_t               energy_dist;
    const TargetDensityProfile        target_density;
    const spatial_dist_t              spatial_dist;
    const std::pair<spidx_t, spidx_t> species;

    AtmosphereInjector(const M&                           metric,
                       const ndfield_t<M::Dim, 6>&        density,
                       real_t                             nmax,
                       real_t                             height,
                       real_t                             xsurf,
                       real_t                             ds,
                       real_t                             T,
                       random_number_pool_t&              pool,
                       const std::pair<spidx_t, spidx_t>& species)
      : energy_dist { metric, pool, T }
      , target_density { nmax, height, xsurf, ds }
      , spatial_dist { metric, density, 0, target_density, nmax }
      , species { species } {}

    ~AtmosphereInjector() = default;
  };

  template <SimEngine::type S, class M, in O>
  struct MovingInjector {
    struct TargetDensityProfile {
      const real_t nmax, xinj, xdrift;

      TargetDensityProfile(real_t xinj, real_t xdrift, real_t nmax)
        : xinj { xinj }
        , xdrift { xdrift }
        , nmax { nmax } {}

      Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
        if constexpr ((O == in::x1) or
                      (O == in::x2 and (M::Dim == Dim::_2D or M::Dim == Dim::_3D)) or
                      (O == in::x3 and M::Dim == Dim::_3D)) {
          const auto xi = x_Ph[static_cast<dim_t>(O)];
          // + direction
          if (xi < xdrift or xi >= xinj) {
            return ZERO;
          } else {
            if constexpr (M::CoordType == Coord::Cart) {
              return nmax;
            } else {
              raise::KernelError(
                HERE,
                "Moving injector in +x cannot be applied for non-cartesian");
              return ZERO;
            }
          }
        } else {
          raise::KernelError(HERE, "Wrong direction");
          return ZERO;
        }
      }
    };

    using energy_dist_t  = Maxwellian<S, M>;
    using spatial_dist_t = Replenish<S, M, TargetDensityProfile>;
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr bool      is_nonuniform_injector { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    const energy_dist_t               energy_dist;
    const TargetDensityProfile        target_density;
    const spatial_dist_t              spatial_dist;
    const std::pair<spidx_t, spidx_t> species;

    MovingInjector(const M&                           metric,
                   const ndfield_t<M::Dim, 6>&        density,
                   const energy_dist_t&               energy_dist,
                   real_t                             xinj,
                   real_t                             xdrift,
                   real_t                             nmax,
                   const std::pair<spidx_t, spidx_t>& species)
      : energy_dist { energy_dist }
      , target_density { xinj, xdrift, nmax }
      , spatial_dist { metric, density, 0, target_density, nmax }
      , species { species } {}

    ~MovingInjector() = default;
  };

  /**
   * @brief Injects uniform number density of particles everywhere in the domain
   * @param domain Domain object
   * @param injector Uniform injector object
   * @param number_density Total number density (in units of n0)
   * @param use_weights Use weights
   * @param box Region to inject the particles in global coords
   * @tparam S Simulation engine type
   * @tparam M Metric type
   * @tparam I Injector type
   */
  template <SimEngine::type S, class M, class I>
  inline void InjectUniform(const SimulationParams&     params,
                            Domain<S, M>&               domain,
                            const I&                    injector,
                            real_t                      number_density,
                            bool                        use_weights = false,
                            const boundaries_t<real_t>& box         = {}) {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(I::is_uniform_injector, "I must be a uniform injector class");
    raise::ErrorIf((M::CoordType != Coord::Cart) && (not use_weights),
                   "Weights must be used for non-Cartesian coordinates",
                   HERE);
    raise::ErrorIf((M::CoordType == Coord::Cart) && use_weights,
                   "Weights should not be used for Cartesian coordinates",
                   HERE);
    raise::ErrorIf(params.template get<bool>("particles.use_weights") != use_weights,
                   "Weights must be enabled from the input file to use them in "
                   "the injector",
                   HERE);
    if (domain.species[injector.species.first - 1].charge() +
          domain.species[injector.species.second - 1].charge() !=
        0.0f) {
      raise::Warning("Total charge of the injected species is non-zero", HERE);
    }

    {
      boundaries_t<real_t> nonempty_box;
      for (auto d { 0u }; d < M::Dim; ++d) {
        if (d < box.size()) {
          nonempty_box.push_back({ box[d].first, box[d].second });
        } else {
          nonempty_box.push_back(Range::All);
        }
      }
      const auto result = injector.ComputeNumInject(params,
                                                    domain,
                                                    number_density,
                                                    nonempty_box);
      if (not std::get<0>(result)) {
        return;
      }
      const auto nparticles = std::get<1>(result);
      const auto xi_min     = std::get<2>(result);
      const auto xi_max     = std::get<3>(result);

      Kokkos::parallel_for(
        "InjectUniform",
        nparticles,
        kernel::UniformInjector_kernel<S, M, typename I::energy_dist_t>(
          injector.species.first,
          injector.species.second,
          domain.species[injector.species.first - 1],
          domain.species[injector.species.second - 1],
          domain.species[injector.species.first - 1].npart(),
          domain.species[injector.species.second - 1].npart(),
          domain.mesh.metric,
          xi_min,
          xi_max,
          injector.energy_dist,
          ONE / params.template get<real_t>("scales.V0"),
          domain.random_pool));
      domain.species[injector.species.first - 1].set_npart(
        domain.species[injector.species.first - 1].npart() + nparticles);
      domain.species[injector.species.second - 1].set_npart(
        domain.species[injector.species.second - 1].npart() + nparticles);
    }
  }

  namespace experimental {

    template <SimEngine::type S,
              class M,
              template <SimEngine::type, class> class ED1,
              template <SimEngine::type, class> class ED2>
    struct UniformInjector : BaseInjector<S, M> {
      using energy_dist_1_t = ED1<S, M>;
      using energy_dist_2_t = ED2<S, M>;
      static_assert(M::is_metric, "M must be a metric class");
      static_assert(energy_dist_1_t::is_energy_dist,
                    "ED1 must be an energy distribution class");
      static_assert(energy_dist_2_t::is_energy_dist,
                    "ED2 must be an energy distribution class");
      static constexpr bool      is_uniform_injector { true };
      static constexpr Dimension D { M::Dim };
      static constexpr Coord     C { M::CoordType };

      const energy_dist_1_t             energy_dist_1;
      const energy_dist_2_t             energy_dist_2;
      const std::pair<spidx_t, spidx_t> species;

      UniformInjector(const energy_dist_1_t&             energy_dist_1,
                      const energy_dist_2_t&             energy_dist_2,
                      const std::pair<spidx_t, spidx_t>& species)
        : energy_dist_1 { energy_dist_1 }
        , energy_dist_2 { energy_dist_2 }
        , species { species } {}

      ~UniformInjector() = default;
    };

    /**
     * @brief Injects uniform number density of particles everywhere in the domain
     * @param domain Domain object
     * @param injector Uniform injector object
     * @param number_density Total number density (in units of n0)
     * @param use_weights Use weights
     * @param box Region to inject the particles in global coords
     * @tparam S Simulation engine type
     * @tparam M Metric type
     * @tparam I Injector type
     */
    template <SimEngine::type S, class M, class I>
    inline void InjectUniform(const SimulationParams&     params,
                              Domain<S, M>&               domain,
                              const I&                    injector,
                              real_t                      number_density,
                              bool                        use_weights = false,
                              const boundaries_t<real_t>& box         = {}) {
      static_assert(M::is_metric, "M must be a metric class");
      static_assert(I::is_uniform_injector, "I must be a uniform injector class");
      raise::ErrorIf((M::CoordType != Coord::Cart) && (not use_weights),
                     "Weights must be used for non-Cartesian coordinates",
                     HERE);
      raise::ErrorIf((M::CoordType == Coord::Cart) && use_weights,
                     "Weights should not be used for Cartesian coordinates",
                     HERE);
      raise::ErrorIf(
        params.template get<bool>("particles.use_weights") != use_weights,
        "Weights must be enabled from the input file to use them in "
        "the injector",
        HERE);
      if (domain.species[injector.species.first - 1].charge() +
            domain.species[injector.species.second - 1].charge() !=
          0.0f) {
        raise::Warning("Total charge of the injected species is non-zero", HERE);
      }

      {
        boundaries_t<real_t> nonempty_box;
        for (auto d { 0u }; d < M::Dim; ++d) {
          if (d < box.size()) {
            nonempty_box.push_back({ box[d].first, box[d].second });
          } else {
            nonempty_box.push_back(Range::All);
          }
        }
        const auto result = injector.ComputeNumInject(params,
                                                      domain,
                                                      number_density,
                                                      nonempty_box);
        if (not std::get<0>(result)) {
          return;
        }
        const auto nparticles = std::get<1>(result);
        const auto xi_min     = std::get<2>(result);
        const auto xi_max     = std::get<3>(result);

        Kokkos::parallel_for(
          "InjectUniform",
          nparticles,
          kernel::experimental::
            UniformInjector_kernel<S, M, typename I::energy_dist_1_t, typename I::energy_dist_2_t>(
              injector.species.first,
              injector.species.second,
              domain.species[injector.species.first - 1],
              domain.species[injector.species.second - 1],
              domain.species[injector.species.first - 1].npart(),
              domain.species[injector.species.second - 1].npart(),
              domain.mesh.metric,
              xi_min,
              xi_max,
              injector.energy_dist_1,
              injector.energy_dist_2,
              ONE / params.template get<real_t>("scales.V0"),
              domain.random_pool));
        domain.species[injector.species.first - 1].set_npart(
          domain.species[injector.species.first - 1].npart() + nparticles);
        domain.species[injector.species.second - 1].set_npart(
          domain.species[injector.species.second - 1].npart() + nparticles);
      }
    }

  } // namespace experimental

  /**
   * @brief Injects particles from a globally-defined map
   * @note very inefficient, should only be used for debug purposes
   * @note (or when injecting very small # of particles)
   * @param global_domain Global metadomain object
   * @param local_domain Local domain object
   * @param spidx Species index
   * @param data Map containing all the coordinates/velocities of particles to inject
   * @param use_weights Boolean toggle to use weights or not
   */
  template <SimEngine::type S, class M>
  inline void InjectGlobally(const Metadomain<S, M>& global_domain,
                             Domain<S, M>&           local_domain,
                             spidx_t                 spidx,
                             const std::map<std::string, std::vector<real_t>>& data,
                             bool use_weights = false) {
    static_assert(M::is_metric, "M must be a metric class");
    const auto n_inject        = data.at("ux1").size();
    auto       injector_kernel = kernel::GlobalInjector_kernel<S, M>(
      local_domain.species[spidx - 1],
      global_domain.mesh().metric,
      local_domain,
      data,
      use_weights);
    Kokkos::parallel_for("InjectGlobally", n_inject, injector_kernel);
    const auto n_inj = injector_kernel.number_injected();
    local_domain.species[spidx - 1].set_npart(
      local_domain.species[spidx - 1].npart() + n_inj);
  }

  /**
   * @brief Injects particles based on spatial distribution function
   * @param params Simulation parameters
   * @param domain Local domain object
   * @param injector Non-uniform injector object
   * @param number_density Total number density (in units of n0)
   * @param use_weights Use weights
   * @param box Region to inject the particles in
   */
  template <SimEngine::type S, class M, class I>
  inline void InjectNonUniform(const SimulationParams&     params,
                               Domain<S, M>&               domain,
                               const I&                    injector,
                               real_t                      number_density,
                               bool                        use_weights = false,
                               const boundaries_t<real_t>& box         = {}) {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(I::is_nonuniform_injector,
                  "I must be a nonuniform injector class");
    raise::ErrorIf((M::CoordType != Coord::Cart) && (not use_weights),
                   "Weights must be used for non-Cartesian coordinates",
                   HERE);
    raise::ErrorIf((M::CoordType == Coord::Cart) && use_weights,
                   "Weights should not be used for Cartesian coordinates",
                   HERE);
    raise::ErrorIf(
      params.template get<bool>("particles.use_weights") and not use_weights,
      "Weights are enabled in the input but not enabled in the injector",
      HERE);
    raise::ErrorIf(
      not params.template get<bool>("particles.use_weights") and use_weights,
      "Weights are not enabled in the input but enabled in the injector",
      HERE);
    if (domain.species[injector.species.first - 1].charge() +
          domain.species[injector.species.second - 1].charge() !=
        0.0f) {
      raise::Warning("Total charge of the injected species is non-zero", HERE);
    }
    {
      range_t<M::Dim> cell_range;
      if (box.size() == 0) {
        cell_range = domain.mesh.rangeActiveCells();
      } else {
        raise::ErrorIf(box.size() != M::Dim,
                       "Box must have the same dimension as the mesh",
                       HERE);
        boundaries_t<bool> incl_ghosts;
        for (auto d = 0; d < M::Dim; ++d) {
          incl_ghosts.push_back({ false, false });
        }
        const auto extent = domain.mesh.ExtentToRange(box, incl_ghosts);
        tuple_t<ncells_t, M::Dim> x_min { 0 }, x_max { 0 };
        for (auto d = 0; d < M::Dim; ++d) {
          x_min[d] = extent[d].first;
          x_max[d] = extent[d].second;
        }
        cell_range = CreateRangePolicy<M::Dim>(x_min, x_max);
      }
      const auto ppc = number_density *
                       params.template get<real_t>("particles.ppc0") * HALF;
      auto injector_kernel =
        kernel::NonUniformInjector_kernel<S, M, typename I::energy_dist_t, typename I::spatial_dist_t>(
          ppc,
          injector.species.first,
          injector.species.second,
          domain.species[injector.species.first - 1],
          domain.species[injector.species.second - 1],
          domain.species[injector.species.first - 1].npart(),
          domain.species[injector.species.second - 1].npart(),
          domain.mesh.metric,
          injector.energy_dist,
          injector.spatial_dist,
          ONE / params.template get<real_t>("scales.V0"),
          domain.random_pool);
      Kokkos::parallel_for("InjectNonUniformNumberDensity",
                           cell_range,
                           injector_kernel);
      const auto n_inj = injector_kernel.number_injected();
      domain.species[injector.species.first - 1].set_npart(
        domain.species[injector.species.first - 1].npart() + n_inj);
      domain.species[injector.species.second - 1].set_npart(
        domain.species[injector.species.second - 1].npart() + n_inj);
    }
  }

} // namespace arch

#endif // ARCHETYPES_PARTICLE_INJECTOR_H
