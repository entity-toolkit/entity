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
  using spidx_t = unsigned short;

  template <SimEngine::type S, class M, template <SimEngine::type, class> class ED>
  struct UniformInjector {
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

    auto DeduceInjectRegion(const Domain<S, M>&         domain,
                            const boundaries_t<real_t>& box) const
      -> std::tuple<bool, ncells_t, array_t<real_t*>, array_t<real_t*>> {
      auto i_min = array_t<real_t*> { "i_min", M::Dim };
      auto i_max = array_t<real_t*> { "i_max", M::Dim };

      if (not domain.mesh.Intersects(box)) {
        return { false, (ncells_t)0, i_min, i_max };
      }

      tuple_t<ncells_t, M::Dim> range_min { 0 };
      tuple_t<ncells_t, M::Dim> range_max { 0 };

      boundaries_t<bool> incl_ghosts;
      for (auto d { 0u }; d < M::Dim; ++d) {
        incl_ghosts.push_back({ false, false });
      }
      const auto intersect_range = domain.mesh.ExtentToRange(box, incl_ghosts);

      for (auto d { 0u }; d < M::Dim; ++d) {
        range_min[d] = intersect_range[d].first;
        range_max[d] = intersect_range[d].second;
      }

      ncells_t ncells = 1;
      for (auto d = 0u; d < M::Dim; ++d) {
        ncells *= (range_max[d] - range_min[d]);
      }

      auto i_min_h = Kokkos::create_mirror_view(i_min);
      auto i_max_h = Kokkos::create_mirror_view(i_max);
      for (auto d = 0u; d < M::Dim; ++d) {
        i_min_h(d) = (real_t)(range_min[d]);
        i_max_h(d) = (real_t)(range_max[d]);
      }

      Kokkos::deep_copy(i_min, i_min_h);
      Kokkos::deep_copy(i_max, i_max_h);
      return { true, ncells, i_min, i_max };
    }

    auto ComputeNumInject(const SimulationParams&     params,
                          const Domain<S, M>&         domain,
                          real_t                      number_density,
                          const boundaries_t<real_t>& box) const
      -> std::tuple<bool, npart_t, array_t<real_t*>, array_t<real_t*>> {
      const auto result = DeduceInjectRegion(domain, box);
      const auto i_min  = std::get<2>(result);
      const auto i_max  = std::get<3>(result);

      if (not std::get<0>(result)) {
        return { false, (npart_t)0, i_min, i_max };
      }
      const auto ncells = std::get<1>(result);

      const auto ppc0       = params.template get<real_t>("particles.ppc0");
      const auto nparticles = static_cast<npart_t>(
        (long double)(ppc0 * number_density * 0.5) * (long double)(ncells));

      return { true, nparticles, i_min, i_max };
    }
  };

  template <SimEngine::type S, class M, template <SimEngine::type, class> class ED>
  struct KeepConstantInjector : UniformInjector<S, M, ED> {
    using energy_dist_t = ED<S, M>;
    using UniformInjector<S, M, ED>::D;
    using UniformInjector<S, M, ED>::C;

    boundaries_t<real_t> probe_box;

    KeepConstantInjector(const energy_dist_t&               energy_dist,
                         const std::pair<spidx_t, spidx_t>& species,
                         boundaries_t<real_t>               box = {})
      : UniformInjector<S, M, ED> { energy_dist, species } {
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
                           const Domain<S, M>&     domain,
                           boundaries_t<real_t>    box) const -> real_t {
      const auto use_weights = params.template get<bool>(
        "particles.use_weights");
      const auto ni2    = domain.mesh.n_active(in::x2);
      const auto inv_n0 = ONE / params.template get<real_t>("scales.n0");

      auto scatter_buff = Kokkos::Experimental::create_scatter_view(
        domain.fields.buff);

      for (const auto& sp : { this->species.first, this->species.second }) {
        raise::ErrorIf(sp >= domain.species.size(),
                       "Species index out of bounds",
                       HERE);
        const auto& prtl_spec = domain.species[sp - 1];
        Kokkos::deep_copy(domain.fields.buff, ZERO);
        // clang-format off
        Kokkos::parallel_for(
          "ComputeMoments",
          prtl_spec.rangeActiveParticles(),
          kernel::ParticleMoments_kernel<S, M, FldsID::N, 3>({}, scatter_buff, 0,
                                                             prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
                                                             prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
                                                             prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
                                                             prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
                                                             prtl_spec.mass(), prtl_spec.charge(),
                                                             use_weights,
                                                             domain.mesh.metric, domain.mesh.flds_bc(),
                                                             ni2, inv_n0, 0));
        // clang-format on
      }
      Kokkos::Experimental::contribute(domain.fields.buff, scatter_buff);

      real_t dens { ZERO };

      const auto result       = DeduceInjectRegion(domain, box);
      const auto should_probe = std::get<0>(result);
      const auto ncells       = std::get<1>(result);
      const auto i_min        = std::get<2>(result);
      const auto i_max        = std::get<3>(result);

      if (should_probe) {
        range_t<M::Dim> probe_range = CreateRangePolicy<M::Dim>(i_min, i_max);
        Kokkos::parallel_reduce(
          "AvgDensity",
          probe_range,
          kernel::ComputeSum_kernel<M::Dim, 3>(domain.fields.buff, 0),
          dens);
      }
#if defined(MPI_ENABLED)
      real_t   tot_dens { ZERO };
      ncells_t tot_ncells { 0 };
      MPI_Allreduce(dens, &tot_dens, 1, mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(ncells,
                    &tot_ncells,
                    1,
                    mpi::get_type<ncells_t>(),
                    MPI_SUM,
                    MPI_COMM_WORLD);
      dens   = tot_dens;
      ncells = tot_ncells;
#endif
      if (ncells > 0) {
        return dens / (real_t)(ncells);
      } else {
        return ZERO;
      }
    }

    auto ComputeNumInject(const SimulationParams&     params,
                          const Domain<S, M>&         domain,
                          real_t                      number_density,
                          const boundaries_t<real_t>& box) const
      -> std::tuple<bool, npart_t, array_t<real_t*>, array_t<real_t*>> {
      const auto computed_avg_density = ComputeAvgDensity(params, domain);

      const auto result = DeduceInjectRegion(domain, box);
      const auto i_min  = std::get<2>(result);
      const auto i_max  = std::get<3>(result);

      if (not std::get<0>(result)) {
        return { false, (npart_t)0, i_min, i_max };
      }
      const auto ncells = std::get<1>(result);

      const auto ppc0       = params.template get<real_t>("particles.ppc0");
      const auto nparticles = static_cast<npart_t>(
        (long double)(ppc0 * (number_density - computed_avg_density) * 0.5) *
        (long double)(ncells));

      return { true, nparticles, i_min, i_max };
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
          const auto xi = x_Ph[static_cast<unsigned short>(O)];
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
          const auto xi = x_Ph[static_cast<unsigned short>(O)];
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
      const auto i_min      = std::get<2>(result);
      const auto i_max      = std::get<3>(result);

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
          i_min,
          i_max,
          injector.energy_dist,
          ONE / params.template get<real_t>("scales.V0"),
          domain.random_pool));
      domain.species[injector.species.first - 1].set_npart(
        domain.species[injector.species.first - 1].npart() + nparticles);
      domain.species[injector.species.second - 1].set_npart(
        domain.species[injector.species.second - 1].npart() + nparticles);
    }
  }

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
