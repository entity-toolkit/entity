#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct DipoleField {
    Inline auto radius(const coord_t<D>&) const -> real_t;

    Inline auto bx1(const coord_t<D>& x) const -> real_t {
      return THREE * x[0] * x[1] / math::pow(radius(x), 5);
    }

    Inline auto bx2(const coord_t<D>& x) const -> real_t {
      const auto r = radius(x);
      return (THREE * x[1] * x[1] - SQR(r)) / math::pow(r, 5);
    }

    Inline auto bx3(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_3D) {
        return THREE * x[2] * x[1] / math::pow(radius(x), 5);
      } else {
        return ZERO;
      }
    }
  };

  template <Dimension D>
  struct ZeroFields {
    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }
  };

  template <class M>
  Inline auto GetParticlePosition(const M&                  metric,
                                  index_t                   p,
                                  const array_t<int*>&      i1,
                                  const array_t<prtldx_t*>& dx1,
                                  const array_t<int*>&      i2,
                                  const array_t<prtldx_t*>& dx2,
                                  const array_t<int*>&      i3,
                                  const array_t<prtldx_t*>& dx3,
                                  coord_t<M::Dim>&          x) -> real_t {
    if constexpr (M::Dim == Dim::_2D) {
      const auto i1_ = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
      const auto i2_ = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
      metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x);
      return math::sqrt(SQR(x[0]) + SQR(x[1]));
    } else if constexpr (M::Dim == Dim::_3D) {
      const auto i1_ = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
      const auto i2_ = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
      const auto i3_ = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
      metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ }, x);
      return math::sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2]));
    }
  }

  template <Dimension D>
  Inline void SetParticleSpeed(index_t                 p,
                               const array_t<real_t*>& ux1,
                               const array_t<real_t*>& ux2,
                               const array_t<real_t*>& ux3,
                               const coord_t<D>&       x,
                               real_t                  dist,
                               real_t                  velocity) {
    ux1(p) = velocity * x[0] / dist;
    ux2(p) = velocity * x[1] / dist;
    if constexpr (D == Dim::_3D) {
      ux3(p) = velocity * x[2] / dist;
    } else {
      ux3(p) = ZERO;
    }
  }

  template <>
  Inline auto DipoleField<Dim::_2D>::radius(const coord_t<Dim::_2D>& x) const
    -> real_t {
    return math::sqrt(SQR(x[0]) + SQR(x[1]));
  }

  template <>
  Inline auto DipoleField<Dim::_3D>::radius(const coord_t<Dim::_3D>& x) const
    -> real_t {
    return math::sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2]));
  }

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {}
    };

    // DipoleField<M::Dim> init_flds;
    auto ExternalFields(simtime_t, spidx_t, const Domain<S, M>&) const
      -> std::pair<bool, DipoleField<M::Dim>> {
      return { true, DipoleField<M::Dim> {} };
    }

    const Metadomain<S, M>& metadomain;

    PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : arch::ProblemGenerator<S, M> { p }
      , metadomain { metadomain } {}

    void CustomPostStep(timestep_t /*step*/, simtime_t /*time*/, Domain<S, M>& domain) {
      const auto temperature = this->params.template get<real_t>(
        "setup.temperature");
      const auto drift_vel = this->params.template get<real_t>(
        "setup.drift_vel");
      const auto inject_xrange = this->params.template get<std::vector<real_t>>(
        "setup.inject_xrange");
      // compute the density of species #1 and #2
      // and save in the field buffer (index 0)
      arch::ComputeMomentWithSpecies<S, M, FldsID::N, 3>(this->params,
                                                         domain,
                                                         { 1u, 2u },
                                                         domain.fields.buff);
      const auto energy_dist = arch::Maxwellian<S, M>(
        domain.mesh.metric,
        domain.random_pool(),
        temperature,                // <-- target temperature for injection
        { drift_vel, ZERO, ZERO }); // <-- drift 4-velocity

      // pass the computed density to the replenisher
      const auto replenish_sdist = arch::ReplenishUniform<S, M, 3>(
        domain.mesh.metric,
        domain.fields.buff,
        0u,   // <-- index in buff where the density is stored
        ONE); // <-- target density for replenishment
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
        this->params,
        domain,
        {
          1u,
          2u
      },                       // <-- species to inject
        { energy_dist, energy_dist }, // <-- energy distributions for both species
        replenish_sdist,
        ONE, // <-- target max number density
        false,
        { { inject_xrange[0], inject_xrange[1] }, Range::All, Range::All }); // <-- injection region

      const auto r_purge   = this->params.template get<real_t>("setup.r_purge");
      const auto r_plummet = this->params.template get<real_t>(
        "setup.r_plummet");
      const auto plummet_speed = this->params.template get<real_t>(
        "setup.plummet_speed");
      // particles below r_plummet, gain a constant speed towards the origin,
      // and are removed when they reach r_purge
      for (auto& species : domain.species) {
        const auto i1  = species.i1;
        const auto i2  = species.i2;
        const auto i3  = species.i3;
        const auto dx1 = species.dx1;
        const auto dx2 = species.dx2;
        const auto dx3 = species.dx3;

        auto ux1 = species.ux1;
        auto ux2 = species.ux2;
        auto ux3 = species.ux3;
        auto tag = species.tag;

        const auto& mesh = domain.mesh;
        Kokkos::parallel_for(
          "PurgeParticles",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (tag(p) != ParticleTag::alive) {
              return;
            }
            coord_t<M::Dim> x { ZERO };
            const auto      distance =
              GetParticlePosition<M>(mesh.metric, p, i1, dx1, i2, dx2, i3, dx3, x);
            if (distance < r_purge) {
              tag(p) = ParticleTag::dead;
            } else if (distance < r_plummet) {
              SetParticleSpeed<M::Dim>(p, ux1, ux2, ux3, x, distance, plummet_speed);
            }
          });
      }
    }

    auto MatchFields(simtime_t) const -> ZeroFields<M::Dim> {
      return ZeroFields<M::Dim> {};
    }
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H
