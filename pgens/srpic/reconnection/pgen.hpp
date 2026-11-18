#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct CurrentLayer {
    CurrentLayer(real_t cs_width, real_t center_x, real_t cs_y)
      : cs_width { cs_width }
      , center_x { center_x }
      , cs_y { cs_y } {}

    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      return ONE / SQR(math::cosh((x_Ph[1] - cs_y) / cs_width)) *
             (ONE - math::exp(-SQR((x_Ph[0] - center_x) / cs_width)));
    }

  private:
    const real_t cs_width, center_x, cs_y;
  };

  // field initializer
  template <Dimension D>
  struct InitFields {
    InitFields(real_t bg_B, real_t bg_Bguide, real_t cs_width, real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , cs_width { cs_width }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

  private:
    const real_t bg_B, bg_Bguide, cs_width, cs_y;
  };

  template <Dimension D>
  struct BoundaryFieldsInX1 {
    BoundaryFieldsInX1(real_t bg_B,
                       real_t bg_Bguide,
                       real_t beta_rec,
                       real_t cs_width,
                       real_t cs_x,
                       real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , beta_rec { beta_rec }
      , cs_width { cs_width }
      , cs_x { cs_x }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return beta_rec * bg_B * (math::tanh((x_Ph[0] - cs_x) / cs_width));
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return beta_rec * bg_Bguide * math::tanh((x_Ph[1] - cs_y) / cs_width);
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return -beta_rec * bg_B;
    }

  private:
    const real_t bg_B, bg_Bguide, beta_rec, cs_width, cs_x, cs_y;
  };

  template <Dimension D>
  struct BoundaryFieldsInX2 {
    BoundaryFieldsInX2(real_t bg_B, real_t bg_Bguide, real_t cs_width, real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , cs_width { cs_width }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t bg_B, bg_Bguide, cs_width, cs_y;
  };

  // constant particle density for particle boundaries
  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams& params;
    Metadomain<S, M>&       metadomain;

    const real_t    bg_B, bg_Bguide, bg_temperature, inj_ypad;
    const real_t    cs_width, cs_overdensity, cs_x, cs_y;
    const real_t    ymin, ymax;
    const simtime_t t_open;
    bool            bc_opened { false };

    InitFields<D> init_flds;

    PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , bg_B { params.template get<real_t>("setup.bg_B", 1.0) }
      , bg_Bguide { params.template get<real_t>("setup.bg_Bguide", 0.0) }
      , bg_temperature { params.template get<real_t>("setup.bg_temperature", 0.001) }
      , inj_ypad { params.template get<real_t>("setup.inj_ypad", (real_t)0.05) }
      , cs_width { params.template get<real_t>("setup.cs_width") }
      , cs_overdensity { params.template get<real_t>("setup.cs_overdensity") }
      , cs_x { INV_2 *
               (m.mesh().extent(in::x1).second + m.mesh().extent(in::x1).first) }
      , cs_y { INV_2 *
               (m.mesh().extent(in::x2).second + m.mesh().extent(in::x2).first) }
      , ymin { m.mesh().extent(in::x2).first }
      , ymax { m.mesh().extent(in::x2).second }
      , t_open { params.template get<simtime_t>(
          "setup.t_open",
          1.5 * HALF *
            (m.mesh().extent(in::x1).second - m.mesh().extent(in::x1).first)) }
      , init_flds { bg_B, bg_Bguide, cs_width, cs_y } {}

    auto MatchFieldsInX1(simtime_t) const -> BoundaryFieldsInX1<D> {
      return BoundaryFieldsInX1<D> { bg_B,     bg_Bguide, (real_t)0.1,
                                     cs_width, cs_x,      cs_y };
    }

    auto MatchFieldsInX2(simtime_t) const -> BoundaryFieldsInX2<D> {
      return BoundaryFieldsInX2<D> { bg_B, bg_Bguide, cs_width, cs_y };
    }

    void InitPrtls(Domain<S, M>& local_domain) {
      // background
      arch::InjectUniformMaxwellian<S, M>(params,
                                          local_domain,
                                          ONE,
                                          bg_temperature,
                                          { 1, 2 });

      const auto sigma = params.template get<real_t>("scales.sigma0");
      const auto c_omp = params.template get<real_t>("scales.skindepth0");
      const auto cs_drift_beta = math::sqrt(sigma) * c_omp /
                                 (cs_width * cs_overdensity);
      const auto cs_drift_gamma = ONE / math::sqrt(ONE - SQR(cs_drift_beta));
      const auto cs_drift_u     = cs_drift_beta * cs_drift_gamma;
      const auto cs_temperature = HALF * sigma / cs_overdensity;

      // current layer
      auto edist_cs = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        local_domain.random_pool(),
        cs_temperature);
      const auto sdist_cs = CurrentLayer<M::Dim>(cs_width, cs_x, cs_y);
      arch::InjectNonUniform<S, M, decltype(edist_cs), decltype(edist_cs), decltype(sdist_cs)>(
        params,
        local_domain,
        { 1, 2 },
        { edist_cs, edist_cs },
        sdist_cs,
        cs_overdensity);
    }

    void CustomPostStep(timestep_t, simtime_t time, Domain<S, M>& domain) {
      // open boundaries if not yet opened at time = t_open
      if ((t_open > 0.0) and (not bc_opened) and (time > t_open)) {
        bc_opened = true;
        metadomain.setFldsBC(bc_in::Mx1, FldsBC::MATCH);
        metadomain.setPrtlBC(bc_in::Mx1, PrtlBC::ABSORB);
        metadomain.setFldsBC(bc_in::Px1, FldsBC::MATCH);
        metadomain.setPrtlBC(bc_in::Px1, PrtlBC::ABSORB);
      }

      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        bg_temperature);

      const auto dx = domain.mesh.metric.template sqrt_h_<1, 1>({});

      boundaries_t<real_t> inj_box_up, inj_box_down;
      inj_box_up.push_back(Range::All);
      inj_box_down.push_back(Range::All);
      inj_box_up.push_back({ ymax - inj_ypad - 10 * dx, ymax - inj_ypad });
      inj_box_down.push_back({ ymin + inj_ypad, ymin + inj_ypad + 10 * dx });

      if constexpr (M::Dim == Dim::_3D) {
        inj_box_up.push_back(Range::All);
        inj_box_down.push_back(Range::All);
      }

      // compute density of species #1 and #2
      arch::ComputeMomentWithSpecies<S, M, FldsID::Rho, 3>(params,
                                                           domain,
                                                           { 1, 2 },
                                                           domain.fields.buff);

      const auto replenish_sdist = arch::spatial_dist::ReplenishUniform<M, 3>(
        domain.mesh.metric,
        domain.fields.buff,
        0u,
        ONE);
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
        params,
        domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        replenish_sdist,
        ONE,
        params.template get<bool>("particles.use_weights"),
        inj_box_up);
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
        params,
        domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        replenish_sdist,
        ONE,
        params.template get<bool>("particles.use_weights"),
        inj_box_down);
    }
  };

} // namespace user

#endif
