#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <vector>

#define from_Xi_to_i(XI, I)                                                    \
  {                                                                            \
    (I) = static_cast<int>(((XI) + 1)) - 1;                                    \
  }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    (DI) = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);             \
  }

#define i_di_to_Xi(I, DI) (static_cast<real_t>((I)) + static_cast<real_t>((DI)))

namespace user {
  using namespace ntt;

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
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams& params;

    const Metadomain<S, M>& metadomain;

    const real_t temperature, temperature_gradient;

    PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , temperature { params.template get<real_t>("setup.temperature", 0.0) }
      , temperature_gradient {
        params.template get<real_t>("setup.temperature_gradient", 0.0)
      } {}

    void InitPrtls(Domain<S, M>& local_domain) {
      const auto empty = std::vector<real_t> {};
      const auto x1_e  = params.template get<std::vector<real_t>>("setup.x1_e",
                                                                 empty);
      const auto x2_e  = params.template get<std::vector<real_t>>("setup.x2_e",
                                                                 empty);
      const auto x3_e  = params.template get<std::vector<real_t>>("setup.x3_e",
                                                                 empty);
      const auto phi_e = params.template get<std::vector<real_t>>("setup.phi_e",
                                                                  empty);
      const auto ux1_e = params.template get<std::vector<real_t>>("setup.ux1_e",
                                                                  empty);
      const auto ux2_e = params.template get<std::vector<real_t>>("setup.ux2_e",
                                                                  empty);
      const auto ux3_e = params.template get<std::vector<real_t>>("setup.ux3_e",
                                                                  empty);

      const auto x1_i  = params.template get<std::vector<real_t>>("setup.x1_i",
                                                                 empty);
      const auto x2_i  = params.template get<std::vector<real_t>>("setup.x2_i",
                                                                 empty);
      const auto x3_i  = params.template get<std::vector<real_t>>("setup.x3_i",
                                                                 empty);
      const auto phi_i = params.template get<std::vector<real_t>>("setup.phi_i",
                                                                  empty);
      const auto ux1_i = params.template get<std::vector<real_t>>("setup.ux1_i",
                                                                  empty);
      const auto ux2_i = params.template get<std::vector<real_t>>("setup.ux2_i",
                                                                  empty);
      const auto ux3_i = params.template get<std::vector<real_t>>("setup.ux3_i",
                                                                  empty);
      std::map<std::string, std::vector<real_t>> data_e {
        {  "x1",  x1_e },
        {  "x2",  x2_e },
        { "ux1", ux1_e },
        { "ux2", ux2_e },
        { "ux3", ux3_e }
      };
      std::map<std::string, std::vector<real_t>> data_i {
        {  "x1",  x1_i },
        {  "x2",  x2_i },
        { "ux1", ux1_i },
        { "ux2", ux2_i },
        { "ux3", ux3_i }
      };
      if constexpr (M::CoordType == Coord::Cartesian or D == Dim::_3D) {
        data_e["x3"] = x3_e;
        data_i["x3"] = x3_i;
      } else if constexpr (D == Dim::_2D) {
        data_e["phi"] = phi_e;
        data_i["phi"] = phi_i;
      }

      arch::InjectGlobally<S, M>(metadomain, local_domain, (spidx_t)1, data_e);
      arch::InjectGlobally<S, M>(metadomain, local_domain, (spidx_t)2, data_i);
    }

    auto FixFieldsConst(const bc_in&, const em&) const -> std::pair<real_t, bool> {
      return { ZERO, false };
    }

    struct CustomPrtlUpdate {
      random_number_pool_t pool;
      real_t               temp_cold, temp_hot;
      real_t               xmin, xmax;

      CustomPrtlUpdate(random_number_pool_t& pool,
                       real_t                temp_cold,
                       real_t                temp_hot,
                       real_t                xmin,
                       real_t                xmax)
        : pool { pool }
        , temp_cold { temp_cold }
        , temp_hot { temp_hot }
        , xmin { xmin }
        , xmax { xmax } {}

      Inline void operator()(index_t                          p,
                             const kernel::sr::PusherContext& ctx,
                             const kernel::sr::PusherBoundaries<M::Dim>&,
                             const kernel::PusherArrays& particles,
                             const M&                    metric) const {

        const auto x_Cd = static_cast<real_t>(particles.i1(p)) +
                          static_cast<real_t>(particles.dx1(p));
        const auto x_Ph = metric.template convert<1, Crd::Cd, Crd::XYZ>(x_Cd);
        vec_t<Dim::_3D>       v { ZERO };
        const coord_t<M::Dim> x_dummy { ZERO };

        // step 1: calculate the particle 3 velocity
        const real_t gamma_p = math::sqrt(ONE + SQR(particles.ux1(p)) +
                                          SQR(particles.ux2(p)) +
                                          SQR(particles.ux3(p)));

        const real_t beta_x_p = math::abs(particles.ux1(p)) / gamma_p;
        const real_t xp_prev  = i_di_to_Xi(particles.i1_prev(p),
                                          particles.dx1_prev(p));

        // Reflecting boundary that resamples velocity
        if (x_Ph < xmin) {
          arch::energy_dist::JuttnerSinge(v, temp_cold, pool);

          // calculate the time for the particle to reach the wall
          const int      delta_i1_to_wall  = particles.i1_prev(p);
          const prtldx_t delta_dx1_to_wall = particles.dx1_prev(p);
          const real_t dx_to_wall = i_di_to_Xi(delta_i1_to_wall, delta_dx1_to_wall);
          const real_t dt_to_wall = dx_to_wall /
                                    metric.template transform<1, Idx::XYZ, Idx::U>(
                                      x_dummy,
                                      beta_x_p);

          // update the velocity to the post-collision value (reflection with new speed sampled from Juttner distribution)
          particles.ux1(p) = math::abs(v[0]);
          particles.ux2(p) = v[1];
          particles.ux3(p) = v[2];

          // calculate remaining time after the collision
          const real_t remaining_dt            = ctx.dt - dt_to_wall;
          const real_t remaining_dt_inv_energy = remaining_dt /
                                                 math::sqrt(
                                                   ONE + SQR(particles.ux1(p)) +
                                                   SQR(particles.ux2(p)) +
                                                   SQR(particles.ux3(p)));

          // update the position to the post-collision value (reflection with new speed sampled from Juttner distribution)
          particles.i1(p)  = 0;
          particles.dx1(p) = metric.template transform<1, Idx::XYZ, Idx::U>(
                               x_dummy,
                               particles.ux1(p)) *
                             remaining_dt_inv_energy;

        } else if (x_Ph > xmax) {
          arch::energy_dist::JuttnerSinge(v, temp_hot, pool);

          // step 2: calculate the time for the particle to reach the piston
          const int      delta_i1_to_wall  = ctx.ni1 - 1 - particles.i1_prev(p);
          const prtldx_t delta_dx1_to_wall = ONE - particles.dx1_prev(p);
          const real_t dx_to_wall = i_di_to_Xi(delta_i1_to_wall, delta_dx1_to_wall);
          const real_t dt_to_wall = dx_to_wall /
                                    metric.template transform<1, Idx::XYZ, Idx::U>(
                                      x_dummy,
                                      beta_x_p);

          // update the velocity to the post-collision value (reflection with new speed sampled from Juttner distribution)
          particles.ux1(p) = -math::abs(v[0]);
          particles.ux2(p) = v[1];
          particles.ux3(p) = v[2];

          // step 3: calculate remaining time after the collision
          const real_t remaining_dt            = ctx.dt - dt_to_wall;
          const real_t remaining_dt_inv_energy = remaining_dt /
                                                 math::sqrt(
                                                   ONE + SQR(particles.ux1(p)) +
                                                   SQR(particles.ux2(p)) +
                                                   SQR(particles.ux3(p)));

          // update the position to the post-collision value (reflection with new speed sampled from Juttner distribution)
          particles.i1(p) = ctx.ni1 - 2;
          particles.dx1(p) = ONE - metric.template transform<1, Idx::XYZ, Idx::U>(
                                     x_dummy,
                                     math::abs(particles.ux1(p))) *
                                     remaining_dt_inv_energy;
        }
      }
    };

    template <class D>
    auto CustomParticleUpdate(simtime_t /*time*/, spidx_t sp, D& domain) const
      -> CustomPrtlUpdate {
      return CustomPrtlUpdate {
        domain.random_pool(),
        temperature / domain.species[sp - 1].mass(), // sp is 1-indexed
        temperature_gradient * temperature / domain.species[sp - 1].mass(),
        metadomain.mesh().extent(in::x1).first, // xmin
        metadomain.mesh().extent(in::x1).second // xmax
      };
    }
  };

} // namespace user

#undef from_Xi_to_i_di
#undef from_Xi_to_i
#undef i_di_to_Xi

#endif // PROBLEM_GENERATOR_H