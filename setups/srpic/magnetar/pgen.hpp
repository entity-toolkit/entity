#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar) : Bsurf { bsurf }, Rstar { rstar } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return Bsurf * HALF * math::sin(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
    }

  private:
    const real_t Bsurf, Rstar;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t bsurf, real_t rstar, real_t omega)
      : InitFields<D> { bsurf, rstar }
      , time { time }
      , Omega { omega } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      auto sigma = (x_Ph[1] - HALF * constant::PI) /
                   (static_cast<real_t>(0.2) * constant::PI);
      return Omega * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]) * sigma *
             math::exp((ONE - SQR(SQR(sigma))) * INV_4);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      auto sigma = (x_Ph[1] - 0.5 * constant::PI) /
                   (static_cast<real_t>(0.2) * constant::PI);
      return -Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]) * sigma *
             math::exp((ONE - SQR(SQR(sigma))) * INV_4);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const Metadomain<S, M>& global_domain;

    const real_t  Bsurf, Rstar, Omega;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , global_domain { m }
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Omega { p.template get<real_t>("setup.omega") }
      , init_flds { Bsurf, Rstar } {}

    inline PGen() {}

    // inline void InitPrtls(Domain<S, M>& local_domain) {
    //   Kokkos::deep_copy(local_domain.fields.bckp, ZERO);
    //   // @HACK
    //   const auto x_surf = 1.1f;
    //   const auto ds     = params.template get<real_t>(
    //     "grid.boundaries.atmosphere.ds");
    //   const auto temp = params.template get<real_t>(
    //     "grid.boundaries.atmosphere.temperature");
    //   const auto height = params.template get<real_t>(
    //     "grid.boundaries.atmosphere.height");
    //   const auto species = params.template get<std::pair<unsigned short, unsigned short>>(
    //     "grid.boundaries.atmosphere.species");
    //   const auto nmax = params.template get<real_t>(
    //     "grid.boundaries.atmosphere.density");
    //   const auto atm_injector =
    //     arch::AtmosphereInjector<SimEngine::SRPIC, M, false, in::x1> {
    //       local_domain.mesh.metric,
    //       local_domain.fields.bckp,
    //       nmax,
    //       height,
    //       x_surf,
    //       ds,
    //       temp,
    //       local_domain.random_pool,
    //       species
    //     };
    //   arch::InjectNonUniform<S, M, decltype(atm_injector)>(params,
    //                                                        local_domain,
    //                                                        atm_injector,
    //                                                        nmax,
    //                                                        true);
    // }

    auto FieldDriver(real_t time) const -> DriveFields<D> {
      const real_t omega_t =
        Omega *
        ((ONE - math::tanh((static_cast<real_t>(5.0) - time) * HALF)) *
         (ONE + (-ONE + math::tanh((static_cast<real_t>(45.0) - time) * HALF)) *
                  HALF)) *
        HALF;
      return DriveFields<D> { time, Bsurf, Rstar, omega_t };
    }

    //     void CustomPostStep(std::size_t time, long double, Domain<S, M>& domain) {

    //       const auto pp_thres    = 1*10.0;
    //       const auto gamma_pairs = 1*0.5 * 3.5;

    //     auto& species_e = domain.species[4];
    //     auto& species_p = domain.species[5];
    //     auto metric = domain.mesh.metric;

    //      for (std::size_t s { 0 }; s < 6; ++s) {
    //         if ((s == 1) || (s == 2) || (s == 3)) {
    //           continue;
    //         }

    //         array_t<std::size_t> elec_ind("elec_ind");
    //         array_t<std::size_t> pos_ind("pos_ind");
    //         auto offset_e = species_e.npart();
    //         auto offset_p = species_p.npart();

    //         auto ux1_e    = species_e.ux1;
    //         auto ux2_e    = species_e.ux2;
    //         auto ux3_e    = species_e.ux3;
    //         auto i1_e     = species_e.i1;
    //         auto i2_e     = species_e.i2;
    //         auto dx1_e    = species_e.dx1;
    //         auto dx2_e    = species_e.dx2;
    //         auto phi_e    = species_e.phi;
    //         auto weight_e = species_e.weight;
    //         auto tag_e    = species_e.tag;

    //         auto ux1_p    = species_p.ux1;
    //         auto ux2_p    = species_p.ux2;
    //         auto ux3_p    = species_p.ux3;
    //         auto i1_p     = species_p.i1;
    //         auto i2_p     = species_p.i2;
    //         auto dx1_p    = species_p.dx1;
    //         auto dx2_p    = species_p.dx2;
    //         auto phi_p    = species_p.phi;
    //         auto weight_p = species_p.weight;
    //         auto tag_p    = species_p.tag;

    //         auto& species = domain.species[s];
    //         auto ux1    = species.ux1;
    //         auto ux2    = species.ux2;
    //         auto ux3    = species.ux3;
    //         auto i1     = species.i1;
    //         auto i2     = species.i2;
    //         auto dx1    = species.dx1;
    //         auto dx2    = species.dx2;
    //         auto phi    = species.phi;
    //         auto weight = species.weight;
    //         auto tag    = species.tag;

    // Kokkos::parallel_for(
    //     "InjectPairs", species.rangeActiveParticles(), Lambda(index_t p) {
    //       if (tag(p) == ParticleTag::dead) {
    //         return;
    //       }

    //             auto px      = ux1(p);
    //             auto py      = ux2(p);
    //             auto pz      = ux3(p);
    //             auto gamma   = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));

    //           const coord_t<D> xCd{
    //               static_cast<real_t>(i1(p)) + dx1(p),
    //               static_cast<real_t>(i2(p)) + dx2(p)};

    //         coord_t<D> xPh { ZERO };
    //         metric.template convert<Crd::Cd, Crd::Ph>(xCd, xPh);

    //       if ((gamma > pp_thres) && (math::sin(xPh[1]) > 0.1)) {

    //         auto new_gamma = gamma - 2.0 * gamma_pairs;
    //         auto new_fac = math::sqrt(SQR(new_gamma) - 1.0) / math::sqrt(SQR(gamma) - 1.0);
    //         auto pair_fac = math::sqrt(SQR(gamma_pairs) - 1.0) / math::sqrt(SQR(gamma) - 1.0);

    //           auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
    //           auto pos_p  = Kokkos::atomic_fetch_add(&pos_ind(), 1);

    //           i1_e(elec_p + offset_e) = i1(p);
    //           dx1_e(elec_p + offset_e) = dx1(p);
    //           i2_e(elec_p + offset_e) = i2(p);
    //           dx2_e(elec_p + offset_e) = dx2(p);
    //           phi_e(elec_p + offset_e) = phi(p);
    //           ux1_e(elec_p + offset_e) = px * pair_fac;
    //           ux2_e(elec_p + offset_e) = py * pair_fac;
    //           ux3_e(elec_p + offset_e) = pz * pair_fac;
    //           weight_e(elec_p + offset_e) = weight(p);
    //           tag_e(elec_p + offset_e) = ParticleTag::alive;

    //           i1_p(pos_p + offset_p) = i1(p);
    //           dx1_p(pos_p + offset_p) = dx1(p);
    //           i2_p(pos_p + offset_p) = i2(p);
    //           dx2_p(pos_p + offset_p) = dx2(p);
    //           phi_p(pos_p + offset_p) = phi(p);
    //           ux1_p(pos_p + offset_p) = px * pair_fac;
    //           ux2_p(pos_p + offset_p) = py * pair_fac;
    //           ux3_p(pos_p + offset_p) = pz * pair_fac;
    //           weight_p(pos_p + offset_p) = weight(p);
    //           tag_p(pos_p + offset_p) = ParticleTag::alive;

    //           ux1(p) *= new_fac;
    //           ux2(p) *= new_fac;
    //           ux3(p) *= new_fac;

    //       }

    //     });

    //         auto elec_ind_h = Kokkos::create_mirror(elec_ind);
    //         Kokkos::deep_copy(elec_ind_h, elec_ind);
    //         species_e.set_npart(offset_e + elec_ind_h());

    //         auto pos_ind_h = Kokkos::create_mirror(pos_ind);
    //         Kokkos::deep_copy(pos_ind_h, pos_ind);
    //         species_p.set_npart(offset_p + pos_ind_h());

    //       }

    //     }
  };

} // namespace user

#endif
