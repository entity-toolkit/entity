#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

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
      auto pival        = 3.141592653589793;
      auto sigma  = (x_Ph[1] - 0.5 * pival) / (0.2 * pival);
      return Omega * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]) * sigma * math::exp((1.0 - SQR(SQR(sigma))) / 4.0);
      // return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      auto pival        = 3.141592653589793;
      auto sigma  = (x_Ph[1] - 0.5 * pival) / (0.2 * pival);
      return - Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]) * sigma * math::exp((1.0 - SQR(SQR(sigma))) / 4.0);
      // return ZERO;
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
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , init_flds { Bsurf, Rstar } {}

    inline PGen() {}

    auto FieldDriver(real_t time) const -> DriveFields<D> {
      return DriveFields<D> {
        time,
        Bsurf,
        Rstar,
        Omega * ((1 - math::tanh((5.0 - time) / 2.0)) *
                          (1 + (-1 + math::tanh((45.0 - time) / 2.0)) / 2.)) /
                         2.
      };
    }

    void CustomPostStep(std::size_t time, long double, Domain<S, M>& domain) {
      const auto pp_thres    = 4*10.0;
      const auto gamma_pairs = 4*0.5 * 3.5;

    // initialize the injector index
    auto idx = Kokkos::View<std::size_t>{"idx"};

    auto& species_e = domain.species[4];
    auto& species_p = domain.species[5];
    auto metric = domain.mesh.metric;

     for (std::size_t s { 0 }; s < 6; ++s) {
        if ((s == 1) || (s == 2) || (s == 3)) {
          continue;
        }

        array_t<std::size_t> elec_ind("elec_ind");
        array_t<std::size_t> pos_ind("pos_ind");
        auto offset_e = species_e.npart();
        auto offset_p = species_p.npart();

        auto ux1_e    = species_e.ux1;
        auto ux2_e    = species_e.ux2;
        auto ux3_e    = species_e.ux3;
        auto i1_e     = species_e.i1;
        auto i2_e     = species_e.i2;
        auto dx1_e    = species_e.dx1;
        auto dx2_e    = species_e.dx2;
        auto phi_e     = species_e.phi;
        auto weight_e = species_e.weight;        
        auto tag_e    = species_e.tag;

        auto ux1_p    = species_p.ux1;
        auto ux2_p    = species_p.ux2;
        auto ux3_p    = species_p.ux3;
        auto i1_p     = species_p.i1;
        auto i2_p     = species_p.i2;
        auto dx1_p    = species_p.dx1;
        auto dx2_p    = species_p.dx2;
        auto phi_p     = species_p.phi;
        auto weight_p = species_p.weight;
        auto tag_p    = species_p.tag;

        auto& species = domain.species[s];
        auto ux1    = species.ux1;
        auto ux2    = species.ux2;
        auto ux3    = species.ux3;
        auto i1     = species.i1;
        auto i2     = species.i2;
        auto dx1    = species.dx1;
        auto dx2    = species.dx2;
        auto phi     = species.phi;
        auto weight = species.weight;        
        auto tag    = species.tag;

Kokkos::parallel_for(
    "InjectPairs", species.rangeActiveParticles(), Lambda(index_t p) {
      if (tag(p) == ParticleTag::dead) {
        return;
      }

            auto px      = ux1(p);
            auto py      = ux2(p);
            auto pz      = ux3(p);
            auto gamma   = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));

        // const coord_t<M::PrtlDim> xCd{
        //     static_cast<real_t>(i1(p)) + dx1(p),
        //     static_cast<real_t>(i2(p)) + dx2(p),
        //     phi(p)};

      if (gamma > pp_thres) {

        auto new_gamma = gamma - 2.0 * gamma_pairs;
        auto new_fac = math::sqrt(SQR(new_gamma) - 1.0) / math::sqrt(SQR(gamma) - 1.0);
        auto pair_fac = math::sqrt(SQR(gamma_pairs) - 1.0) / math::sqrt(SQR(gamma) - 1.0);

              auto elec_p = Kokkos::atomic_fetch_add(&elec_ind(), 1);
              auto pos_p  = Kokkos::atomic_fetch_add(&pos_ind(), 1);

          i1_e(elec_p + offset_e) = i1(p);
          dx1_e(elec_p + offset_e) = dx1(p);
          i2_e(elec_p + offset_e) = i2(p);
          dx2_e(elec_p + offset_e) = dx2(p);
          phi_e(elec_p + offset_e) = phi(p);
          ux1_e(elec_p + offset_e) = px * pair_fac;
          ux2_e(elec_p + offset_e) = py * pair_fac;
          ux3_e(elec_p + offset_e) = pz * pair_fac;
          weight_e(elec_p + offset_e) = weight(p);
          tag_e(elec_p + offset_e) = ParticleTag::alive;

          i1_p(pos_p + offset_p) = i1(p);
          dx1_p(pos_p + offset_p) = dx1(p);
          i2_p(pos_p + offset_p) = i2(p);
          dx2_p(pos_p + offset_p) = dx2(p);
          phi_p(pos_p + offset_p) = phi(p);
          ux1_p(pos_p + offset_p) = px * pair_fac;
          ux2_p(pos_p + offset_p) = py * pair_fac;
          ux3_p(pos_p + offset_p) = pz * pair_fac;
          weight_p(pos_p + offset_p) = weight(p);
          tag_p(pos_p + offset_p) = ParticleTag::alive;

              ux1(p) *= new_fac;
              ux2(p) *= new_fac;
              ux3(p) *= new_fac;

      }

    });

        auto elec_ind_h = Kokkos::create_mirror(elec_ind);
        Kokkos::deep_copy(elec_ind_h, elec_ind);
        species_e.set_npart(offset_e + elec_ind_h());

        auto pos_ind_h = Kokkos::create_mirror(pos_ind);
        Kokkos::deep_copy(pos_ind_h, pos_ind);
        species_p.set_npart(offset_p + pos_ind_h());
        
//         auto& species = domain.species[s];

//         auto ux1    = species.ux1;
//         auto ux2    = species.ux2;
//         auto ux3    = species.ux3;
//         auto i1     = species.i1;
//         auto i2     = species.i2;
//         auto i3     = species.i3;
//         auto dx1    = species.dx1;
//         auto dx2    = species.dx2;
//         auto dx3    = species.dx3;
//         auto weight = species.weight;        
//         auto tag    = species.tag;

//         Kokkos::parallel_for(
//           "ResonantScattering",
//           species.rangeActiveParticles(),
//           Lambda(index_t p) {
//             if (tag(p) != ParticleTag::alive) {
//               return;
//             }

//             auto px      = ux1(p);
//             auto py      = ux2(p);
//             auto pz      = ux3(p);
//             auto gamma   = math::sqrt(ONE + SQR(px) + SQR(py) + SQR(pz));

//           // TODO: Calculate angular coordinate for setting limit close to axis
//             const vec_t<Dim::_3D> xi { i_di_to_Xi(i1(p), dx1(p)),
//                                   i_di_to_Xi(i2(p), dx2(p)) ,
//                                   i_di_to_Xi(i3(p), dx3(p)) };
//           //   coord_t<Dim2>     xs;
//           //   m_mblock.metric.x_Code2Sph(xi, xs);

//           //   if ((gamma > pp_thres) && (math::sin(xs[1]) > 0.1)) {
//             if ((gamma > pp_thres)) {

//               auto new_gamma = gamma - 2.0 * gamma_pairs;
//               auto new_fac = math::sqrt(SQR(new_gamma) - 1.0) / math::sqrt(SQR(gamma) - 1.0);
//               auto pair_fac = math::sqrt(SQR(gamma_pairs) - 1.0) / math::sqrt(SQR(gamma) - 1.0);

//       std::vector<real_t> x1p, x2p, x3p, ux1p, ux2p, ux3p;
//         x1p.push_back(xi[0]);
//         x2p.push_back(xi[1]);
//         x3p.push_back(xi[2]);
//         ux1p.push_back(px * pair_fac);
//         ux2p.push_back(py * pair_fac);
//         ux3p.push_back(pz * pair_fac);

//       const std::map<std::string, std::vector<real_t>> data_1 {
//         { "x1",  x1p},
//         { "x2",  x2p},
//         { "phi",  x3p},
//         {"ux1", ux1p},
//         {"ux2", ux2p},
//         {"ux3", ux3p}
//       };

//       arch::InjectGlobally<S, M>(global_domain, domain, (arch::spidx_t)1, data_1);

          // TODO: Inject positron-electron pair
              // init_prtl_2d_i_di(electrons,
              //                   elec_offset + elec_p,
              //                   species.i1(p),
              //                   species.i2(p),
              //                   species.dx1(p),
              //                   species.dx2(p),
              //                   px * pair_fac,
              //                   py * pair_fac,
              //                   pz * pair_fac,
              //                   species.weight(p));

              // init_prtl_2d_i_di(positrons,
              //                   pos_offset + pos_p,
              //                   species.i1(p),
              //                   species.i2(p),
              //                   species.dx1(p),
              //                   species.dx2(p),
              //                   px * pair_fac,
              //                   py * pair_fac,
              //                   pz * pair_fac,
              //                   species.weight(p));

              // ux1(p) *= new_fac;
              // ux2(p) *= new_fac;
              // ux3(p) *= new_fac;

            // }
          // });

      }
      
    }

  };

} // namespace user

#endif

