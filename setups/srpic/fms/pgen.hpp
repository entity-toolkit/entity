#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include <iostream>

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/field_setter.h"

namespace user {
  using namespace ntt;


  template <Dimension D>
  struct ExternalFields {
    ExternalFields(real_t dx, real_t fb0, real_t lgrad, real_t pow)
      : dx { dx }
      , fb0 { fb0 }
      , lgrad { lgrad }
      , pow { pow } {}

    const std::vector<unsigned short> species { 1, 2 };

    ExternalFields() = default;

    Inline auto bx3(const unsigned short&,
                    const real_t& time,
                    const coord_t<D>& x_Ph) const -> real_t {
      real_t xshift = time - std::fmod(time, dx);
      return fb0 * math::pow( (x_Ph[0] + xshift)/lgrad + ONE, pow);
    }

  private:
    real_t dx, fb0, lgrad, pow;
  };

  template <Dimension D>
  struct InitFields {
    InitFields(real_t Bz0, real_t drift_ux)
     : fb0 { Bz0 } 
     , fu0 { drift_ux } {}

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return  ZERO;
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
    const real_t fb0, fu0;
  };

  template <SimEngine::type S, class M>
  struct Cathode : public arch::SpatialDistribution<S, M> {
    Cathode(const M& metric, real_t x1c)
      : arch::SpatialDistribution<S, M> { metric }
      , x1c { x1c } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      if (x_Ph[0] > x1c) {
        return ONE;
      } else {
        return ZERO;
      }
    }

  private:
    const real_t x1c;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    Metadomain<S, M>& global_domain;
    ExternalFields<M::PrtlDim> ext_force;

    const real_t drift_ux, temperature, Bz0, deltax, amp, om, damp;
    const std::size_t imax, iosc;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , global_domain { global_domain }
      , ext_force { p.template get<real_t>("scales.dx0"),
                    p.template get<real_t>("setup.Bz0"),
                    p.template get<real_t>("setup.lgrad"),
                    p.template get<real_t>("setup.pow") }
      , drift_ux      { p.template get<real_t>("setup.drift_ux") }
      , temperature   { p.template get<real_t>("setup.temperature") }
      , Bz0           { p.template get<real_t>("setup.Bz0") }
      , amp           { p.template get<real_t>("setup.amp") }
      , om            { p.template get<real_t>("setup.omega") }
      , damp          { p.template get<real_t>("setup.damp") }
      , imax          { static_cast<std::size_t>(p.template get<int>("setup.imax"))}
      , iosc          { static_cast<std::size_t>(p.template get<int>("setup.iosc"))}
      , deltax        { p.template get<real_t>("scales.dx0")}
      , init_flds     { Bz0, drift_ux }  {
        std::cout << "drift_ux: " << drift_ux << std::endl;
        std::cout << "temperature: " << temperature << std::endl;
        std::cout << "Bz0: " << Bz0 << std::endl;
        std::cout << "imax: " << imax << std::endl;
        std::cout << "iosc: " << iosc << std::endl;
        std::cout << "deltax: " << deltax << std::endl;
      }

    // inline PGen() {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      temperature,
                                                      drift_ux,
                                                      in::x1);
      const auto injector    = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist,
        { 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        injector,
        ONE);
    }

    void CustomPostStep(std::size_t nstep, long double , Domain<S, M>& domain) {

      auto EM = domain.fields.em;
      auto BCKP = domain.fields.bckp;
      auto last_index = domain.mesh.n_active()[0];
      auto x1c = domain.mesh.extent(in::x1).second;
      auto fb0 = this->Bz0;
      // auto fu0 = this->drift_ux;
      auto imax = this->imax;
      auto iosc = this->iosc;
      auto deltax = this->deltax;
      auto amp = this->amp;
      auto om = this->om;
      auto damp = this->damp;

      if ( nstep%2 == 0 ) {
        for (auto& species : domain.species) {
          auto i1 = species.i1;
          auto tag = species.tag;

          Kokkos::parallel_for("offset",
            species.rangeActiveParticles(),
            Lambda(index_t p) {
              i1(p) -= 1;
              if ( i1(p) <= 0 ) {
                tag(p) = ParticleTag::dead;
              }
            });

        }

        Kokkos::deep_copy(BCKP,
                          EM);

        if constexpr (D == Dim::_1D) {
          Kokkos::parallel_for("field_loop",
            domain.mesh.rangeActiveCells(),
            Lambda(index_t i1) {
              EM(i1, em::ex1) = BCKP(i1 + 1, em::ex1);
              EM(i1, em::ex2) = BCKP(i1 + 1, em::ex2);
              EM(i1, em::ex3) = BCKP(i1 + 1, em::ex3);
              EM(i1, em::bx1) = BCKP(i1 + 1, em::bx1);
              EM(i1, em::bx2) = BCKP(i1 + 1, em::bx2);
              EM(i1, em::bx3) = BCKP(i1 + 1, em::bx3);
              if (i1 >= imax) {
                EM(i1, em::ex1) = ZERO;
                EM(i1, em::ex2) = ZERO;
                EM(i1, em::ex3) = ZERO;
                EM(i1, em::bx1) = ZERO;
                EM(i1, em::bx2) = ZERO;
                EM(i1, em::bx3) = ZERO;
              }
            });
        }
        else if constexpr (D == Dim::_2D) {
          Kokkos::parallel_for("field_loop",
            domain.mesh.rangeActiveCells(),
            Lambda(index_t i1, index_t i2) {
              EM(i1, i2, em::ex1) = BCKP(i1 + 1, i2, em::ex1);
              EM(i1, i2, em::ex2) = BCKP(i1 + 1, i2, em::ex2);
              EM(i1, i2, em::ex3) = BCKP(i1 + 1, i2, em::ex3);
              EM(i1, i2, em::bx1) = BCKP(i1 + 1, i2, em::bx1);
              EM(i1, i2, em::bx2) = BCKP(i1 + 1, i2, em::bx2);
              EM(i1, i2, em::bx3) = BCKP(i1 + 1, i2, em::bx3);
              if (i1 >= imax) {
                EM(i1, i2, em::ex1) = ZERO;
                EM(i1, i2, em::ex2) = ZERO;
                EM(i1, i2, em::ex3) = ZERO;
                EM(i1, i2, em::bx1) = ZERO;
                EM(i1, i2, em::bx2) = ZERO;
                EM(i1, i2, em::bx3) = ZERO;
              }
            });
        }
        else if constexpr (D == Dim::_3D) {
          Kokkos::parallel_for("field_loop",
            domain.mesh.rangeActiveCells(),
            Lambda(index_t i1, index_t i2, index_t i3) {
              EM(i1, i2, i3, em::ex1) = BCKP(i1 + 1, i2, i3, em::ex1);
              EM(i1, i2, i3, em::ex2) = BCKP(i1 + 1, i2, i3, em::ex2);
              EM(i1, i2, i3, em::ex3) = BCKP(i1 + 1, i2, i3, em::ex3);
              EM(i1, i2, i3, em::bx1) = BCKP(i1 + 1, i2, i3, em::bx1);
              EM(i1, i2, i3, em::bx2) = BCKP(i1 + 1, i2, i3, em::bx2);
              EM(i1, i2, i3, em::bx3) = BCKP(i1 + 1, i2, i3, em::bx3);
              if (i1 >= imax) {
                EM(i1, i2, i3, em::ex1) = ZERO;
                EM(i1, i2, i3, em::ex2) = ZERO;
                EM(i1, i2, i3, em::ex3) = ZERO;
                EM(i1, i2, i3, em::bx1) = ZERO;
                EM(i1, i2, i3, em::bx2) = ZERO;
                EM(i1, i2, i3, em::bx3) = ZERO;
              }
            });
        }

        global_domain.CommunicateFields(domain, Comm::B | Comm::E | Comm::J);



        const auto energy_dist = arch::Maxwellian<S, M>(domain.mesh.metric,
                                    domain.random_pool,
                                    temperature,
                                    drift_ux,
                                    in::x1);
        const auto spatial_dist = Cathode<S, M>(domain.mesh.metric, x1c - deltax);
        const auto injectornf = arch::NonUniformInjector<S, M, arch::Maxwellian, Cathode>(
          energy_dist,
          spatial_dist,
          { 1, 2 });

        arch::InjectNonUniform<S, M, arch::NonUniformInjector<S, M, arch::Maxwellian, Cathode>>(
          params,
          domain,
          injectornf,
          ONE);
        
          std::cout << "x1c: " << x1c << std::endl;
          std::cout << "x1c - deltax: " << x1c - deltax << std::endl;
      }

      if constexpr (D == Dim::_1D) {
        if (iosc > nstep/2) {
          Kokkos::parallel_for("field_loop",
            domain.mesh.rangeActiveCells(),
            Lambda(index_t i1) {
              if (i1 <= iosc - nstep/2  ) {
                EM(i1, em::ex1) = ZERO;
                EM(i1, em::ex2) = - amp * fb0 * math::sin(2.0 * constant::PI * nstep * om ) * ( ONE - math::exp(- std::pow( nstep/static_cast<real_t>(damp) , 2) ) );
                EM(i1, em::ex3) = ZERO;
                EM(i1, em::bx1) = ZERO;
                EM(i1, em::bx2) = ZERO;
                EM(i1, em::bx3) = - amp * fb0 * math::sin(2.0 * constant::PI * nstep * om ) * ( ONE - math::exp(- std::pow( nstep/static_cast<real_t>(damp) , 2) ) );
              }
              // if (i1 < 4) {
              //   EM(i1, em::ex1) = ZERO;
              //   EM(i1, em::ex2) = ZERO;
              //   EM(i1, em::ex3) = ZERO;
              //   EM(i1, em::bx1) = ZERO;
              //   EM(i1, em::bx2) = ZERO;
              //   EM(i1, em::bx3) = ZERO;
              // }
          });
        }
      }
      else if constexpr (D == Dim::_2D) {
        if (iosc > nstep/2) {
          Kokkos::parallel_for("field_loop",
            domain.mesh.rangeActiveCells(),
            Lambda(index_t i1, index_t i2) {
              if (i1 <= iosc - nstep/2  ) {
                EM(i1, i2, em::ex1) = ZERO;
                EM(i1, i2, em::ex2) = - amp * fb0 * math::sin(2.0 * constant::PI * nstep * om ) * ( ONE - math::exp(- std::pow( nstep/static_cast<real_t>(damp) , 2) ) );
                EM(i1, i2, em::ex3) = ZERO;
                EM(i1, i2, em::bx1) = ZERO;
                EM(i1, i2, em::bx2) = ZERO;
                EM(i1, i2, em::bx3) = - amp * fb0 * math::sin(2.0 * constant::PI * nstep * om ) * ( ONE - math::exp(- std::pow( nstep/static_cast<real_t>(damp) , 2) ) );
              }
          });
        }
      }
      else if constexpr (D == Dim::_3D) {
        if (iosc > nstep/2) {
          Kokkos::parallel_for("field_loop",
            domain.mesh.rangeActiveCells(),
            Lambda(index_t i1, index_t i2, index_t i3) {
              if (i1 <= iosc - nstep/2  ) {
                EM(i1, i2, i3, em::ex1) = ZERO;
                EM(i1, i2, i3, em::ex2) = - amp * fb0 * math::sin(2.0 * constant::PI * nstep * om ) * ( ONE - math::exp(- std::pow( nstep/static_cast<real_t>(damp) , 2) ) );
                EM(i1, i2, i3, em::ex3) = ZERO;
                EM(i1, i2, i3, em::bx1) = ZERO;
                EM(i1, i2, i3, em::bx2) = ZERO;
                EM(i1, i2, i3, em::bx3) = - amp * fb0 * math::sin(2.0 * constant::PI * nstep * om ) * ( ONE - math::exp(- std::pow( nstep/static_cast<real_t>(damp) , 2) ) );
              }
          });
        }
      }

    } // CustomPostStep  
  }; // PGEN
} // namespace user

#endif
