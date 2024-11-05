#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    /*
      Sets up magnetic and electric field components for the simulation.
      Must satisfy E = -v x B for Lorentz Force to be zero.

      @param bmag: magnetic field scaling
      @param btheta: magnetic field polar angle
      @param bphi: magnetic field azimuthal angle
      @param drift_ux: drift velocity in the x direction
    */
    InitFields(real_t bmag, real_t btheta, real_t bphi, real_t drift_ux)
      : Bmag { bmag * static_cast<real_t>(convert::deg2rad) }
      , Btheta { btheta * static_cast<real_t>(convert::deg2rad) }
      , Bphi { bphi * static_cast<real_t>(convert::deg2rad) }
      , Vx { drift_ux } {}

    // magnetic field components
    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return Bmag * math::cos(Btheta);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    // electric field components
    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return -Vx * Bmag * math::sin(Btheta) * math::cos(Bphi);
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return Vx * Bmag * math::sin(Btheta) * math::sin(Bphi);
    }

  private:
    const real_t Btheta, Bphi, Vx, Bmag;
  };

  /* Enforce resetting magnetic and electric field at the boundary */
  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t bmag, real_t btheta, real_t bphi, real_t drift_ux)
      : InitFields<D> { bmag, btheta, bphi, drift_ux } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;
    using InitFields<D>::bx3;

    using InitFields<D>::ex1;
    using InitFields<D>::ex2;
    using InitFields<D>::ex3;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t drift_ux, temperature;

    const real_t  Btheta, Bphi, Bmag;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M> { p }
      , drift_ux { p.template get<real_t>("setup.drift_ux") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , Bmag { p.template get<real_t>("setup.Bmag", ZERO) }
      , Btheta { p.template get<real_t>("setup.Btheta", ZERO) }
      , Bphi { p.template get<real_t>("setup.Bphi", ZERO) }
      , init_flds { Bmag, Btheta, Bphi, drift_ux } {}

    inline PGen() {}

    auto FieldDriver(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { Bmag, Btheta, Bphi, drift_ux };
    }

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      temperature,
                                                      -drift_ux,
                                                      in::x1);

      const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist,
        { 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        injector,
        1.0);
    }
  };

} // namespace user

#endif
