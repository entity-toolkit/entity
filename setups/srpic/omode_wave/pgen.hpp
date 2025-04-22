#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    // Simplified external current driver: single x1-directional mode with spatial/temporal variation
    template <Dimension D>
    struct ExternalCurrent {
        ExternalCurrent(real_t amplitude, int num_waves_x, int num_waves_y, int num_waves_z,
                        real_t frequency, real_t Lx, real_t Ly, real_t Lz)
            : A(amplitude), omega(frequency), Lx(Lx), Ly(Ly), Lz(Lz) {

            // Calculate wavevector components based on number of desired waves
            if constexpr (D == Dim::_2D) {
                kx = constant::TWO_PI * num_waves_x / Lx;
                ky = constant::TWO_PI * num_waves_y / Ly;
            }
            if constexpr (D == Dim::_3D) {
                kx = constant::TWO_PI * num_waves_x / Lx;
                ky = constant::TWO_PI * num_waves_y / Ly;
                kz = constant::TWO_PI * num_waves_z / Lz;
            }
        }

        Inline auto jx1(const coord_t<D>& x_Ph, real_t time) const -> real_t {
            if constexpr (D == Dim::_2D) {
                auto phase = kx * x_Ph[0] + ky * x_Ph[1] - omega * time;
                return A * math::cos(phase);
            }
            if constexpr (D == Dim::_3D) {
                auto phase = kx * x_Ph[0] + ky * x_Ph[1] + kz * x_Ph[2] - omega * time;
                return A * math::cos(phase);
            }
        }

        Inline auto jx2(const coord_t<D>&) const -> real_t { return ZERO; }
        Inline auto jx3(const coord_t<D>&) const -> real_t { return ZERO; }

    private:
        real_t A, omega, kx, ky, kz;
        real_t Lx, Ly, Lz;
    };


    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions =
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t sx1, sx2, sx3;
    const real_t temp, amplitude;
    const real_t nwave_x, nwave_y, nwave_z, frequency;

    ExternalCurrent<D> ExternalCurrent;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , sx1 { global_domain.mesh().extent(in::x1).second -
              global_domain.mesh().extent(in::x1).first }
      , sx2 { global_domain.mesh().extent(in::x2).second -
              global_domain.mesh().extent(in::x2).first }
      , sx3 { global_domain.mesh().extent(in::x3).second -
              global_domain.mesh().extent(in::x3).first }
      , temp { p.template get<real_t>("setup.temp") }
      , amplitude { p.template get<real_t>("setup.amplitude") }
      , nwave_x { p.template get<real_t>("setup.nwave_x") }
      , nwave_y { p.template get<real_t>("setup.nwave_y") }
      , nwave_z { p.template get<real_t>("setup.nwave_z") }
      , frequency { p.template get<real_t>("setup.frequency") }
      , ExternalCurrent {  amplitude, nwave_x, nwave_y, nwave_z, frequency, sx1, sx2, sx3 }

      {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,local_domain.random_pool,temp);
      const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(energy_dist,{ 1, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(params,local_domain,injector,HALF);
    }
  };

} // namespace user

#endif
