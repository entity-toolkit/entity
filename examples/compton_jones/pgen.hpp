#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"

#include "archetypes/particle_injector.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct DeltaDistribution {
    const real_t         energy0;
    bool                 monodirectional;
    random_number_pool_t random_pool;

    DeltaDistribution(real_t                energy0,
                      bool                  monodirectional,
                      random_number_pool_t& random_pool)
      : energy0 { energy0 }
      , monodirectional { monodirectional }
      , random_pool { random_pool } {}

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
      if (not monodirectional) {
        auto gen  = random_pool.get_state();
        auto rnd1 = Random<real_t>(gen);
        auto rnd2 = Random<real_t>(gen);
        random_pool.free_state(gen);
        // random direction
        const auto phi = static_cast<real_t>(constant::TWO_PI) * rnd1;
        const auto ct  = 2.0 * rnd2 - 1.0;
        const auto st  = math::sqrt(1.0 - ct * ct);
        v[0]           = energy0 * st * math::cos(phi);
        v[1]           = energy0 * st * math::sin(phi);
        v[2]           = energy0 * ct;
      } else {
        v[0] = energy0;
        v[1] = 0.0;
        v[2] = 0.0;
      }
    }
  };

  template <SimEngine::type S, class M>
  struct PGen {

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

    PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , metadomain { m } {}

    void InitPrtls(Domain<S, M>& domain) {
      auto delta_electrons = DeltaDistribution<M::Dim> {
        params.template get<real_t>("setup.electron_4vel"),
        true,
        domain.random_pool()
      };
      arch::InjectUniform<S, M, decltype(delta_electrons)>(params,
                                                           domain,
                                                           1u,
                                                           delta_electrons,
                                                           ONE);
    }

    void CustomPostStep(timestep_t /*step*/, simtime_t /*time*/, Domain<S, M>& domain) {
      // copy all photons from species #2 (idx 1) to #3 (idx 2) with an offset
      const auto offset     = domain.species[2].npart();
      const auto new_copies = domain.species[1].npart();
      const auto new_size   = offset + new_copies;
      const auto from_slice = prtl_slice_t { 0, new_copies };
      const auto to_slice   = prtl_slice_t { offset, new_size };

      Kokkos::deep_copy(Kokkos::subview(domain.species[2].i1, to_slice),
                        Kokkos::subview(domain.species[1].i1, from_slice));
      Kokkos::deep_copy(Kokkos::subview(domain.species[2].i1_prev, to_slice),
                        Kokkos::subview(domain.species[1].i1_prev, from_slice));
      Kokkos::deep_copy(Kokkos::subview(domain.species[2].dx1, to_slice),
                        Kokkos::subview(domain.species[1].dx1, from_slice));
      Kokkos::deep_copy(Kokkos::subview(domain.species[2].dx1_prev, to_slice),
                        Kokkos::subview(domain.species[1].dx1_prev, from_slice));
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        Kokkos::deep_copy(Kokkos::subview(domain.species[2].i2, to_slice),
                          Kokkos::subview(domain.species[1].i2, from_slice));
        Kokkos::deep_copy(Kokkos::subview(domain.species[2].i2_prev, to_slice),
                          Kokkos::subview(domain.species[1].i2_prev, from_slice));
        Kokkos::deep_copy(Kokkos::subview(domain.species[2].dx2, to_slice),
                          Kokkos::subview(domain.species[1].dx2, from_slice));
        Kokkos::deep_copy(Kokkos::subview(domain.species[2].dx2_prev, to_slice),
                          Kokkos::subview(domain.species[1].dx2_prev, from_slice));
      }
      if constexpr (M::Dim == Dim::_3D) {
        Kokkos::deep_copy(Kokkos::subview(domain.species[2].i3, to_slice),
                          Kokkos::subview(domain.species[1].i3, from_slice));
        Kokkos::deep_copy(Kokkos::subview(domain.species[2].i3_prev, to_slice),
                          Kokkos::subview(domain.species[1].i3_prev, from_slice));
        Kokkos::deep_copy(Kokkos::subview(domain.species[2].dx3, to_slice),
                          Kokkos::subview(domain.species[1].dx3, from_slice));
        Kokkos::deep_copy(Kokkos::subview(domain.species[2].dx3_prev, to_slice),
                          Kokkos::subview(domain.species[1].dx3_prev, from_slice));
      }
      Kokkos::deep_copy(Kokkos::subview(domain.species[2].ux1, to_slice),
                        Kokkos::subview(domain.species[1].ux1, from_slice));
      Kokkos::deep_copy(Kokkos::subview(domain.species[2].ux2, to_slice),
                        Kokkos::subview(domain.species[1].ux2, from_slice));
      Kokkos::deep_copy(Kokkos::subview(domain.species[2].ux3, to_slice),
                        Kokkos::subview(domain.species[1].ux3, from_slice));
      Kokkos::deep_copy(Kokkos::subview(domain.species[2].weight, to_slice),
                        Kokkos::subview(domain.species[1].weight, from_slice));
      Kokkos::deep_copy(Kokkos::subview(domain.species[2].tag, to_slice),
                        Kokkos::subview(domain.species[1].tag, from_slice));

      domain.species[1].set_npart(0);
      domain.species[2].set_npart(new_size);

      auto delta_photons = DeltaDistribution<M::Dim> {
        params.template get<real_t>("setup.photon_energy"),
        false,
        domain.random_pool()
      };
      arch::InjectUniform<S, M, decltype(delta_photons)>(params,
                                                         domain,
                                                         2u,
                                                         delta_photons,
                                                         ONE);
    }
  };

} // namespace user

#endif
