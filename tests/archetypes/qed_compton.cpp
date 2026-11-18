#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "archetypes/qed/compton.h"
#include "framework/containers/particles.h"
#include "kernels/twobody_interactions.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <iostream>
#include <vector>

using namespace ntt;

void fill_random(array_t<int*>&        i1,
                 array_t<int*>&        i2,
                 array_t<real_t*>&     ux1,
                 array_t<real_t*>&     ux2,
                 array_t<real_t*>&     ux3,
                 array_t<real_t*>&     weight,
                 array_t<short*>&      tag,
                 npart_t               npart,
                 ncells_t              nx1,
                 ncells_t              nx2,
                 random_number_pool_t& rpool) {
  Kokkos::parallel_for(
    "FillRandom",
    npart,
    KOKKOS_LAMBDA(const npart_t p) {
      auto gen = rpool.get_state();
      i1(p)    = static_cast<int>(gen.urand() % static_cast<unsigned int>(nx1));
      i2(p)    = static_cast<int>(gen.urand() % static_cast<unsigned int>(nx2));
      ux1(p)   = Random<real_t>(gen) * TWO - ONE;
      ux2(p)   = Random<real_t>(gen) * TWO - ONE;
      ux3(p)   = Random<real_t>(gen) * TWO - ONE;
      weight(p) = ONE;
      tag(p)    = ParticleTag::alive;
      rpool.free_state(gen);
    });
  Kokkos::fence();
}

auto get_total_energy(bool              is_massive,
                      array_t<real_t*>& ux1,
                      array_t<real_t*>& ux2,
                      array_t<real_t*>& ux3,
                      npart_t           npart) -> real_t {
  real_t total_energy = ZERO;
  Kokkos::parallel_reduce(
    "TotalEnergy",
    npart,
    Lambda(const npart_t p, real_t& local_sum) {
      if (is_massive) {
        local_sum += U2GAMMA(ux1(p), ux2(p), ux3(p));

      } else {
        local_sum += NORM(ux1(p), ux2(p), ux3(p));
      }
    },
    total_energy);
  return total_energy;
}

auto get_total_momentum_in(in                dir,
                           array_t<real_t*>& ux1,
                           array_t<real_t*>& ux2,
                           array_t<real_t*>& ux3,
                           npart_t           npart) -> real_t {
  real_t total_momentum_in = ZERO;
  Kokkos::parallel_reduce(
    "TotalMomentumIn",
    npart,
    Lambda(const npart_t p, real_t& local_sum) {
      if (dir == in::x1) {
        local_sum += ux1(p);
      } else if (dir == in::x2) {
        local_sum += ux2(p);
      } else if (dir == in::x3) {
        local_sum += ux3(p);
      }
    },
    total_momentum_in);
  return total_momentum_in;
}

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);

  try {
    const ncells_t              nx1       = 32u;
    const ncells_t              nx2       = 64u;
    const ncells_t              tile_size = 3u;
    const std::vector<ncells_t> ncells    = { nx1, nx2 };
    const ncells_t              ntx1      = static_cast<ncells_t>(
      math::ceil(static_cast<double>(nx1) / static_cast<double>(tile_size)));
    const ncells_t ntx2 = static_cast<ncells_t>(
      math::ceil(static_cast<double>(nx2) / static_cast<double>(tile_size)));
    const npart_t        npart = 1000u;
    random_number_pool_t random_pool { 12345u };

    Particles<Dim::_2D, Coord::Cartesian> sp1 { 1u,
                                                "sp1",
                                                1.0f,
                                                1.0f,
                                                npart,
                                                0u,
                                                0u,
                                                ParticlePusher::BORIS,
                                                false,
                                                RadiativeDrag::NONE,
                                                EmissionType::NONE,
                                                0u,
                                                0u };
    Particles<Dim::_2D, Coord::Cartesian> sp2 { 2u,
                                                "sp2",
                                                1.0f,
                                                -1.0f,
                                                npart,
                                                0u,
                                                0u,
                                                ParticlePusher::BORIS,
                                                false,
                                                RadiativeDrag::NONE,
                                                EmissionType::NONE,
                                                0u,
                                                0u };
    Particles<Dim::_2D, Coord::Cartesian> sp3 { 3u,
                                                "sp3",
                                                0.0f,
                                                0.0f,
                                                npart,
                                                0u,
                                                0u,
                                                ParticlePusher::PHOTON,
                                                false,
                                                RadiativeDrag::NONE,
                                                EmissionType::NONE,
                                                0u,
                                                0u };

    for (auto* sp : { &sp1, &sp2, &sp3 }) {
      sp->set_npart(npart);
      fill_random(sp->i1,
                  sp->i2,
                  sp->ux1,
                  sp->ux2,
                  sp->ux3,
                  sp->weight,
                  sp->tag,
                  npart,
                  nx1,
                  nx2,
                  random_pool);
    }

    boundaries_t<real_t> extent {
      { ZERO, ONE },
      { -ONE, ONE }
    };
    const auto ppc0 = static_cast<real_t>(npart) / (nx1 * nx2);

    prm::Parameters params;
    params.set("compton_scattering.nominal_probability_density",
               static_cast<real_t>(1e-3));

    auto policy = arch::qed::ComptonScattering<Dim::_2D, true, true>(params,
                                                                     random_pool);
    policy.species[0] = static_cast<const ParticleArrays&>(sp1);
    policy.species[1] = static_cast<const ParticleArrays&>(sp2);
    policy.species[2] = static_cast<const ParticleArrays&>(sp3);

    std::array<real_t, 3> init_energies {
      get_total_energy(true, sp1.ux1, sp1.ux2, sp1.ux3, sp1.npart()),
      get_total_energy(true, sp2.ux1, sp2.ux2, sp2.ux3, sp2.npart()),
      get_total_energy(false, sp3.ux1, sp3.ux2, sp3.ux3, sp3.npart())
    };
    std::array<real_t, 3> init_moms_x1 {
      get_total_momentum_in(in::x1, sp1.ux1, sp1.ux2, sp1.ux3, sp1.npart()),
      get_total_momentum_in(in::x1, sp2.ux1, sp2.ux2, sp2.ux3, sp2.npart()),
      get_total_momentum_in(in::x1, sp3.ux1, sp3.ux2, sp3.ux3, sp3.npart())
    };
    std::array<real_t, 3> init_moms_x2 {
      get_total_momentum_in(in::x2, sp1.ux1, sp1.ux2, sp1.ux3, sp1.npart()),
      get_total_momentum_in(in::x2, sp2.ux1, sp2.ux2, sp2.ux3, sp2.npart()),
      get_total_momentum_in(in::x2, sp3.ux1, sp3.ux2, sp3.ux3, sp3.npart())
    };
    std::array<real_t, 3> init_moms_x3 {
      get_total_momentum_in(in::x3, sp1.ux1, sp1.ux2, sp1.ux3, sp1.npart()),
      get_total_momentum_in(in::x3, sp2.ux1, sp2.ux2, sp2.ux3, sp2.npart()),
      get_total_momentum_in(in::x3, sp3.ux1, sp3.ux2, sp3.ux3, sp3.npart())
    };

    for (int i = 0; i < 1000; ++i) {
      kernel::mink::TwoBodyInteraction<Dim::_2D>({ &sp1, &sp2 },
                                                 { &sp3 },
                                                 ncells,
                                                 extent,
                                                 tile_size,
                                                 ppc0,
                                                 random_pool,
                                                 policy);
    }

    {
      std::array<real_t, 3> fin_energies {
        get_total_energy(true, sp1.ux1, sp1.ux2, sp1.ux3, sp1.npart()),
        get_total_energy(true, sp2.ux1, sp2.ux2, sp2.ux3, sp2.npart()),
        get_total_energy(false, sp3.ux1, sp3.ux2, sp3.ux3, sp3.npart())
      };
      std::array<real_t, 3> fin_moms_x1 {
        get_total_momentum_in(in::x1, sp1.ux1, sp1.ux2, sp1.ux3, sp1.npart()),
        get_total_momentum_in(in::x1, sp2.ux1, sp2.ux2, sp2.ux3, sp2.npart()),
        get_total_momentum_in(in::x1, sp3.ux1, sp3.ux2, sp3.ux3, sp3.npart())
      };
      std::array<real_t, 3> fin_moms_x2 {
        get_total_momentum_in(in::x2, sp1.ux1, sp1.ux2, sp1.ux3, sp1.npart()),
        get_total_momentum_in(in::x2, sp2.ux1, sp2.ux2, sp2.ux3, sp2.npart()),
        get_total_momentum_in(in::x2, sp3.ux1, sp3.ux2, sp3.ux3, sp3.npart())
      };
      std::array<real_t, 3> fin_moms_x3 {
        get_total_momentum_in(in::x3, sp1.ux1, sp1.ux2, sp1.ux3, sp1.npart()),
        get_total_momentum_in(in::x3, sp2.ux1, sp2.ux2, sp2.ux3, sp2.npart()),
        get_total_momentum_in(in::x3, sp3.ux1, sp3.ux2, sp3.ux3, sp3.npart())
      };

      const auto fin_energy = fin_energies[0] + fin_energies[1] + fin_energies[2];
      const auto init_energy = init_energies[0] + init_energies[1] +
                               init_energies[2];
      const auto fin_mom_x1 = fin_moms_x1[0] + fin_moms_x1[1] + fin_moms_x1[2];
      const auto init_mom_x1 = init_moms_x1[0] + init_moms_x1[1] + init_moms_x1[2];
      const auto fin_mom_x2 = fin_moms_x2[0] + fin_moms_x2[1] + fin_moms_x2[2];
      const auto init_mom_x2 = init_moms_x2[0] + init_moms_x2[1] + init_moms_x2[2];
      const auto fin_mom_x3 = fin_moms_x3[0] + fin_moms_x3[1] + fin_moms_x3[2];
      const auto init_mom_x3 = init_moms_x3[0] + init_moms_x3[1] + init_moms_x3[2];

      const auto err_energy = (fin_energy - init_energy) / init_energy;
      const auto err_mom_x1 = (fin_mom_x1 - init_mom_x1) /
                              (std::abs(init_mom_x1) + 1e-10);
      const auto err_mom_x2 = (fin_mom_x2 - init_mom_x2) /
                              (std::abs(init_mom_x2) + 1e-10);
      const auto err_mom_x3 = (fin_mom_x3 - init_mom_x3) /
                              (std::abs(init_mom_x3) + 1e-10);

      raise::ErrorIf(err_energy > 1e-5,
                     fmt::format("energy is not conserved %e -> %e [%e]",
                                 init_energy,
                                 fin_energy,
                                 err_energy),
                     HERE);
      raise::ErrorIf(err_mom_x1 > 1e-5,
                     fmt::format("x1 momentum is not conserved %e -> %e [%e]",
                                 init_mom_x1,
                                 fin_mom_x1,
                                 err_mom_x1),
                     HERE);
      raise::ErrorIf(err_mom_x2 > 1e-5,
                     fmt::format("x2 momentum is not conserved %e -> %e [%e]",
                                 init_mom_x2,
                                 fin_mom_x2,
                                 err_mom_x2),
                     HERE);
      raise::ErrorIf(err_mom_x3 > 1e-5,
                     fmt::format("x3 momentum is not conserved %e -> %e [%e]",
                                 init_mom_x3,
                                 fin_mom_x3,
                                 err_mom_x3),
                     HERE);
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    ntt::GlobalFinalize();
    return 1;
  }
  ntt::GlobalFinalize();
  return 0;
}
