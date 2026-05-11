#include "kernels/twobody_interactions.hpp"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"

#include "framework/containers/particles.h"
#include "kernels/twobody_interactions.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>

using namespace ntt;

// Verifies that each paired particle from group1 and group2 lies in the same tile
struct SameTilePolicy {
  const array_t<int*> i1_1, i2_1;
  const array_t<int*> i1_2, i2_2;
  const array_t<int*> i1_3, i2_3;
  const array_t<int*> i1_4, i2_4;
  const ncells_t      tile_size;
  const ncells_t      ncx1, ncx2; // number of cells in each direction
  const ncells_t      ntx1, ntx2; // numbers of tiles
  array_t<int>        diff_tile_errors { "diff_tile_errors" };
  array_t<int>        tile_vol_errors { "tile_vol_errors" };

  SameTilePolicy(const array_t<int*>& i1_1,
                 const array_t<int*>& i2_1,
                 const array_t<int*>& i1_2,
                 const array_t<int*>& i2_2,
                 const array_t<int*>& i1_3,
                 const array_t<int*>& i2_3,
                 const array_t<int*>& i1_4,
                 const array_t<int*>& i2_4,
                 ncells_t             tile_size,
                 ncells_t             ncx1,
                 ncells_t             ncx2,
                 ncells_t             ntx1,
                 ncells_t             ntx2)
    : i1_1 { i1_1 }
    , i2_1 { i2_1 }
    , i1_2 { i1_2 }
    , i2_2 { i2_2 }
    , i1_3 { i1_3 }
    , i2_3 { i2_3 }
    , i1_4 { i1_4 }
    , i2_4 { i2_4 }
    , tile_size { tile_size }
    , ncx1 { ncx1 }
    , ncx2 { ncx2 }
    , ntx1 { ntx1 }
    , ntx2 { ntx2 } {}

  Inline void operator()(spidx_t sp1,
                         npart_t p1,
                         spidx_t sp2,
                         npart_t p2,
                         real_t  tile_volume) const {
    const auto x1_1 = (sp1 == 1u) ? i1_1(p1) : i1_2(p1);
    const auto x2_1 = (sp1 == 1u) ? i2_1(p1) : i2_2(p1);
    const auto x1_2 = (sp2 == 3u) ? i1_3(p2) : i1_4(p2);
    const auto x2_2 = (sp2 == 3u) ? i2_3(p2) : i2_4(p2);
    const auto t1   = static_cast<ncells_t>(x1_1 / tile_size) * ntx2 +
                    static_cast<ncells_t>(x2_1 / tile_size);
    const auto t2 = static_cast<ncells_t>(x1_2 / tile_size) * ntx2 +
                    static_cast<ncells_t>(x2_2 / tile_size);
    if (t1 != t2) {
      Kokkos::atomic_add(&diff_tile_errors(), 1);
    }

    real_t vol1 { ONE }, vol2 { ONE };
    {
      const auto ti1      = t1 / ntx2;
      const auto tj1      = t1 % ntx2;
      const auto i1_min_1 = ti1 * tile_size;
      const auto i1_max_1 = math::min(i1_min_1 + tile_size, ncx1);
      const auto i2_min_1 = tj1 * tile_size;
      const auto i2_max_1 = math::min(i2_min_1 + tile_size, ncx2);

      vol1 *= static_cast<real_t>(i1_max_1 - i1_min_1);
      vol1 *= static_cast<real_t>(i2_max_1 - i2_min_1);
    }
    {
      const auto ti2      = t2 / ntx2;
      const auto tj2      = t2 % ntx2;
      const auto i1_min_2 = ti2 * tile_size;
      const auto i1_max_2 = math::min(i1_min_2 + tile_size, ncx1);
      const auto i2_min_2 = tj2 * tile_size;
      const auto i2_max_2 = math::min(i2_min_2 + tile_size, ncx2);

      vol2 *= static_cast<real_t>(i1_max_2 - i1_min_2);
      vol2 *= static_cast<real_t>(i2_max_2 - i2_min_2);
    }
    vol1 *= SQR(0.03125);
    vol2 *= SQR(0.03125);

    if (not cmp::AlmostEqual(tile_volume, vol1) or
        not cmp::AlmostEqual(tile_volume, vol2)) {
      Kokkos::atomic_add(&tile_vol_errors(), 1);
    }
  }
};

void fill_random(array_t<int*>&        i1,
                 array_t<int*>&        i2,
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
      tag(p)   = ParticleTag::alive;
      rpool.free_state(gen);
    });
  Kokkos::fence();
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
    Particles<Dim::_2D, Coord::Cartesian> sp3 { 3u,
                                                "sp3",
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
    Particles<Dim::_2D, Coord::Cartesian> sp4 { 4u,
                                                "sp4",
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

    for (auto* sp : { &sp1, &sp2, &sp3, &sp4 }) {
      sp->set_npart(npart);
      fill_random(sp->i1, sp->i2, sp->tag, npart, nx1, nx2, random_pool);
    }

    const std::vector<const Particles<Dim::_2D, Coord::Cartesian>*> group1 = { &sp1,
                                                                               &sp2 };
    const std::vector<const Particles<Dim::_2D, Coord::Cartesian>*> group2 = { &sp3,
                                                                               &sp4 };

    auto policy = SameTilePolicy { sp1.i1, sp1.i2, sp2.i1, sp2.i2,    sp3.i1,
                                   sp3.i2, sp4.i1, sp4.i2, tile_size, nx1,
                                   nx2,    ntx1,   ntx2 };

    kernel::mink::TwoBodyInteraction<Dim::_2D>(group1,
                                               group2,
                                               ncells,
                                               {
                                                 { ZERO, ONE },
                                                 { ZERO, TWO }
    },
                                               tile_size,
                                               random_pool,
                                               policy);
    Kokkos::fence();

    {
      auto errors_h = Kokkos::create_mirror_view(policy.diff_tile_errors);
      Kokkos::deep_copy(errors_h, policy.diff_tile_errors);
      raise::ErrorIf(errors_h() != 0,
                     "paired particles from different tiles detected",
                     HERE);
    }

    {
      auto errors_h = Kokkos::create_mirror_view(policy.tile_vol_errors);
      Kokkos::deep_copy(errors_h, policy.tile_vol_errors);
      raise::ErrorIf(errors_h() != 0, "tile volume errors detected", HERE);
    }

  } catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    ntt::GlobalFinalize();
    return 1;
  }
  ntt::GlobalFinalize();
  return 0;
}
