#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include "framework/containers/particles.h"
#include "framework/domain/grid.h"

#include <iostream>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    {
      // 2D
      auto grid = ntt::Grid<Dim::_2D> {
        {           10u,             30u },
        { { -5.0, 5.0 }, { -15.0, 15.0 } }
      };
      auto prtls = ntt::Particles<Dim::_2D, ntt::Coord::Cartesian>(
        1,
        "test",
        1.0f,
        1.0f,
        100u,
        0u,
        0u,
        ntt::ParticlePusher::BORIS,
        false,
        ntt::RadiativeDrag::NONE,
        ntt::EmissionType::NONE,
        2u,
        1u);
      auto& i1_p      = prtls.i1;
      auto& i2_p      = prtls.i2;
      auto& i1_prev_p = prtls.i1_prev;
      auto& i2_prev_p = prtls.i2_prev;
      auto& tag_p     = prtls.tag;
      auto& weight_p  = prtls.weight;
      auto& pld_r     = prtls.pld_r;
      auto& pld_i     = prtls.pld_i;
      Kokkos::parallel_for(
        "InitParticles",
        prtls.maxnpart(),
        Lambda(prtlidx_t p) {
          if (p < 66u) {
            tag_p(p) = (p % 10u == 0u) ? ntt::ParticleTag::dead
                                       : ntt::ParticleTag::alive;
            if (p % 4u == 0u) {
              i1_p(p)     = 8u;
              i2_p(p)     = 2u;
              weight_p(p) = 0.0;
            } else if (p % 4u == 1u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 8u;
              weight_p(p) = 1.0;
            } else if (p % 4u == 2u) {
              i1_p(p)     = 5u;
              i2_p(p)     = 15u;
              weight_p(p) = 2.0;
            } else {
              i1_p(p)     = 0u;
              i2_p(p)     = 23u;
              weight_p(p) = 3.0;
            }
            // team_policy keys on min(i, i_prev); without a meaningful
            // i_prev every key would collapse to 0. Set i_prev = i so the
            // tile key reduces to the particle's current cell.
            i1_prev_p(p) = i1_p(p);
            i2_prev_p(p) = i2_p(p);
            pld_r(p, 0)  = weight_p(p) + static_cast<real_t>(0.5);
            pld_r(p, 1)  = weight_p(p) + static_cast<real_t>(10.5);
            pld_i(p, 0)  = static_cast<npart_t>(weight_p(p) + 10.0);
          } else {
            tag_p(p) = ntt::ParticleTag::dead;
          }
          if (tag_p(p) == ntt::ParticleTag::dead) {
            weight_p(p) = -1.0;
          }
        });
      prtls.set_npart(66u);

      prtls.SortSpatially(grid);

      auto i1_h     = Kokkos::create_mirror_view(prtls.i1);
      auto i2_h     = Kokkos::create_mirror_view(prtls.i2);
      auto tag_h    = Kokkos::create_mirror_view(prtls.tag);
      auto weight_h = Kokkos::create_mirror_view(prtls.weight);
      auto pld_r_h  = Kokkos::create_mirror_view(prtls.pld_r);
      auto pld_i_h  = Kokkos::create_mirror_view(prtls.pld_i);
      Kokkos::deep_copy(i1_h, prtls.i1);
      Kokkos::deep_copy(i2_h, prtls.i2);
      Kokkos::deep_copy(tag_h, prtls.tag);
      Kokkos::deep_copy(weight_h, prtls.weight);
      Kokkos::deep_copy(pld_r_h, prtls.pld_r);
      Kokkos::deep_copy(pld_i_h, prtls.pld_i);

      // Tile geometry, mirroring sort::PositionToTileIndex. T = 1 (no
      // team_policy) reproduces the legacy per-cell ordering.
#if defined(TEAM_POLICY)
      const ncells_t T = static_cast<ncells_t>(TEAM_POLICY_TILE_SIZE);
#else
      const ncells_t T = 1u;
#endif
      const auto     na   = grid.n_active();
      const ncells_t ntx2 = (na[1] + T - 1u) / T;
      const auto     tile_of = [&](int a, int b) -> ncells_t {
        return (static_cast<ncells_t>(a) / T) * ntx2 +
               (static_cast<ncells_t>(b) / T);
      };

      // SortSpatially is order-by-tile, not order-by-cell: assert the
      // invariants that hold for any tile size rather than a hardwired
      // permutation. (1) alive particles form a prefix sorted by
      // non-decreasing tile index; (2) every SoA member is permuted by the
      // *same* permutation, so each alive slot still satisfies
      // pld == f(weight); (3) no alive particle is lost. Only [0, npart())
      // is defined after a sort. The team_policy path compacts — it drops
      // the dead, so npart() equals the alive count and [0, npart()) is
      // entirely alive; the legacy (non-team) path keeps the dead as a
      // weight == -1 suffix, leaving npart() unchanged. Iterating
      // [0, npart()) exercises both: the prefix-sorted / no-alive-after-dead
      // checks below hold either way.
#if defined(TEAM_POLICY)
      raise::ErrorIf(prtls.npart() != 59u,
                     "team_policy sort must compact: npart() should equal "
                     "the alive count",
                     HERE);
#else
      raise::ErrorIf(prtls.npart() != 66u,
                     "legacy sort should leave npart() unchanged",
                     HERE);
#endif
      bool     seen_dead   = false;
      bool     have_prev   = false;
      ncells_t prev_tile   = 0u;
      npart_t  n_alive_obs = 0u;
      for (auto p { 0u }; p < prtls.npart(); ++p) {
        if (tag_h(p) != ntt::ParticleTag::alive) {
          seen_dead = true;
          raise::ErrorIf(weight_h(p) != -1.0,
                         "dead particle has unexpected weight",
                         HERE);
          continue;
        }
        raise::ErrorIf(seen_dead,
                       "alive particle after a dead one (not sorted to prefix)",
                       HERE);
        const auto tile = tile_of(i1_h(p), i2_h(p));
        raise::ErrorIf(have_prev && (tile < prev_tile),
                       "alive particles not sorted by tile index",
                       HERE);
        prev_tile = tile;
        have_prev = true;
        ++n_alive_obs;
        raise::ErrorIf(pld_r_h(p, 0) != weight_h(p) + static_cast<real_t>(0.5),
                       "error in sorting particle real payload 0",
                       HERE);
        raise::ErrorIf(pld_r_h(p, 1) != weight_h(p) + static_cast<real_t>(10.5),
                       "error in sorting particle real payload 1",
                       HERE);
        raise::ErrorIf(
          pld_i_h(p, 0) !=
            static_cast<npart_t>(weight_h(p) + static_cast<real_t>(10.0)),
          "error in sorting particle integer payload 0",
          HERE);
      }
      raise::ErrorIf(n_alive_obs != 59u,
                     "wrong number of alive particles after sort",
                     HERE);
    }
    {
      // 3D
      auto grid = ntt::Grid<Dim::_3D> {
        {            6u,            7u,            8u },
        { { -3.0, 3.0 }, { -3.5, 3.5 }, { -4.0, 4.0 } }
      };
      auto prtls = ntt::Particles<Dim::_3D, ntt::Coord::Cartesian>(
        1,
        "test",
        1.0f,
        1.0f,
        100u,
        0u,
        0u,
        ntt::ParticlePusher::BORIS,
        false,
        ntt::RadiativeDrag::NONE,
        ntt::EmissionType::NONE,
        0u,
        0u);
      auto& i1_p      = prtls.i1;
      auto& i2_p      = prtls.i2;
      auto& i3_p      = prtls.i3;
      auto& i1_prev_p = prtls.i1_prev;
      auto& i2_prev_p = prtls.i2_prev;
      auto& i3_prev_p = prtls.i3_prev;
      auto& tag_p     = prtls.tag;
      auto& weight_p  = prtls.weight;
      Kokkos::parallel_for(
        "InitParticles",
        prtls.maxnpart(),
        Lambda(prtlidx_t p) {
          if (p < 66u) {
            tag_p(p) = (p % 10u == 0u) ? ntt::ParticleTag::dead
                                       : ntt::ParticleTag::alive;
            if (p % 5u == 0u) {
              i1_p(p)     = 3u;
              i2_p(p)     = 2u;
              i3_p(p)     = 7u;
              weight_p(p) = 0.0;
            } else if (p % 5u == 1u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 4u;
              i3_p(p)     = 3u;
              weight_p(p) = 1.0;
            } else if (p % 5u == 2u) {
              i1_p(p)     = 2u;
              i2_p(p)     = 6u;
              i3_p(p)     = 6u;
              weight_p(p) = 2.0;
            } else if (p % 5u == 3u) {
              i1_p(p)     = 3u;
              i2_p(p)     = 6u;
              i3_p(p)     = 6u;
              weight_p(p) = 3.0;
            } else {
              i1_p(p)     = 0u;
              i2_p(p)     = 6u;
              i3_p(p)     = 7u;
              weight_p(p) = 4.0;
            }
            // see 2D block: i_prev = i so the team_policy tile key reduces
            // to the particle's current cell.
            i1_prev_p(p) = i1_p(p);
            i2_prev_p(p) = i2_p(p);
            i3_prev_p(p) = i3_p(p);
          } else {
            tag_p(p) = ntt::ParticleTag::dead;
          }
          if (tag_p(p) == ntt::ParticleTag::dead) {
            weight_p(p) = -1.0;
          }
        });
      prtls.set_npart(66u);

      prtls.SortSpatially(grid);

      auto i1_h     = Kokkos::create_mirror_view(prtls.i1);
      auto i2_h     = Kokkos::create_mirror_view(prtls.i2);
      auto i3_h     = Kokkos::create_mirror_view(prtls.i3);
      auto tag_h    = Kokkos::create_mirror_view(prtls.tag);
      auto weight_h = Kokkos::create_mirror_view(prtls.weight);
      Kokkos::deep_copy(i1_h, prtls.i1);
      Kokkos::deep_copy(i2_h, prtls.i2);
      Kokkos::deep_copy(i3_h, prtls.i3);
      Kokkos::deep_copy(tag_h, prtls.tag);
      Kokkos::deep_copy(weight_h, prtls.weight);

      // Same invariants as the 2D block (no payloads here): alive prefix
      // sorted by non-decreasing tile index, alive count preserved. The
      // team_policy path compacts the dead away (npart() == alive count);
      // the legacy path keeps them as a weight == -1 suffix. T = 1
      // reproduces the legacy per-cell order.
#if defined(TEAM_POLICY)
      const ncells_t T = static_cast<ncells_t>(TEAM_POLICY_TILE_SIZE);
#else
      const ncells_t T = 1u;
#endif
      const auto     na   = grid.n_active();
      const ncells_t ntx2 = (na[1] + T - 1u) / T;
      const ncells_t ntx3 = (na[2] + T - 1u) / T;
      const auto     tile_of = [&](int a, int b, int c) -> ncells_t {
        return ((static_cast<ncells_t>(a) / T) * ntx2 +
                (static_cast<ncells_t>(b) / T)) *
                 ntx3 +
               (static_cast<ncells_t>(c) / T);
      };

#if defined(TEAM_POLICY)
      raise::ErrorIf(prtls.npart() != 59u,
                     "team_policy sort must compact: npart() should equal "
                     "the alive count",
                     HERE);
#else
      raise::ErrorIf(prtls.npart() != 66u,
                     "legacy sort should leave npart() unchanged",
                     HERE);
#endif
      bool     seen_dead   = false;
      bool     have_prev   = false;
      ncells_t prev_tile   = 0u;
      npart_t  n_alive_obs = 0u;
      for (auto p { 0u }; p < prtls.npart(); ++p) {
        if (tag_h(p) != ntt::ParticleTag::alive) {
          seen_dead = true;
          raise::ErrorIf(weight_h(p) != -1.0,
                         "dead particle has unexpected weight",
                         HERE);
          continue;
        }
        raise::ErrorIf(seen_dead,
                       "alive particle after a dead one (not sorted to prefix)",
                       HERE);
        const auto tile = tile_of(i1_h(p), i2_h(p), i3_h(p));
        raise::ErrorIf(have_prev && (tile < prev_tile),
                       "alive particles not sorted by tile index",
                       HERE);
        prev_tile = tile;
        have_prev = true;
        ++n_alive_obs;
      }
      raise::ErrorIf(n_alive_obs != 59u,
                     "wrong number of alive particles after sort",
                     HERE);
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    ntt::GlobalFinalize();
    return 1;
  }
  ntt::GlobalFinalize();
  return 0;
}
