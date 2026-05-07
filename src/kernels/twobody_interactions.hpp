#ifndef KERNELS_TWOBODY_INTERACTIONS_HPP
#define KERNELS_TWOBODY_INTERACTIONS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/policies.h"
#include "utils/error.h"
#include "utils/sorting.h"

#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <cstdint>
#include <vector>

namespace kernel::mink {
  using namespace ntt;

  namespace {
    struct CollisionSpecies {
      const spidx_t sp;
      const npart_t npart;
      ncells_t      num_tiles { 0u };

      array_t<ncells_t*> tileidx;
      array_t<npart_t*>  num_ppt;

      CollisionSpecies(spidx_t                   sp,
                       npart_t                   npart,
                       const array_t<ncells_t*>& tileidx,
                       const array_t<npart_t*>&  num_ppt)
        : sp { sp }
        , npart { npart }
        , tileidx { tileidx }
        , num_ppt { num_ppt } {}
    };

    template <Dimension D>
    struct CollisionGroup {
      std::vector<CollisionSpecies> group;

      array_t<uint64_t*> combined_idx;
      array_t<ncells_t*> combined_tileidx;
      array_t<npart_t*>  combined_num_ppt;
      array_t<npart_t*>  tile_offsets;

      ncells_t num_tiles { 0u };

      CollisionGroup(const std::vector<const Particles<D, Coord::Cartesian>*>& particles,
                     const std::vector<ncells_t>& ncells,
                     ncells_t                     tile_size,
                     random_number_pool_t&        random_pool) {
        for (const auto* species : particles) {
          const auto             npart_s = species->npart();
          array_t<ncells_t*>     tileidx { "tile_idx", npart_s };
          array_t<npart_t*>      num_ppt_s {};
          group.emplace_back(species->sp, npart_s, tileidx, num_ppt_s);
          auto tile_indexing_kernel = sort::PositionToTileIndex<D, true>(
            species->i1,
            species->i2,
            species->i3,
            species->tag,
            group.back().tileidx,
            ncells,
            tile_size,
            group.back().num_ppt);
          Kokkos::parallel_for("TileIndexing",
                               species->npart(),
                               tile_indexing_kernel);
          group.back().num_tiles = tile_indexing_kernel.total_tiles;

          if (num_tiles == 0u) {
            num_tiles = group.back().num_tiles;
          } else if (num_tiles != group.back().num_tiles) {
            raise::Error("unequal num_tiles across species in group", HERE);
          }
          raise::ErrorIf(group.back().tileidx.extent(0) != species->npart(),
                         "tileidx must have the same extent as npart for all "
                         "species in group",
                         HERE);
        }

        npart_t tot_npart = 0u;
        for (const auto& species : group) {
          tot_npart += species.npart;
        }

        combined_idx     = array_t<uint64_t*> { "combined_idx", tot_npart };
        combined_tileidx = array_t<ncells_t*> { "combined_tileidx", tot_npart };
        combined_num_ppt = array_t<npart_t*> { "combined_num_ppt", num_tiles };
        tile_offsets     = array_t<npart_t*> { "tile_offsets", num_tiles };

        {
          // combine particle indices in the group & compute total number in each tile
          npart_t offset = 0u;
          for (const auto& species : group) {
            Kokkos::parallel_for(
              "CombineInGroup",
              species.npart,
              ClassLambda(const npart_t p) {
                // pack species idx into top 8 bits + prtl index into the remaining 56 bits
                combined_idx(offset + p) = (static_cast<uint64_t>(species.sp)
                                            << 56) |
                                           static_cast<uint64_t>(p);
                combined_tileidx(offset + p) = species.tileidx(p);
              });
            offset += species.npart;
            Kokkos::parallel_for(
              "CombineNumPpt",
              species.num_tiles,
              ClassLambda(const ncells_t t) {
                combined_num_ppt(t) += species.num_ppt(t);
              });
            Kokkos::fence();
          }
        }
        {
          // randomly shuffle particles within each tile and sort by tiles
          array_t<uint64_t*> shuffle_key { "shuffle_key", tot_npart };
          Kokkos::parallel_for(
            "PackRandom",
            tot_npart,
            ClassLambda(const npart_t p) {
              auto       gen = random_pool.get_state();
              const auto rnd = static_cast<uint64_t>(gen.urand());
              random_pool.free_state(gen);
              const auto tile_idx = static_cast<uint64_t>(combined_tileidx(p));
              // packing top 32 bits with tile index, and the rest -- random
              shuffle_key(p)      = (tile_idx << 32) | rnd;
            });
          Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace {},
                                            shuffle_key,
                                            combined_idx);
        }
        {
          // compute index offsets for each tile
          Kokkos::parallel_scan(
            "TileOffsets",
            num_tiles,
            ClassLambda(cellidx_t t, npart_t & acc, const bool final) {
              if (final) {
                tile_offsets(t) = acc;
              }
              acc += combined_num_ppt(t);
            });
        }
      }
    };
  } // namespace

  template <Dimension D, TwoBodyInteractionPolicyClass I>
  void TwoBodyInteraction(
    const std::vector<const Particles<D, Coord::Cartesian>*>& species1,
    const std::vector<const Particles<D, Coord::Cartesian>*>& species2,
    const std::vector<ncells_t>&                              ncells,
    ncells_t                                                  tile_size,
    random_number_pool_t&                                     random_pool,
    const I&                                                  interaction_policy) {
    const auto group1 = CollisionGroup<D>(species1, ncells, tile_size, random_pool);
    const auto group2 = CollisionGroup<D>(species2, ncells, tile_size, random_pool);
    raise::ErrorIf(group1.num_tiles != group2.num_tiles,
                   "number of tiles differ in group1 vs group2",
                   HERE);
    const auto num_tiles = group1.num_tiles;

    const auto& combined_idx1     = group1.combined_idx;
    const auto& combined_idx2     = group2.combined_idx;
    const auto& combined_num_ppt1 = group1.combined_num_ppt;
    const auto& combined_num_ppt2 = group2.combined_num_ppt;
    const auto& tile_offsets1     = group1.tile_offsets;
    const auto& tile_offsets2     = group2.tile_offsets;

    Kokkos::parallel_for(
      "EmitPairs",
      Kokkos::TeamPolicy<>(num_tiles, Kokkos::AUTO),
      Lambda(const Kokkos::TeamPolicy<>::member_type& team) {
        const auto t  = team.league_rank();
        const auto k  = math::min(combined_num_ppt1(t), combined_num_ppt2(t));
        const auto o1 = tile_offsets1(t);
        const auto o2 = tile_offsets2(t);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k), [&](prtlidx_t i) {
          // unpack the higher 8 bits
          const auto sp1 = static_cast<spidx_t>(combined_idx1(o1 + i) >> 56);
          const auto sp2 = static_cast<spidx_t>(combined_idx2(o2 + i) >> 56);

          // unpack the lower 56 bits
          const auto p1 = static_cast<npart_t>(combined_idx1(o1 + i) &
                                               ((1ull << 56) - 1));
          const auto p2 = static_cast<npart_t>(combined_idx2(o2 + i) &
                                               ((1ull << 56) - 1));
          interaction_policy(sp1, p1, sp2, p2);
        });
      });
  }

} // namespace kernel::mink

#endif // KERNELS_TWOBODY_INTERACTIONS_HPP
