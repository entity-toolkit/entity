#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"
#include "utils/sorting.h"

#include <Kokkos_Core.hpp>

#include <iostream>

template <Dimension D>
Inline void CheckValue(prtlidx_t,
                       const array_t<int*>&,
                       const array_t<int*>&,
                       const array_t<int*>&,
                       const array_t<short*>&,
                       const array_t<ncells_t*>&,
                       ncells_t,
                       ncells_t,
                       ncells_t,
                       ncells_t,
                       ncells_t);

template <>
Inline void CheckValue<Dim::_1D>(prtlidx_t            p,
                                 const array_t<int*>& i1,
                                 const array_t<int*>&,
                                 const array_t<int*>&,
                                 const array_t<short*>&    tag,
                                 const array_t<ncells_t*>& tile_indices,
                                 ncells_t,
                                 ncells_t,
                                 ncells_t,
                                 ncells_t ntiles,
                                 ncells_t ts) {
  if (tag(p) != ntt::ParticleTag::alive) {
    if (tile_indices(p) != ntiles + 1u) {
      raise::KernelError(HERE, "Dead particle assigned to wrong tile index");
    }
    return;
  }
  const auto ti = static_cast<npart_t>(i1(p) / ts);
  if (tile_indices(p) != ti) {
    raise::KernelError(HERE, "Alive particle assigned to wrong tile index");
  }
}

template <>
Inline void CheckValue<Dim::_2D>(prtlidx_t            p,
                                 const array_t<int*>& i1,
                                 const array_t<int*>& i2,
                                 const array_t<int*>&,
                                 const array_t<short*>&    tag,
                                 const array_t<ncells_t*>& tile_indices,
                                 ncells_t,
                                 ncells_t nt2,
                                 ncells_t,
                                 ncells_t ntiles,
                                 ncells_t ts) {
  if (tag(p) != ntt::ParticleTag::alive) {
    if (tile_indices(p) != ntiles + 1u) {
      raise::KernelError(HERE, "Dead particle assigned to wrong tile index");
    }
    return;
  }
  const auto ti = static_cast<npart_t>(i1(p) / ts) * nt2 +
                  static_cast<npart_t>(i2(p) / ts);
  if (tile_indices(p) != ti) {
    raise::KernelError(HERE, "Alive particle assigned to wrong tile index");
  }
}

template <>
Inline void CheckValue<Dim::_3D>(prtlidx_t                 p,
                                 const array_t<int*>&      i1,
                                 const array_t<int*>&      i2,
                                 const array_t<int*>&      i3,
                                 const array_t<short*>&    tag,
                                 const array_t<ncells_t*>& tile_indices,
                                 ncells_t,
                                 ncells_t nt2,
                                 ncells_t nt3,
                                 ncells_t ntiles,
                                 ncells_t ts) {
  if (tag(p) != ntt::ParticleTag::alive) {
    if (tile_indices(p) != ntiles + 1u) {
      raise::KernelError(HERE, "Dead particle assigned to wrong tile index");
    }
    return;
  }
  const auto ti = (static_cast<npart_t>(i1(p) / ts) * nt2 +
                   static_cast<npart_t>(i2(p) / ts)) *
                    nt3 +
                  static_cast<npart_t>(i3(p) / ts);
  if (tile_indices(p) != ti) {
    raise::KernelError(HERE, "Alive particle assigned to wrong tile index");
  }
}

template <Dimension D>
void test_tiling(const array_t<int*>&         i1,
                 const array_t<int*>&         i2,
                 const array_t<int*>&         i3,
                 const array_t<short*>&       tag,
                 array_t<ncells_t*>&          tile_indices,
                 npart_t                      npart,
                 const std::vector<ncells_t>& ncells,
                 npart_t                      ndead) {
  for (auto ts { 1u }; ts <= 11u; ++ts) {
    ncells_t nt1 = 1u, nt2 = 1u, nt3 = 1u;

    nt1 = static_cast<ncells_t>(
      math::ceil(static_cast<double>(ncells[0]) / static_cast<double>(ts)));
    if constexpr ((D == Dim::_2D) or (D == Dim::_3D)) {
      nt2 = static_cast<ncells_t>(
        math::ceil(static_cast<double>(ncells[1]) / static_cast<double>(ts)));
    }
    if constexpr (D == Dim::_3D) {
      nt3 = static_cast<ncells_t>(
        math::ceil(static_cast<double>(ncells[2]) / static_cast<double>(ts)));
    }

    const auto ntiles = nt1 * nt2 * nt3;

    array_t<npart_t*> num_ppt { "num_ppt", ntiles };
    Kokkos::parallel_for(
      "Tiling",
      npart,
      sort::PositionToTileIndex<D, true> { i1, i2, i3, tag, tile_indices, ncells, ts, num_ppt });
    Kokkos::parallel_for(
      "Checking",
      npart,
      Lambda(prtlidx_t p) {
        CheckValue<D>(p, i1, i2, i3, tag, tile_indices, nt1, nt2, nt3, ntiles, ts);
      });

    npart_t tot_alive = 0u;
    Kokkos::parallel_reduce(
      "CountAliveInTiles",
      ntiles,
      Lambda(prtlidx_t t, npart_t & count) { count += num_ppt(t); },
      tot_alive);
    raise::ErrorIf(tot_alive != npart - ndead,
                   "Error in counting particles per tile",
                   HERE);
  }
}

auto main(int argc, char* argv[]) -> int {
  using namespace ntt;
  Kokkos::initialize(argc, argv);
  try {
    {
      const ncells_t nx1 = 123u, nx2 = 325u, nx3 = 111u;
      const npart_t  npart = 1234u;

      array_t<int*>   i1 { "i1", npart };
      array_t<int*>   i2 { "i2", npart };
      array_t<int*>   i3 { "i3", npart };
      array_t<short*> tag { "tag", npart };

      array_t<ncells_t*> tile_indices { "tile_indices", npart };

      random_number_pool_t random_pool { constant::RandomSeed };

      array_t<npart_t> ndead { "ndead" };

      // test # 1
      Kokkos::parallel_for(
        "Initialize",
        npart,
        Lambda(prtlidx_t p) {
          auto gen = random_pool.get_state();
          i1(p)    = static_cast<int>(gen.urand(0u, nx1));
          i2(p)    = static_cast<int>(gen.urand(0u, nx2));
          i3(p)    = static_cast<int>(gen.urand(0u, nx3));
          tag(p) = (gen.drand() > 0.01) ? ParticleTag::alive : ParticleTag::dead;
          random_pool.free_state(gen);
          if (tag(p) == ParticleTag::dead) {
            Kokkos::atomic_add(&ndead(), 1u);
          }
        });

      auto ndead_h = Kokkos::create_mirror_view(ndead);
      Kokkos::deep_copy(ndead_h, ndead);

      test_tiling<Dim::_1D>(i1, i2, i3, tag, tile_indices, npart, { nx1 }, ndead_h());
      test_tiling<Dim::_2D>(i1, i2, i3, tag, tile_indices, npart, { nx1, nx2 }, ndead_h());
      test_tiling<Dim::_3D>(i1,
                            i2,
                            i3,
                            tag,
                            tile_indices,
                            npart,
                            { nx1, nx2, nx3 },
                            ndead_h());
    }

    // test # 2
    {
      const ncells_t nx1 = 23u, nx2 = 31u;
      const npart_t  npart = 4236u;

      array_t<int*>   i1 { "i1", npart };
      array_t<int*>   i2 { "i2", npart };
      array_t<int*>   i3 { "i2" };
      array_t<short*> tag { "tag", npart };

      for (auto tile_size { 6u }; tile_size <= 6u; ++tile_size) {
        array_t<ncells_t*> tile_indices { "tile_indices", npart };
        const auto         nt1 = static_cast<ncells_t>(
          math::ceil(static_cast<double>(nx1) / static_cast<double>(tile_size)));
        const auto nt2 = static_cast<ncells_t>(
          math::ceil(static_cast<double>(nx2) / static_cast<double>(tile_size)));
        const auto ntiles = nt1 * nt2;
        const auto ncells = nx1 * nx2;
        Kokkos::parallel_for(
          "Initialize",
          npart,
          Lambda(prtlidx_t p) {
            const auto cell_idx = p % ncells;
            i1(p)  = static_cast<int>(cell_idx) % static_cast<int>(nx1);
            i2(p)  = static_cast<int>(cell_idx) / static_cast<int>(nx1);
            tag(p) = ParticleTag::alive;
          });
        Kokkos::parallel_for("Tiling",
                             npart,
                             sort::PositionToTileIndex<Dim::_2D, false> {
                               i1,
                               i2,
                               i3,
                               tag,
                               tile_indices,
                               std::vector<ncells_t> { nx1, nx2 },
                               tile_size
        });
        Kokkos::parallel_for(
          "CheckOutOfBounds",
          npart,
          Lambda(prtlidx_t p) {
            const auto tile_idx = tile_indices(p);
            const auto t1       = tile_idx / nt2;
            const auto t2       = tile_idx % nt2;
            const auto i1_min   = t1 * tile_size;
            const auto i1_max   = math::min(i1_min + tile_size, nx1);
            const auto i2_min   = t2 * tile_size;
            const auto i2_max   = math::min(i2_min + tile_size, nx2);
            if (i1(p) < i1_min or i1(p) >= i1_max or i2(p) < i2_min or
                i2(p) >= i2_max) {
              raise::KernelError(HERE, "Particle assigned to wrong tile index");
            }
          });

        // sort
        using sorter_op_t = Kokkos::BinOp1D<decltype(tile_indices)>;
        using sorter_t = Kokkos::BinSort<decltype(tile_indices), sorter_op_t>;
        auto bin_op = sorter_op_t { static_cast<int>(ntiles + 1u), 0u, ntiles + 1u };
        auto sorter = sorter_t { tile_indices, bin_op, false };
        sorter.create_permute_vector();
        sorter.sort(i1);
        sorter.sort(i2);
        sorter.sort(tile_indices);

        auto i1_h = Kokkos::create_mirror_view(i1);
        auto i2_h = Kokkos::create_mirror_view(i2);
        auto ti_h = Kokkos::create_mirror_view(tile_indices);
        Kokkos::deep_copy(i1_h, i1);
        Kokkos::deep_copy(i2_h, i2);
        Kokkos::deep_copy(ti_h, tile_indices);

        ncells_t current_tile = ti_h(0);
        for (auto p { 0u }; p < npart; ++p) {
          if (ti_h(p) != current_tile) {
            if (ti_h(p) < current_tile) {
              raise::Error("Tile indices not sorted correctly", HERE);
            }
            current_tile = ti_h(p);
          }
          const auto t1     = current_tile / nt2;
          const auto t2     = current_tile % nt2;
          const auto i1_min = t1 * tile_size;
          const auto i1_max = math::min(i1_min + tile_size, nx1);
          const auto i2_min = t2 * tile_size;
          const auto i2_max = math::min(i2_min + tile_size, nx2);
          if (i1_h(p) < i1_min or i1_h(p) >= i1_max or i2_h(p) < i2_min or
              i2_h(p) >= i2_max) {
            raise::Error("Particle assigned to wrong tile index after sorting",
                         HERE);
          }
        }

        // write to csv file
        // std::fstream fout { "tiling_test_output.csv", std::fstream::out };
        // fout << "i1,i2,tile_index\n";
        // auto i1_h = Kokkos::create_mirror_view(i1);
        // auto i2_h = Kokkos::create_mirror_view(i2);
        // auto ti_h = Kokkos::create_mirror_view(tile_indices);
        // Kokkos::deep_copy(i1_h, i1);
        // Kokkos::deep_copy(i2_h, i2);
        // Kokkos::deep_copy(ti_h, tile_indices);
        // for (auto p { 0u }; p < npart; ++p) {
        //   fout << i1_h(p) << "," << i2_h(p) << "," << ti_h(p) << "\n";
        // }
        // fout.close();
      }
    }

  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
