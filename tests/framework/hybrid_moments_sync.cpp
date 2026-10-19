/**
 * Ghost remap of the hybrid moments after a deposit: the additive halo sync,
 * the image-plasma fold at the x1 conductor walls and the ghost fills.
 *
 * Every storage cell of every domain (active and ghost) holds an
 * integer-valued deposit of comps 0..3 (V, N), so every sum is exact in
 * single precision whatever the order. Each deposit belongs to the global
 * active cell it maps to (periodic wrap on x2/x3, mirror about the x1 walls
 * with V_x odd). After the remap every active cell must hold the sum of the
 * deposits that map to it, and every ghost cell the (mirrored) value of the
 * active cell it copies.
 */
#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "metrics/minkowski.h"

#include "engines/hybrid/fields_bcs.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include <Kokkos_Core.hpp>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace ntt;

namespace {

  // integer-valued deposit in [1, 8] of domain `idx` at storage cell `i`
  auto deposit(unsigned int idx, const int i[3], int c) -> real_t {
    std::uint32_t h = 2166136261u;
    for (const auto v : { i[0], i[1], i[2], c, static_cast<int>(idx) }) {
      h = (h ^ static_cast<std::uint32_t>(v + 1000)) * 16777619u;
    }
    return static_cast<real_t>(1u + (h >> 9) % 8u);
  }

  // global active cell of storage cell `i` of a domain at `offset`, and the
  // sign of comp `c` under the wall mirror
  template <Dimension D>
  void image(const int                    i[3],
             const std::vector<ncells_t>& offset,
             const std::vector<ncells_t>& nglob,
             int                          g[3],
             bool&                        flip) {
    flip = false;
    for (auto d { 0u }; d < 3u; ++d) {
      g[d] = 0;
    }
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      const int n = static_cast<int>(nglob[d]);
      int x = i[d] - static_cast<int>(N_GHOSTS) + static_cast<int>(offset[d]);
      if (d == 0u) {
        if (x < 0) {
          x    = -1 - x;
          flip = true;
        } else if (x >= n) {
          x    = 2 * n - 1 - x;
          flip = true;
        }
      } else {
        x = ((x % n) + n) % n;
      }
      g[d] = x;
    }
  }

  template <Dimension D>
  auto run(const std::vector<ncells_t>& res,
           const std::vector<int>&      decomposition,
           const std::string&           label) -> bool {
    using M  = metric::Minkowski<D>;
    int rank = 0, nranks = 1;
#if defined(MPI_ENABLED)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
    boundaries_t<real_t> extent;
    boundaries_t<FldsBC> fldsbc;
    boundaries_t<PrtlBC> prtlbc;
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      extent.push_back({ ZERO, static_cast<real_t>(res[d]) });
      if (d == 0u) {
        fldsbc.push_back({ FldsBC::CONDUCTOR, FldsBC::CONDUCTOR });
        prtlbc.push_back({ PrtlBC::REFLECT, PrtlBC::REFLECT });
      } else {
        fldsbc.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
        prtlbc.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
      }
    }
    Metadomain<SimEngine::HYBRID, M> metadomain { static_cast<unsigned int>(nranks),
                                                  decomposition,
                                                  res,
                                                  extent,
                                                  fldsbc,
                                                  prtlbc,
                                                  {},
                                                  {} };
    const auto ldidx = metadomain.l_subdomain_indices()[0];
    auto&      dom   = *metadomain.subdomain_ptr(ldidx);

    // storage extents of a domain (1 for absent axes)
    const auto extents_of = [](const Domain<SimEngine::HYBRID, M>& d) {
      std::vector<int> e { 1, 1, 1 };
      for (auto r { 0u }; r < static_cast<unsigned>(D); ++r) {
        e[r] = static_cast<int>(d.mesh.n_active()[r] + 2 * N_GHOSTS);
      }
      return e;
    };

    // deposits of the local domain
    auto       aux_h = Kokkos::create_mirror_view(dom.fields.aux);
    const auto el    = extents_of(dom);
    for (int i1 = 0; i1 < el[0]; ++i1) {
      for (int i2 = 0; i2 < el[1]; ++i2) {
        for (int i3 = 0; i3 < el[2]; ++i3) {
          const int i[3] { i1, i2, i3 };
          for (int c = 0; c < 6; ++c) {
            const real_t v = (c < 4) ? deposit(ldidx, i, c) : ZERO;
            if constexpr (D == Dim::_1D) {
              aux_h(i1, c) = v;
            } else if constexpr (D == Dim::_2D) {
              aux_h(i1, i2, c) = v;
            } else {
              aux_h(i1, i2, i3, c) = v;
            }
          }
        }
      }
    }
    Kokkos::deep_copy(dom.fields.aux, aux_h);

    // the remap after a deposit, as in the hybrid engine
    metadomain.SynchronizeFields(dom, Comm::AUX);
    hybrid::MomentsWallBC(dom, metadomain.mesh(), true);
    metadomain.CommunicateFields(dom, Comm::AUX);
    hybrid::MomentsWallBC(dom, metadomain.mesh(), false);

    Kokkos::deep_copy(aux_h, dom.fields.aux);

    // expected sums over the deposits of all domains, per global active cell
    const auto       nglob = metadomain.mesh().n_active();
    std::vector<int> ng { 1, 1, 1 };
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      ng[d] = static_cast<int>(nglob[d]);
    }
    const auto gidx = [&](const int g[3], int c) -> std::size_t {
      return static_cast<std::size_t>(
        ((g[2] * ng[1] + g[1]) * ng[0] + g[0]) * 4 + c);
    };
    std::vector<double> total(static_cast<std::size_t>(ng[0] * ng[1] * ng[2] * 4),
                              0.0);
    double total_n = 0.0;
    for (unsigned int idx = 0; idx < metadomain.ndomains(); ++idx) {
      const auto& sd  = metadomain.subdomain(idx);
      const auto  e   = extents_of(sd);
      const auto  off = sd.offset_ncells();
      for (int i1 = 0; i1 < e[0]; ++i1) {
        for (int i2 = 0; i2 < e[1]; ++i2) {
          for (int i3 = 0; i3 < e[2]; ++i3) {
            const int i[3] { i1, i2, i3 };
            int       g[3];
            bool      flip;
            image<D>(i, off, nglob, g, flip);
            for (int c = 0; c < 4; ++c) {
              const double v     = static_cast<double>(deposit(idx, i, c));
              total[gidx(g, c)] += (flip and c == 0) ? -v : v;
            }
            total_n += static_cast<double>(deposit(idx, i, 3));
          }
        }
      }
    }

    // compare every storage cell of the local domain
    const auto off  = dom.offset_ncells();
    long   nbad[4]  = { 0, 0, 0, 0 }; // active, wall ghost, halo ghost, corner
    long   ncell[4] = { 0, 0, 0, 0 };
    double maxdiff  = 0.0;
    double sum_n    = 0.0;
    for (int i1 = 0; i1 < el[0]; ++i1) {
      for (int i2 = 0; i2 < el[1]; ++i2) {
        for (int i3 = 0; i3 < el[2]; ++i3) {
          const int i[3] { i1, i2, i3 };
          bool      ghost[3] { false, false, false };
          for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
            ghost[d] = (i[d] < static_cast<int>(N_GHOSTS)) or
                       (i[d] >= el[d] - static_cast<int>(N_GHOSTS));
          }
          const bool wall_side = ghost[0] and
                                 ((i1 < static_cast<int>(N_GHOSTS) and off[0] == 0u) or
                                  (i1 >= el[0] - static_cast<int>(N_GHOSTS) and
                                   off[0] + dom.mesh.n_active()[0] == nglob[0]));
          const bool halo = (ghost[0] and not wall_side) or ghost[1] or ghost[2];
          const int cat = wall_side ? (halo ? 3 : 1) : (halo ? 2 : 0);
          int       g[3];
          bool      flip;
          image<D>(i, off, nglob, g, flip);
          bool bad = false;
          for (int c = 0; c < 4; ++c) {
            double expect = total[gidx(g, c)];
            if (flip and c == 0) {
              expect = -expect;
            }
            real_t got;
            if constexpr (D == Dim::_1D) {
              got = aux_h(i1, c);
            } else if constexpr (D == Dim::_2D) {
              got = aux_h(i1, i2, c);
            } else {
              got = aux_h(i1, i2, i3, c);
            }
            const double diff = static_cast<double>(got) - expect;
            if (diff != 0.0) {
              bad     = true;
              maxdiff = std::max(maxdiff, std::abs(diff));
            }
            if (c == 3 and cat == 0) {
              sum_n += static_cast<double>(got);
            }
          }
          ++ncell[cat];
          if (bad) {
            ++nbad[cat];
          }
        }
      }
    }
#if defined(MPI_ENABLED)
    MPI_Allreduce(MPI_IN_PLACE, nbad, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ncell, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum_n, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    const bool ok = (nbad[0] + nbad[1] + nbad[2] + nbad[3]) == 0;
    if (rank == 0) {
      std::cout << label << " (" << nranks << " ranks): mismatched cells "
                << "active " << nbad[0] << "/" << ncell[0] << ", wall ghost "
                << nbad[1] << "/" << ncell[1] << ", halo ghost " << nbad[2]
                << "/" << ncell[2] << ", corner " << nbad[3] << "/" << ncell[3]
                << ", max |diff| " << maxdiff << "; sum N active " << sum_n
                << " of deposited " << total_n << (ok ? "  OK" : "  FAIL")
                << std::endl;
    }
    return ok;
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  // errors raised inside the framework are reported through plog
  static plog::ConsoleAppender<plog::TxtFormatter> console;
  plog::init(plog::warning, &console);
  bool ok = true;
  try {
    ok &= run<Dim::_1D>({ 24 }, { -1 }, "1D x1 split");
    ok &= run<Dim::_2D>({ 16, 24 }, { 1, -1 }, "2D x2 split");
    ok &= run<Dim::_2D>({ 24, 16 }, { -1, 1 }, "2D x1 split");
    ok &= run<Dim::_3D>({ 12, 20, 20 }, { 1, -1, -1 }, "3D x2/x3 split");
    ok &= run<Dim::_3D>({ 24, 12, 12 }, { -1, 1, 1 }, "3D x1 split");
    ok &= run<Dim::_2D>({ 24, 20 }, { -1, -1 }, "2D x1/x2 split");
    ok &= run<Dim::_3D>({ 24, 20, 20 }, { -1, -1, -1 }, "3D x1/x2/x3 split");
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return ok ? 0 : 1;
}
