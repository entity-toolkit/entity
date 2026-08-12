#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "archetypes/particle_injector.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

/**
 * @brief Constant-velocity energy distribution (satisfies EnrgDistClass)
 */
template <Dimension D>
struct ConstVelocity {
  ConstVelocity(real_t vx) : vx { vx } {}

  Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
    v[0] = vx;
    v[1] = ZERO;
    v[2] = ZERO;
  }

private:
  const real_t vx;
};

/**
 * @brief Verifies kernel::LatticeInjector_kernel for a given dimension
 *
 * Checks that the injector is a proper quiet start:
 *   1. every cell of the region receives EXACTLY prod(nppd) particles
 *   2. within each cell the sub-cell positions are exactly the lattice sites
 *      dx_i = (s_i + 1/2) / nppd_i, each visited once -- with (1) this makes the
 *      deposited density exactly uniform for any shape function
 *   3. particles land in the requested cells only (offsets are respected)
 *   4. tags/velocities are set, and the placement is deterministic (bitwise
 *      reproducible between two identical launches)
 */
template <Dimension D>
void testLatticeInjector(const std::vector<ncells_t>&  res,
                         const std::array<int, 3>&     offsets,
                         const std::array<npart_t, 3>& ncells,
                         const std::array<npart_t, 3>& nppd) {
  using M = Minkowski<D>;
  errorIf(res.size() != static_cast<std::size_t>(D), "res.size() != D");

  boundaries_t<real_t> ext;
  for (auto d { 0u }; d < static_cast<unsigned int>(D); ++d) {
    ext.push_back({ ZERO, static_cast<real_t>(res[d]) }); // dx = 1
  }
  M metric { res, ext };

  const auto ppc_cell = nppd[0] * nppd[1] * nppd[2];
  npart_t    ncells_tot { 1 };
  for (auto d { 0u }; d < static_cast<unsigned int>(D); ++d) {
    ncells_tot *= ncells[d];
  }
  const auto nparticles = ppc_cell * ncells_tot;

  const auto make_species = [&](spidx_t idx) {
    return Particles<M::Dim, M::CoordType> { idx,
                                             "test",
                                             1.0f,
                                             1.0f,
                                             nparticles,
                                             0,
                                             0,
                                             ParticlePusher::NONE,
                                             false,
                                             RadiativeDrag::NONE,
                                             EmissionType::NONE,
                                             0,
                                             0 };
  };

  const auto inject = [&](Particles<M::Dim, M::CoordType>& sp1) {
    Kokkos::parallel_for(
      "InjectLattice",
      nparticles,
      kernel::LatticeInjector_kernel<SimEngine::SRPIC, M, ConstVelocity<D>>(
        sp1,
        0u,
        metric,
        offsets[0],
        offsets[1],
        offsets[2],
        ncells[0],
        ncells[1],
        nppd[0],
        nppd[1],
        nppd[2],
        ConstVelocity<D> { ONE },
        ONE));
    Kokkos::fence();
  };

  auto species1 = make_species(1u);
  inject(species1);

  auto i1 = Kokkos::create_mirror_view(species1.i1);
  auto i2 = Kokkos::create_mirror_view(species1.i2);
  auto i3 = Kokkos::create_mirror_view(species1.i3);
  auto d1 = Kokkos::create_mirror_view(species1.dx1);
  auto d2 = Kokkos::create_mirror_view(species1.dx2);
  auto d3 = Kokkos::create_mirror_view(species1.dx3);
  auto tg = Kokkos::create_mirror_view(species1.tag);
  auto u1 = Kokkos::create_mirror_view(species1.ux1);
  Kokkos::deep_copy(i1, species1.i1);
  Kokkos::deep_copy(i2, species1.i2);
  Kokkos::deep_copy(i3, species1.i3);
  Kokkos::deep_copy(d1, species1.dx1);
  Kokkos::deep_copy(d2, species1.dx2);
  Kokkos::deep_copy(d3, species1.dx3);
  Kokkos::deep_copy(tg, species1.tag);
  Kokkos::deep_copy(u1, species1.ux1);
  // per-cell occupancy and per-cell lattice-site bookkeeping
  std::vector<npart_t> counts(ncells_tot, 0);
  std::vector<char>    sites(static_cast<std::size_t>(ncells_tot) * ppc_cell, 0);
  const auto           eps = static_cast<real_t>(1e-5);

  const auto site_of = [&](prtldx_t dx, npart_t n) -> npart_t {
    const auto s = static_cast<npart_t>(
      static_cast<real_t>(dx) * static_cast<real_t>(n));
    errorIf(s >= n, "sub-cell index out of range");
    const auto expected = (static_cast<real_t>(s) + HALF) /
                          static_cast<real_t>(n);
    errorIf(math::abs(static_cast<real_t>(dx) - expected) > eps,
            "particle is not on a lattice site: dx = " +
              std::to_string(static_cast<real_t>(dx)));
    return s;
  };

  for (npart_t p { 0 }; p < nparticles; ++p) {
    errorIf(tg(p) != ParticleTag::alive, "particle is not alive");
    errorIf(math::abs(u1(p) - ONE) > eps,
            "velocity was not set from the energy distribution");

    npart_t cell { 0 }, site { 0 };
    { // x1
      const auto c = i1(p) - offsets[0];
      errorIf(c < 0 or static_cast<npart_t>(c) >= ncells[0],
              "particle outside of the injection region in x1");
      cell = static_cast<npart_t>(c);
      site = site_of(d1(p), nppd[0]);
    }
    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      const auto c = i2(p) - offsets[1];
      errorIf(c < 0 or static_cast<npart_t>(c) >= ncells[1],
              "particle outside of the injection region in x2");
      cell += ncells[0] * static_cast<npart_t>(c);
      site += nppd[0] * site_of(d2(p), nppd[1]);
    }
    if constexpr (D == Dim::_3D) {
      const auto c = i3(p) - offsets[2];
      errorIf(c < 0 or static_cast<npart_t>(c) >= ncells[2],
              "particle outside of the injection region in x3");
      cell += ncells[0] * ncells[1] * static_cast<npart_t>(c);
      site += nppd[0] * nppd[1] * site_of(d3(p), nppd[2]);
    }
    counts[cell] += 1;
    auto& visited = sites[static_cast<std::size_t>(cell) * ppc_cell + site];
    errorIf(visited != 0, "two particles on the same lattice site");
    visited = 1;
  }
  for (npart_t c { 0 }; c < ncells_tot; ++c) {
    errorIf(counts[c] != ppc_cell,
            "cell " + std::to_string(c) + " holds " +
              std::to_string(counts[c]) + " particles instead of " +
              std::to_string(ppc_cell) + " (density shot noise!)");
  }

  // determinism: an identical launch must reproduce the placement bit for bit
  auto species1b = make_species(1u);
  inject(species1b);
  auto i1b = Kokkos::create_mirror_view(species1b.i1);
  auto d1b = Kokkos::create_mirror_view(species1b.dx1);
  Kokkos::deep_copy(i1b, species1b.i1);
  Kokkos::deep_copy(d1b, species1b.dx1);
  for (npart_t p { 0 }; p < nparticles; ++p) {
    errorIf(i1(p) != i1b(p) or d1(p) != d1b(p),
            "the lattice injector is not deterministic");
  }
}

/**
 * @brief Verifies arch::BalancedLattice: the per-direction counts must multiply
 *        back to ppc exactly for every ppc, in every dimension
 */
void testBalancedLattice(npart_t ppc_max) {
  for (dim_t dim { 1 }; dim <= 3; ++dim) {
    for (npart_t ppc { 1 }; ppc <= ppc_max; ++ppc) {
      const auto nppd = arch::BalancedLattice(ppc, dim);
      errorIf(nppd[0] * nppd[1] * nppd[2] != ppc,
              "BalancedLattice(" + std::to_string(ppc) + ", " +
                std::to_string(dim) + ") does not multiply back to ppc");
      for (auto d { 0u }; d < 3u; ++d) {
        errorIf(nppd[d] == 0, "BalancedLattice produced a zero count");
        errorIf(d >= static_cast<unsigned int>(dim) and nppd[d] != 1,
                "BalancedLattice populated an unused direction");
      }
    }
  }
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    testBalancedLattice(1024);
    // 1D: any ppc is a valid lattice
    testLatticeInjector<Dim::_1D>({ 16 }, { 0, 0, 0 }, { 16, 1, 1 }, { 8, 1, 1 });
    testLatticeInjector<Dim::_1D>({ 16 }, { 0, 0, 0 }, { 16, 1, 1 }, { 1, 1, 1 });
    testLatticeInjector<Dim::_1D>({ 16 }, { 0, 0, 0 }, { 16, 1, 1 }, { 256, 1, 1 });
    // sub-region (cell-aligned): offset respected, nothing spills over
    testLatticeInjector<Dim::_1D>({ 16 }, { 4, 0, 0 }, { 5, 1, 1 }, { 3, 1, 1 });
    // 2D/3D, including rectangular sub-cell lattices (ppc = 512 -> 16 x 32)
    testLatticeInjector<Dim::_2D>({ 8, 8 }, { 0, 0, 0 }, { 8, 8, 1 }, { 4, 4, 1 });
    testLatticeInjector<Dim::_2D>({ 8, 8 }, { 2, 3, 0 }, { 3, 4, 1 }, { 2, 5, 1 });
    testLatticeInjector<Dim::_2D>({ 4, 4 }, { 0, 0, 0 }, { 4, 4, 1 }, { 16, 32, 1 });
    testLatticeInjector<Dim::_3D>({ 4, 4, 4 }, { 0, 0, 0 }, { 4, 4, 4 }, { 3, 3, 3 });
    testLatticeInjector<Dim::_3D>({ 4, 4, 4 }, { 1, 0, 2 }, { 2, 4, 2 }, { 2, 3, 4 });
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();
  return 0;
}
