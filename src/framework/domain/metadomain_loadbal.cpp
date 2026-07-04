#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "framework/containers/fields.h"
#include "framework/domain/metadomain.h"
#include "framework/specialization_registry.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"
  #include "arch/mpi_tags.h"

  #include <mpi.h>
#endif

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace ntt {

#if defined(MPI_ENABLED)
  // Copy old field into new (already-zero) field, applying a shift in the
  // active-cell offset along each dimension.
  // For new (with-ghost) index i, old (with-ghost) index = i + delta[d].
  // Cells whose old index is out of bounds are left at 0 and will be filled
  // by CommunicateFields() afterwards.
  template <Dimension D, unsigned short NC>
  void CopyShifted(const ndfield_mirror_t<D, NC>& src_h,
                   ndfield_t<D, NC>&              dst_dev,
                   const std::vector<int>&        delta) {
    auto dst_h = Kokkos::create_mirror_view(dst_dev);
    Kokkos::deep_copy(dst_h, ZERO);

    if constexpr (D == Dim::_1D) {
      const int new_n1 = static_cast<int>(dst_h.extent(0));
      const int old_n1 = static_cast<int>(src_h.extent(0));
      for (int i1 = 0; i1 < new_n1; ++i1) {
        const int oi1 = i1 + delta[0];
        if (oi1 < 0 or oi1 >= old_n1) {
          continue;
        }
        for (auto c { 0u }; c < NC; ++c) {
          dst_h(i1, c) = src_h(oi1, c);
        }
      }
    } else if constexpr (D == Dim::_2D) {
      const int new_n1 = static_cast<int>(dst_h.extent(0));
      const int new_n2 = static_cast<int>(dst_h.extent(1));
      const int old_n1 = static_cast<int>(src_h.extent(0));
      const int old_n2 = static_cast<int>(src_h.extent(1));
      for (int i1 = 0; i1 < new_n1; ++i1) {
        const int oi1 = i1 + delta[0];
        if (oi1 < 0 or oi1 >= old_n1) {
          continue;
        }
        for (int i2 = 0; i2 < new_n2; ++i2) {
          const int oi2 = i2 + delta[1];
          if (oi2 < 0 or oi2 >= old_n2) {
            continue;
          }
          for (auto c { 0u }; c < NC; ++c) {
            dst_h(i1, i2, c) = src_h(oi1, oi2, c);
          }
        }
      }
    } else if constexpr (D == Dim::_3D) {
      const int new_n1 = static_cast<int>(dst_h.extent(0));
      const int new_n2 = static_cast<int>(dst_h.extent(1));
      const int new_n3 = static_cast<int>(dst_h.extent(2));
      const int old_n1 = static_cast<int>(src_h.extent(0));
      const int old_n2 = static_cast<int>(src_h.extent(1));
      const int old_n3 = static_cast<int>(src_h.extent(2));
      for (int i1 = 0; i1 < new_n1; ++i1) {
        const int oi1 = i1 + delta[0];
        if (oi1 < 0 or oi1 >= old_n1) {
          continue;
        }
        for (int i2 = 0; i2 < new_n2; ++i2) {
          const int oi2 = i2 + delta[1];
          if (oi2 < 0 or oi2 >= old_n2) {
            continue;
          }
          for (int i3 = 0; i3 < new_n3; ++i3) {
            const int oi3 = i3 + delta[2];
            if (oi3 < 0 or oi3 >= old_n3) {
              continue;
            }
            for (auto c { 0u }; c < NC; ++c) {
              dst_h(i1, i2, i3, c) = src_h(oi1, oi2, oi3, c);
            }
          }
        }
      }
    }
    Kokkos::deep_copy(dst_dev, dst_h);
  }
#endif // MPI_ENABLED

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::Rebalance(unsigned int dim_mask,
                                   real_t       tolerance,
                                   ncells_t     max_shift_cells)
    requires(MetricClass<M>)
  {
#if !defined(MPI_ENABLED)
    (void)dim_mask;
    (void)tolerance;
    (void)max_shift_cells;
    return;
#else
    raise::ErrorIf(l_subdomain_indices().size() != 1,
                   "Rebalance assumes one local subdomain per rank",
                   HERE);
    // The theta dimension (idx 1) is bounded by the polar axis for any
    // non-Cartesian metric: moving an interior boundary in theta is fine
    // in principle, but the safety net here forbids it pending validation.
    if constexpr (M::CoordType != ntt::Coord::Cartesian) {
      raise::ErrorIf((dim_mask & (1u << 1)) != 0u,
                     "Rebalance along the polar axis is not supported",
                     HERE);
    }
    // strip-width is constrained by N_GHOSTS so that any new active cells
    // are already present in the rank's old ghost zone (after the most
    // recent ghost-cell exchange).
    if (max_shift_cells > N_GHOSTS) {
      max_shift_cells = static_cast<ncells_t>(N_GHOSTS);
    }
    if (max_shift_cells == 0 or dim_mask == 0) {
      return;
    }

    const auto local_idx = l_subdomain_indices()[0];

    /* --- 1. Allgather active particle counts per domain ------------------ */
    npart_t local_npart { 0 };
    for (const auto& sp : g_subdomains[local_idx].species) {
      local_npart += sp.npart();
    }
    std::vector<npart_t> npart_per_dom(g_ndomains, 0);
    MPI_Allgather(&local_npart,
                  1,
                  mpi::get_type<npart_t>(),
                  npart_per_dom.data(),
                  1,
                  mpi::get_type<npart_t>(),
                  MPI_COMM_WORLD);

    /* --- 2. Project ncells / load onto each balanced dim ----------------- */
    std::vector<std::vector<ncells_t>> ncells_per_pos(M::Dim);
    std::vector<std::vector<double>>   load_per_pos(M::Dim);
    for (auto d { 0u }; d < M::Dim; ++d) {
      ncells_per_pos[d].assign(g_ndomains_per_dim[d], 0);
      load_per_pos[d].assign(g_ndomains_per_dim[d], 0.0);
    }
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      const auto& off    = g_domain_offsets[idx];
      const auto  ncells = g_subdomains[idx].mesh.n_active();
      for (auto d { 0u }; d < M::Dim; ++d) {
        ncells_per_pos[d][off[d]] = ncells[d];
        load_per_pos[d][off[d]]  += static_cast<double>(npart_per_dom[idx]);
      }
    }

    /* --- 3. Diffusion-style boundary shifts per balanced dim ------------- */
    std::vector<std::vector<ncells_t>> new_ncells_per_pos = ncells_per_pos;
    const auto MIN_NCELLS = static_cast<ncells_t>(2 * N_GHOSTS + 4);
    bool any_shift = false;

    for (auto d { 0u }; d < M::Dim; ++d) {
      if ((dim_mask & (1u << d)) == 0u) {
        continue;
      }
      const auto N = g_ndomains_per_dim[d];
      if (N < 2) {
        continue;
      }

      double total_load = 0.0;
      double max_load = 0.0;
      double min_load = std::numeric_limits<double>::infinity();
      for (auto p { 0u }; p < N; ++p) {
        total_load += load_per_pos[d][p];
        max_load = std::max(max_load, load_per_pos[d][p]);
        min_load = std::min(min_load, load_per_pos[d][p]);
      }
      if (total_load <= 0.0) {
        continue;
      }
      const auto mean = total_load / static_cast<double>(N);
      if (((max_load - min_load) / mean) <
          static_cast<double>(tolerance)) {
        continue;
      }

      // bnd_shift[k] is the number of cells transferred from position k-1 to
      // position k by moving the (interior) boundary k. bnd_shift[0] and
      // bnd_shift[N] are exterior boundaries that are pinned at 0.
      std::vector<int> bnd_shift(N + 1, 0);
      const int        cap = static_cast<int>(max_shift_cells);
      for (auto k { 1u }; k < N; ++k) {
        const auto l_load    = load_per_pos[d][k - 1];
        const auto r_load    = load_per_pos[d][k];
        const auto l_density = (ncells_per_pos[d][k - 1] > 0)
                                 ? l_load / static_cast<double>(
                                              ncells_per_pos[d][k - 1])
                                 : 0.0;
        const auto r_density = (ncells_per_pos[d][k] > 0)
                                 ? r_load / static_cast<double>(
                                              ncells_per_pos[d][k])
                                 : 0.0;
        const auto avg_density = std::max(0.5 * (l_density + r_density), 1.0);
        // Move boundary towards the lighter side. Halve the gradient so that
        // a single sweep does roughly one diffusion step.
        int shift = static_cast<int>(
          std::round(0.5 * (l_load - r_load) / avg_density));
        shift = std::clamp(shift, -cap, cap);
        // Don't shrink either side below MIN_NCELLS.
        const int max_pos = static_cast<int>(ncells_per_pos[d][k - 1]) -
                            static_cast<int>(MIN_NCELLS);
        const int max_neg = static_cast<int>(ncells_per_pos[d][k]) -
                            static_cast<int>(MIN_NCELLS);
        shift = std::clamp(shift,
                           -std::max(max_neg, 0),
                           std::max(max_pos, 0));
        bnd_shift[k] = shift;
        if (shift != 0) {
          any_shift = true;
        }
      }
      // new_ncells[p] = ncells[p] + bnd_shift[p] - bnd_shift[p+1]
      for (auto p { 0u }; p < N; ++p) {
        new_ncells_per_pos[d][p] = static_cast<ncells_t>(
          static_cast<int>(ncells_per_pos[d][p]) + bnd_shift[p] -
          bnd_shift[p + 1]);
      }
    }

    if (not any_shift) {
      return;
    }

    /* --- 4. Per-position prefix sums (offset and physical extent) -------- */
    // Face positions are queried from g_mesh.metric in the global code-coordinate
    // system, so curvilinear stretches (log-r, eta-stretching) are honored.
    std::vector<std::vector<ncells_t>> new_offset_per_pos(M::Dim);
    std::vector<std::vector<std::pair<real_t, real_t>>> extent_per_pos(M::Dim);
    auto face_phys = [this](unsigned int d, real_t x_code) -> real_t {
      if (d == 0u) {
        return g_mesh.metric.template convert<1, Crd::Cd, Crd::Ph>(x_code);
      }
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        if (d == 1u) {
          return g_mesh.metric.template convert<2, Crd::Cd, Crd::Ph>(x_code);
        }
      }
      if constexpr (M::Dim == Dim::_3D) {
        if (d == 2u) {
          return g_mesh.metric.template convert<3, Crd::Cd, Crd::Ph>(x_code);
        }
      }
      raise::Error("Invalid dimension index in Rebalance face_phys", HERE);
      return ZERO;
    };
    for (auto d { 0u }; d < M::Dim; ++d) {
      const auto N = g_ndomains_per_dim[d];
      new_offset_per_pos[d].assign(N, 0);
      extent_per_pos[d].resize(N);
      ncells_t running { 0 };
      for (auto p { 0u }; p < N; ++p) {
        new_offset_per_pos[d][p] = running;
        const auto x_lo = face_phys(d, static_cast<real_t>(running));
        running        += new_ncells_per_pos[d][p];
        const auto x_hi = face_phys(d, static_cast<real_t>(running));
        extent_per_pos[d][p] = { x_lo, x_hi };
      }
    }

    /* --- 5. Save local em (and em0/cur0 for GRPIC) to host --------------- */
    auto&      local_dom         = g_subdomains[local_idx];
    const auto old_offset_ncells = local_dom.offset_ncells();

    auto em_old_h = Kokkos::create_mirror_view(local_dom.fields.em);
    Kokkos::deep_copy(em_old_h, local_dom.fields.em);
    auto em0_old_h  = decltype(Kokkos::create_mirror_view(local_dom.fields.em0)) {};
    auto cur0_old_h = decltype(Kokkos::create_mirror_view(local_dom.fields.cur0)) {};
    if constexpr (S == SimEngine::GRPIC) {
      em0_old_h  = Kokkos::create_mirror_view(local_dom.fields.em0);
      cur0_old_h = Kokkos::create_mirror_view(local_dom.fields.cur0);
      Kokkos::deep_copy(em0_old_h, local_dom.fields.em0);
      Kokkos::deep_copy(cur0_old_h, local_dom.fields.cur0);
    }

    /* --- 6. Update bookkeeping for every g_subdomain --------------------- */
    std::vector<ncells_t> new_local_ncells(M::Dim);
    std::vector<ncells_t> new_local_offset(M::Dim);
    for (unsigned int idx { 0 }; idx < g_ndomains; ++idx) {
      auto&                 sub        = g_subdomains[idx];
      const auto&           off_ndoms  = g_domain_offsets[idx];
      std::vector<ncells_t> ncells_d(M::Dim);
      std::vector<ncells_t> offset_d(M::Dim);
      boundaries_t<real_t>  ext_d;
      for (auto d { 0u }; d < M::Dim; ++d) {
        ncells_d[d] = new_ncells_per_pos[d][off_ndoms[d]];
        offset_d[d] = new_offset_per_pos[d][off_ndoms[d]];
        ext_d.push_back(extent_per_pos[d][off_ndoms[d]]);
      }
      sub.mesh.set_resolution_and_extent(ncells_d, ext_d);
      sub.set_offset_ncells(offset_d);
      if (idx == local_idx) {
        new_local_ncells = ncells_d;
        new_local_offset = offset_d;
      }
    }
    // Boundary conditions and neighbor topology are unchanged.

    /* --- 7. Reallocate local fields, copy from saved buffer with shift --- */
    // delta[d] = new_offset[d] - old_offset[d] (in the with-ghost field
    // coordinate system the same delta applies).
    std::vector<int> delta(M::Dim);
    for (auto d { 0u }; d < M::Dim; ++d) {
      delta[d] = static_cast<int>(new_local_offset[d]) -
                 static_cast<int>(old_offset_ncells[d]);
    }

    local_dom.fields = Fields<M::Dim, S> { new_local_ncells };
    CopyShifted<M::Dim, 6>(em_old_h, local_dom.fields.em, delta);
    if constexpr (S == SimEngine::GRPIC) {
      CopyShifted<M::Dim, 6>(em0_old_h, local_dom.fields.em0, delta);
      CopyShifted<M::Dim, 3>(cur0_old_h, local_dom.fields.cur0, delta);
    }

    /* --- 8. Refill ghost zones from neighbors ---------------------------- */
    CommunicateFields(local_dom, Comm::E | Comm::B);

    /* --- 9. Shift particle indices, retag, and migrate ------------------- */
    // Particle indices i_d are in active-cell coordinates: i_d in [0, n_active)
    // for an in-domain particle. After the offset moves by delta, particles
    // get i_d_new = i_d_old - delta[d]. Particles whose new index falls
    // outside [0, n_active) are tagged for the appropriate neighbor and
    // sent by CommunicateParticles().
    for (auto& sp : local_dom.species) {
      if (sp.npart() == 0) {
        continue;
      }
      auto      i1 = sp.i1, i2 = sp.i2, i3 = sp.i3;
      auto      i1p = sp.i1_prev, i2p = sp.i2_prev, i3p = sp.i3_prev;
      auto      tag    = sp.tag;
      const int dx1    = -delta[0];
      int       dx2    = 0;
      int       dx3    = 0;
      int       new_n1 = static_cast<int>(new_local_ncells[0]);
      int       new_n2 = 1;
      int       new_n3 = 1;
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        dx2    = -delta[1];
        new_n2 = static_cast<int>(new_local_ncells[1]);
      }
      if constexpr (M::Dim == Dim::_3D) {
        dx3    = -delta[2];
        new_n3 = static_cast<int>(new_local_ncells[2]);
      }
      Kokkos::parallel_for(
        "RebalanceShiftPrtls",
        sp.rangeActiveParticles(),
        Lambda(prtlidx_t p) {
          if (tag(p) != ParticleTag::alive) {
            return;
          }
          if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or
                        M::Dim == Dim::_3D) {
            i1(p)  += dx1;
            i1p(p) += dx1;
          }
          if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
            i2(p)  += dx2;
            i2p(p) += dx2;
          }
          if constexpr (M::Dim == Dim::_3D) {
            i3(p)  += dx3;
            i3p(p) += dx3;
          }
          if constexpr (M::Dim == Dim::_1D) {
            tag(p) = mpi::SendTag(tag(p), i1(p) < 0, i1(p) >= new_n1);
          } else if constexpr (M::Dim == Dim::_2D) {
            tag(p) = mpi::SendTag(tag(p),
                                  i1(p) < 0,
                                  i1(p) >= new_n1,
                                  i2(p) < 0,
                                  i2(p) >= new_n2);
          } else if constexpr (M::Dim == Dim::_3D) {
            tag(p) = mpi::SendTag(tag(p),
                                  i1(p) < 0,
                                  i1(p) >= new_n1,
                                  i2(p) < 0,
                                  i2(p) >= new_n2,
                                  i3(p) < 0,
                                  i3(p) >= new_n3);
          }
        });
      sp.set_unsorted();
    }

    CommunicateParticles(local_dom);
    logger::Checkpoint("Rebalance: domains shifted, fields and particles redistributed",
                       HERE);
#endif // MPI_ENABLED
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_REBAL(S, M, D)                                              \
  template void Metadomain<S, M<D>>::Rebalance(unsigned int, real_t, ncells_t);

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_REBAL)
#undef METADOMAIN_REBAL
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
