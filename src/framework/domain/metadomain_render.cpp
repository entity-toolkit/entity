/**
 * @file framework/domain/metadomain_render.cpp
 * @brief Metadomain driver for the in-situ volume renderer
 * @implements
 *   - ntt::Metadomain<S, M>::InitRenderer
 *   - ntt::Metadomain<S, M>::Render
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI_ENABLED
 *   - OUTPUT_ENABLED
 * @note
 * This is the templated counterpart of the (plain) out::Renderer: it owns the
 * per-(engine, metric, dim) field preparation, the device ray-march kernel
 * launch, and the device->host copy. It reuses the exact field-prep code paths
 * as Metadomain::Write (ComputeMoments / FieldsToPhys), then hands the per-rank
 * host image to out::Renderer for the MPI ordered composite and PNG write.
 * Active only for 3D Minkowski; a no-op otherwise (the structured-order
 * composite assumes an axis-aligned, affine code<->world map).
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/log.h"

#include "framework/containers/particles.h"
#include "framework/domain/domain.h"
#include "framework/domain/mesh.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"
#include "kernels/fields_to_phys.hpp"
#include "kernels/particle_moments.hpp"
#include "output/render/composite.h"
#include "output/render/fieldlines.h"
#include "output/render/raymarch.hpp"
#include "output/render/reduce.hpp"
#include "output/render/slice2d.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace ntt {

  namespace {

    // Mirror of Metadomain::Write's ComputeMoments, kept with internal linkage
    // so it does not collide with the (identically-named) one in metadomain_io.
    template <SimEngine::type S, MetricClass M, FldsID::type F>
    void renderMoment(const SimulationParams& params,
                      const Mesh<M>&          mesh,
                      const std::vector<Particles<M::Dim, M::CoordType>>& prtl_species,
                      const std::vector<spidx_t>& species,
                      const std::vector<uint8_t>& components,
                      ndfield_t<M::Dim, 6>&       buffer,
                      idx_t                       buff_idx) {
      std::vector<spidx_t> specs = species;
      if (specs.empty()) {
        // default: accumulate over all massive species
        for (auto& sp : prtl_species) {
          if (sp.mass() > 0) {
            specs.push_back(sp.index());
          }
        }
      }
      for (const auto& sp : specs) {
        raise::ErrorIf((sp > prtl_species.size()) or (sp == 0),
                       "Invalid species index " + std::to_string(sp),
                       HERE);
      }
      auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
      const auto use_weights = params.get<bool>("particles.use_weights");
      const auto ni2         = mesh.n_active(in::x2);
      const auto inv_n0      = ONE / params.get<real_t>("scales.n0");
      const auto smooth_order = params.get<unsigned short>(
        "output.fields.smoothing.order");
      const auto smooth_method = OutputSmoothingType::from_string(
        params.get<std::string>("output.fields.smoothing.method"));
      for (const auto& sp : specs) {
        auto& prtl_spec = prtl_species[sp - 1];
        Kokkos::parallel_for(
          "RenderComputeMoments",
          prtl_spec.rangeActiveParticles(),
          kernel::ParticleMoments_kernel<S, M, F, 6>(components,
                                                     scatter_buff,
                                                     buff_idx,
                                                     prtl_spec,
                                                     use_weights,
                                                     mesh.metric,
                                                     mesh.flds_bc(),
                                                     ni2,
                                                     inv_n0,
                                                     smooth_order,
                                                     smooth_method));
      }
      Kokkos::Experimental::contribute(buffer, scatter_buff);
    }

    // Copy 3 contiguous components [from.first, from.first+3) of a source field
    // into bckp(:, 0..2). Dimension-generic (the subview arity depends on D).
    template <Dimension D, int NSRC>
    void copyVec3ToBckp(const ndfield_t<D, NSRC>& src,
                        const ndfield_t<D, 6>&    dst,
                        const cell_range_t&       from) {
      const cell_range_t to { 0, 3 };
      if constexpr (D == Dim::_2D) {
        Kokkos::deep_copy(
          Kokkos::subview(dst, Kokkos::ALL, Kokkos::ALL, to),
          Kokkos::subview(src, Kokkos::ALL, Kokkos::ALL, from));
      } else if constexpr (D == Dim::_3D) {
        Kokkos::deep_copy(
          Kokkos::subview(dst, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, to),
          Kokkos::subview(src, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, from));
      }
    }

    // Volume-average this domain's physical-basis vector field (B/E/J) onto a
    // GLOBAL coarse grid, then MPI-replicate it so every rank holds the same
    // field and can trace identical global field lines locally. `bckp` is used
    // as scratch (overwritten). 3D only (the field-line renderer is Cartesian).
    template <SimEngine::type S, MetricClass M>
    auto buildCoarseFieldVec(const Mesh<M>&            mesh,
                             const Fields<M::Dim, S>&  fields,
                             ndfield_t<M::Dim, 6>&     bckp,
                             char                      fbase,
                             const real_t              gorigin[3],
                             const int                 gnc[3],
                             const real_t              gdx[3]) -> out::CoarseField {
      const auto metric = mesh.metric;
      // raw vector components -> bckp(0,1,2)
      uint8_t            src_base = em::bx1;
      PrepareOutputFlags interp = PrepareOutput::InterpToCellCenterFromFaces;
      bool               is_current = false;
      if (fbase == 'E') {
        src_base = em::ex1;
        interp   = PrepareOutput::InterpToCellCenterFromEdges;
      } else if (fbase == 'J') {
        is_current = true;
        src_base   = cur::jx1;
        interp     = PrepareOutput::InterpToCellCenterFromEdges;
      }
      if (is_current) {
        copyVec3ToBckp<M::Dim, 3>(fields.cur, bckp,
                                  cell_range_t(cur::jx1, cur::jx3 + 1));
      } else {
        copyVec3ToBckp<M::Dim, 6>(fields.em, bckp,
                                  cell_range_t(src_base, src_base + 3));
      }
      // interpolate to cell centers + convert to physical basis -> bckp(3,4,5)
      const PrepareOutputFlags prepare = (S == SimEngine::SRPIC)
                                           ? PrepareOutput::ConvertToHat
                                           : PrepareOutput::ConvertToPhysCntrv;
      list_t<uint8_t, 3>       comp_from = { 0, 1, 2 };
      list_t<uint8_t, 3>       comp_to   = { 3, 4, 5 };
      Kokkos::parallel_for(
        "RenderFLFieldsToPhys",
        mesh.rangeActiveCells(),
        kernel::FieldsToPhys_kernel<M, 6, 6>(bckp, bckp, comp_from, comp_to,
                                             interp | prepare, metric));
      Kokkos::fence();

      // pull the physical components to host and bin into the coarse grid
      auto bckp_h = Kokkos::create_mirror_view(bckp);
      Kokkos::deep_copy(bckp_h, bckp);

      const std::size_t ncell = static_cast<std::size_t>(gnc[0]) *
                                static_cast<std::size_t>(gnc[1]) *
                                static_cast<std::size_t>(gnc[2]);
      std::vector<real_t> sum(ncell * 3, ZERO);
      std::vector<real_t> cnt(ncell, ZERO);

      const auto   le = mesh.extent();
      const real_t llo[3] = { le[0].first, le[1].first, le[2].first };
      const real_t lsz[3] = { le[0].second - le[0].first,
                              le[1].second - le[1].first,
                              le[2].second - le[2].first };
      const int    nl[3]  = { static_cast<int>(mesh.n_active(in::x1)),
                              static_cast<int>(mesh.n_active(in::x2)),
                              static_cast<int>(mesh.n_active(in::x3)) };
      const int    NG     = static_cast<int>(N_GHOSTS);
      for (int k = 0; k < nl[2]; ++k) {
        for (int j = 0; j < nl[1]; ++j) {
          for (int i = 0; i < nl[0]; ++i) {
            const real_t world[3] = {
              llo[0] + (static_cast<real_t>(i) + HALF) * lsz[0] / nl[0],
              llo[1] + (static_cast<real_t>(j) + HALF) * lsz[1] / nl[1],
              llo[2] + (static_cast<real_t>(k) + HALF) * lsz[2] / nl[2]
            };
            int c[3];
            for (int d = 0; d < 3; ++d) {
              int cc = static_cast<int>(
                std::floor((world[d] - gorigin[d]) / gdx[d]));
              cc   = (cc < 0) ? 0 : ((cc > gnc[d] - 1) ? gnc[d] - 1 : cc);
              c[d] = cc;
            }
            const std::size_t lin = (static_cast<std::size_t>(c[2]) * gnc[1] +
                                     c[1]) *
                                      gnc[0] +
                                    c[0];
            sum[lin * 3 + 0] += bckp_h(i + NG, j + NG, k + NG, 3);
            sum[lin * 3 + 1] += bckp_h(i + NG, j + NG, k + NG, 4);
            sum[lin * 3 + 2] += bckp_h(i + NG, j + NG, k + NG, 5);
            cnt[lin]         += ONE;
          }
        }
      }
#if defined(MPI_ENABLED)
      MPI_Allreduce(MPI_IN_PLACE, sum.data(), static_cast<int>(ncell * 3),
                    mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, cnt.data(), static_cast<int>(ncell),
                    mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
#endif
      out::CoarseField cf;
      cf.B.assign(ncell * 3, ZERO);
      for (int d = 0; d < 3; ++d) {
        cf.n[d]      = gnc[d];
        cf.origin[d] = gorigin[d];
        cf.dx[d]     = gdx[d];
      }
      for (std::size_t c = 0; c < ncell; ++c) {
        if (cnt[c] > ZERO) {
          const real_t inv  = ONE / cnt[c];
          cf.B[c * 3 + 0]   = sum[c * 3 + 0] * inv;
          cf.B[c * 3 + 1]   = sum[c * 3 + 1] * inv;
          cf.B[c * 3 + 2]   = sum[c * 3 + 2] * inv;
        }
      }
      return cf;
    }

    // 2D analogue of buildCoarseFieldVec: volume-average the in-plane physical
    // components (Bx, By) onto a coarse global 2D grid and MPI-replicate them,
    // so every rank can integrate the SAME flux function for seamless contours.
    template <SimEngine::type S, MetricClass M>
    auto buildCoarseField2D(const Mesh<M>&           mesh,
                            const Fields<M::Dim, S>& fields,
                            ndfield_t<M::Dim, 6>&    bckp,
                            char                     fbase,
                            const real_t             gorigin[2],
                            const int                gnc[2],
                            const real_t             gdx[2]) -> out::CoarseField2D {
      const auto metric = mesh.metric;
      uint8_t            src_base = em::bx1;
      PrepareOutputFlags interp = PrepareOutput::InterpToCellCenterFromFaces;
      bool               is_current = false;
      if (fbase == 'E') {
        src_base = em::ex1;
        interp   = PrepareOutput::InterpToCellCenterFromEdges;
      } else if (fbase == 'J') {
        is_current = true;
        src_base   = cur::jx1;
        interp     = PrepareOutput::InterpToCellCenterFromEdges;
      }
      if (is_current) {
        copyVec3ToBckp<M::Dim, 3>(fields.cur, bckp,
                                  cell_range_t(cur::jx1, cur::jx3 + 1));
      } else {
        copyVec3ToBckp<M::Dim, 6>(fields.em, bckp,
                                  cell_range_t(src_base, src_base + 3));
      }
      const PrepareOutputFlags prepare = (S == SimEngine::SRPIC)
                                           ? PrepareOutput::ConvertToHat
                                           : PrepareOutput::ConvertToPhysCntrv;
      list_t<uint8_t, 3>       comp_from = { 0, 1, 2 };
      list_t<uint8_t, 3>       comp_to   = { 3, 4, 5 };
      Kokkos::parallel_for(
        "RenderFL2DFieldsToPhys",
        mesh.rangeActiveCells(),
        kernel::FieldsToPhys_kernel<M, 6, 6>(bckp, bckp, comp_from, comp_to,
                                             interp | prepare, metric));
      Kokkos::fence();

      auto bckp_h = Kokkos::create_mirror_view(bckp);
      Kokkos::deep_copy(bckp_h, bckp);

      const std::size_t   ncell = static_cast<std::size_t>(gnc[0]) * gnc[1];
      std::vector<real_t> sum(ncell * 2, ZERO);
      std::vector<real_t> cnt(ncell, ZERO);
      const auto          le = mesh.extent();
      const real_t        llo[2] = { le[0].first, le[1].first };
      const real_t        lsz[2] = { le[0].second - le[0].first,
                                     le[1].second - le[1].first };
      const int           nl[2]  = { static_cast<int>(mesh.n_active(in::x1)),
                                     static_cast<int>(mesh.n_active(in::x2)) };
      const int           NG     = static_cast<int>(N_GHOSTS);
      for (int j = 0; j < nl[1]; ++j) {
        for (int i = 0; i < nl[0]; ++i) {
          const real_t world[2] = {
            llo[0] + (static_cast<real_t>(i) + HALF) * lsz[0] / nl[0],
            llo[1] + (static_cast<real_t>(j) + HALF) * lsz[1] / nl[1]
          };
          int c[2];
          for (int d = 0; d < 2; ++d) {
            int cc = static_cast<int>(
              std::floor((world[d] - gorigin[d]) / gdx[d]));
            cc   = (cc < 0) ? 0 : ((cc > gnc[d] - 1) ? gnc[d] - 1 : cc);
            c[d] = cc;
          }
          const std::size_t lin = static_cast<std::size_t>(c[1]) * gnc[0] + c[0];
          sum[lin * 2 + 0] += bckp_h(i + NG, j + NG, 3); // Bx
          sum[lin * 2 + 1] += bckp_h(i + NG, j + NG, 4); // By
          cnt[lin]         += ONE;
        }
      }
#if defined(MPI_ENABLED)
      MPI_Allreduce(MPI_IN_PLACE, sum.data(), static_cast<int>(ncell * 2),
                    mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, cnt.data(), static_cast<int>(ncell),
                    mpi::get_type<real_t>(), MPI_SUM, MPI_COMM_WORLD);
#endif
      out::CoarseField2D cf;
      cf.B.assign(ncell * 2, ZERO);
      for (int d = 0; d < 2; ++d) {
        cf.n[d]      = gnc[d];
        cf.origin[d] = gorigin[d];
        cf.dx[d]     = gdx[d];
      }
      for (std::size_t c = 0; c < ncell; ++c) {
        if (cnt[c] > ZERO) {
          const real_t inv = ONE / cnt[c];
          cf.B[c * 2 + 0]  = sum[c * 2 + 0] * inv;
          cf.B[c * 2 + 1]  = sum[c * 2 + 1] * inv;
        }
      }
      return cf;
    }

  } // namespace

  template <SimEngine::type S, MetricClass M>
  auto Metadomain<S, M>::prepareRenderScalar(const SimulationParams& params,
                                             Domain<S, M>&           domain,
                                             const std::string&      field_name,
                                             ndfield_t<M::Dim, 6>& bckp) const
    -> bool {
    // Parse an optional trailing per-species suffix "<base>_<s1>_<s2>...";
    // species apply to particle moments only (N, Nppc, Rho, Charge, T, V).
    std::string          base = field_name;
    std::vector<spidx_t> species;
    {
      const auto us = field_name.find('_');
      if (us != std::string::npos) {
        bool        ok_sp = true;
        std::size_t start = us + 1;
        while (start <= field_name.size()) {
          const auto nx  = field_name.find('_', start);
          const auto tok = field_name.substr(
            start,
            (nx == std::string::npos) ? std::string::npos : nx - start);
          if (tok.empty() or
              tok.find_first_not_of("0123456789") != std::string::npos) {
            ok_sp = false;
            break;
          }
          species.push_back(static_cast<spidx_t>(std::stoi(tok)));
          if (nx == std::string::npos) {
            break;
          }
          start = nx + 1;
        }
        if (ok_sp) {
          base = field_name.substr(0, us);
        } else {
          species.clear(); // not a species suffix; keep the full name
        }
      }
    }
    bool bad_species = false;
    for (const auto sp : species) {
      if (sp == 0 or sp > domain.species.size()) {
        bad_species = true;
      }
    }
    if (bad_species) {
      raise::Warning("output.render: invalid species in '" + field_name +
                       "', skipping",
                     HERE);
      return false;
    }

    // axis/index character -> {t,x,y,z} == {0,1,2,3}; -1 if invalid
    auto axisIdx = [](char ch) -> int {
      switch (ch) {
        case 't':
        case '0':
          return 0;
        case 'x':
        case '1':
          return 1;
        case 'y':
        case '2':
          return 2;
        case 'z':
        case '3':
          return 3;
        default:
          return -1;
      }
    };

    const auto& mesh   = domain.mesh;
    const auto  metric = mesh.metric;

    if (base == "N" or base == "Nppc" or base == "Rho" or base == "Charge") {
      // scalar particle moments
      if (base == "N") {
        renderMoment<S, M, FldsID::N>(params, mesh, domain.species, species, {},
                                      bckp, 0u);
      } else if (base == "Nppc") {
        renderMoment<S, M, FldsID::Nppc>(params, mesh, domain.species, species,
                                         {}, bckp, 0u);
      } else if (base == "Rho") {
        renderMoment<S, M, FldsID::Rho>(params, mesh, domain.species, species,
                                        {}, bckp, 0u);
      } else {
        renderMoment<S, M, FldsID::Charge>(params, mesh, domain.species, species,
                                           {}, bckp, 0u);
      }
      // sum boundary-crossing particle deposits back into active cells
      SynchronizeFields(domain, Comm::Bckp, { 0, 1 });
      return true;
    } else if (base.size() == 3 and base[0] == 'T') {
      // a single stress-energy tensor component "T<i><j>" (same for SR & GR;
      // the moment kernel branches on the engine internally)
      const int i = axisIdx(base[1]);
      const int j = axisIdx(base[2]);
      if (i >= 0 and j >= 0) {
        const std::vector<uint8_t> comps { static_cast<uint8_t>(i),
                                           static_cast<uint8_t>(j) };
        renderMoment<S, M, FldsID::T>(params, mesh, domain.species, species,
                                      comps, bckp, 0u);
        SynchronizeFields(domain, Comm::Bckp, { 0, 1 });
        return true;
      }
    } else if (base == "Vmag") {
      // bulk-velocity magnitude |V| = sqrt(V1^2 + V2^2 + V3^2)
      if constexpr (S == SimEngine::GRPIC) {
        // GR: Eckart-frame 4-velocity; need all 4 components for the norm
        renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                      { 0u }, bckp, 0u);
        renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                      { 1u }, bckp, 1u);
        renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                      { 2u }, bckp, 2u);
        renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                      { 3u }, bckp, 3u);
        SynchronizeFields(domain, Comm::Bckp, { 0, 4 });
        Kokkos::parallel_for(
          "RenderNormalize4Vel",
          mesh.rangeActiveCells(),
          kernel::Normalize4VelocityByNorm_kernel<M::Dim, M, 6>(
            bckp, bckp, 0, 1, 2, 3, metric));
        Kokkos::parallel_for(
          "RenderTransform4Vel",
          mesh.rangeActiveCells(),
          kernel::Transform4VelocitySpatialToPhysical_kernel<M::Dim, M, 6>(
            bckp, 1, 2, 3, metric));
        // |spatial physical 4-velocity| -> bckp(0)
        Kokkos::parallel_for("RenderVmagGR",
                             mesh.rangeActiveCells(),
                             kernel::RenderMagnitude3_kernel<M::Dim, 6>(bckp, 1,
                                                                        2, 3, 0));
      } else {
        // SR: mass-weighted bulk 3-velocity, normalized by Rho
        renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                      { 1u }, bckp, 0u);
        renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                      { 2u }, bckp, 1u);
        renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                      { 3u }, bckp, 2u);
        renderMoment<S, M, FldsID::Rho>(params, mesh, domain.species, species,
                                        {}, bckp, 3u);
        SynchronizeFields(domain, Comm::Bckp, { 0, 4 });
        Kokkos::parallel_for("RenderVmagSR",
                             mesh.rangeActiveCells(),
                             kernel::RenderVmagByRho_kernel<M::Dim, 6>(bckp, 0, 1,
                                                                       2, 3, 0));
      }
      return true;
    } else if (base.size() == 2 and base[0] == 'V') {
      // a single bulk-velocity component "V<i>"
      const int c = axisIdx(base[1]);
      if constexpr (S == SimEngine::GRPIC) {
        // GR: 4-velocity component (t/0 = u^0 = Gamma/alpha; x,y,z spatial)
        if (c >= 0 and c <= 3) {
          renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                        { 0u }, bckp, 0u);
          renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                        { 1u }, bckp, 1u);
          renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                        { 2u }, bckp, 2u);
          renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                        { 3u }, bckp, 3u);
          SynchronizeFields(domain, Comm::Bckp, { 0, 4 });
          Kokkos::parallel_for(
            "RenderNormalize4Vel",
            mesh.rangeActiveCells(),
            kernel::Normalize4VelocityByNorm_kernel<M::Dim, M, 6>(
              bckp, bckp, 0, 1, 2, 3, metric));
          Kokkos::parallel_for(
            "RenderTransform4Vel",
            mesh.rangeActiveCells(),
            kernel::Transform4VelocitySpatialToPhysical_kernel<M::Dim, M, 6>(
              bckp, 1, 2, 3, metric));
          if (c != 0) {
            Kokkos::parallel_for(
              "RenderPickV",
              mesh.rangeActiveCells(),
              kernel::RenderPickComp_kernel<M::Dim, 6>(
                bckp, static_cast<uint8_t>(c), 0));
          }
          return true;
        }
      } else {
        // SR: spatial bulk velocity (x,y,z), normalized by Rho
        if (c >= 1 and c <= 3) {
          renderMoment<S, M, FldsID::V>(params, mesh, domain.species, species,
                                        { static_cast<uint8_t>(c) }, bckp, 0u);
          renderMoment<S, M, FldsID::Rho>(params, mesh, domain.species, species,
                                          {}, bckp, 1u);
          SynchronizeFields(domain, Comm::Bckp, { 0, 2 });
          Kokkos::parallel_for("RenderNormalizeV",
                               mesh.rangeActiveCells(),
                               kernel::RenderDivideComp_kernel<M::Dim, 6>(bckp, 0,
                                                                          1));
          return true;
        }
      }
    } else {
      // Vector field as a scalar: "<base><selector>" with base in {E, B, J}
      // and selector in {mag, 1/2/3, x/y/z}. A component (e.g. "B1"/"Bx") is
      // signed; a magnitude (e.g. "Bmag") is non-negative.
      const std::string& f     = base;
      const char         fbase = f.empty()
                                   ? '?'
                                   : static_cast<char>(std::toupper(f[0]));
      bool               ok         = true;
      bool               is_current = false;
      uint8_t            src_base   = 0; // first component of the source field
      PrepareOutputFlags interp     = PrepareOutput::None;
      if (fbase == 'B') {
        src_base = em::bx1;
        interp   = PrepareOutput::InterpToCellCenterFromFaces;
      } else if (fbase == 'E') {
        src_base = em::ex1;
        interp   = PrepareOutput::InterpToCellCenterFromEdges;
      } else if (fbase == 'J') {
        is_current = true;
        src_base   = cur::jx1;
        interp     = PrepareOutput::InterpToCellCenterFromEdges;
      } else {
        ok = false;
      }
      // selector: -1 = magnitude, 0/1/2 = a single component
      int               comp = -2;
      const std::string sel  = (f.size() > 1) ? f.substr(1) : std::string {};
      if (sel == "mag") {
        comp = -1;
      } else if (sel == "1" or sel == "x") {
        comp = 0;
      } else if (sel == "2" or sel == "y") {
        comp = 1;
      } else if (sel == "3" or sel == "z") {
        comp = 2;
      } else {
        ok = false;
      }
      if (ok) {
        // raw vector components into bckp(:, 0..2)
        if (is_current) {
          copyVec3ToBckp<M::Dim, 3>(domain.fields.cur, bckp,
                                    cell_range_t(cur::jx1, cur::jx3 + 1));
        } else {
          copyVec3ToBckp<M::Dim, 6>(domain.fields.em, bckp,
                                    cell_range_t(src_base, src_base + 3));
        }
        // interpolate to cell centers + convert to physical basis -> (3,4,5)
        const PrepareOutputFlags prepare = (S == SimEngine::SRPIC)
                                             ? PrepareOutput::ConvertToHat
                                             : PrepareOutput::ConvertToPhysCntrv;
        list_t<uint8_t, 3>       comp_from = { 0, 1, 2 };
        list_t<uint8_t, 3>       comp_to   = { 3, 4, 5 };
        Kokkos::parallel_for(
          "RenderFieldsToPhys",
          mesh.rangeActiveCells(),
          kernel::FieldsToPhys_kernel<M, 6, 6>(bckp,
                                               bckp,
                                               comp_from,
                                               comp_to,
                                               interp | prepare,
                                               metric));
        // reduce to the scalar to render -> bckp(:, 0)
        if (comp == -1) {
          Kokkos::parallel_for(
            "RenderVectorMagnitude",
            mesh.rangeActiveCells(),
            kernel::RenderMagnitude3_kernel<M::Dim, 6>(bckp, 3, 4, 5, 0));
        } else {
          Kokkos::parallel_for(
            "RenderVectorComponent",
            mesh.rangeActiveCells(),
            kernel::RenderPickComp_kernel<M::Dim, 6>(
              bckp, static_cast<uint8_t>(3 + comp), 0));
        }
        return true;
      }
    }

    raise::Warning("output.render: unknown field '" + field_name +
                     "' (expected N/Nppc/Rho/Charge, T{i}{j}, V{i}/Vmag, or "
                     "{E,B,J}{mag,1,2,3,x,y,z}); skipping",
                   HERE);
    return false;
  }

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::InitRenderer(const SimulationParams& params) {
    g_renderer.init(params, mesh().extent());
  }

  template <SimEngine::type S, MetricClass M>
  auto Metadomain<S, M>::Render(const SimulationParams& params,
                                timestep_t              current_step,
                                timestep_t              finished_step,
                                simtime_t               current_time,
                                simtime_t               finished_time) -> bool {
    (void)current_time;
    if constexpr (M::Dim == Dim::_3D and M::CoordType == Coord::type::Cartesian) {
      // ---- 3D volume ray-march (Minkowski only) ----------------------- //
      // structured-order composite assumes an axis-aligned, affine code<->world
      // map; only Cartesian (Minkowski) 3D qualifies.
      if (not g_renderer.enabled() or
          not g_renderer.shouldRender(finished_step, finished_time)) {
        return false;
      }
      raise::ErrorIf(l_subdomain_indices().size() != 1,
                     "Renderer supports one subdomain per rank only",
                     HERE);
      auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
      raise::ErrorIf(local_domain->is_placeholder(),
                     "local_domain is a placeholder",
                     HERE);
      logger::Checkpoint("Rendering output (3D volume)", HERE);

      // advance the moving view (region + camera) to this frame's time before
      // reading camera()/region(); collective (same time on all ranks).
      g_renderer.updateForTime(current_time);

      const auto& cam = g_renderer.camera();
      const int   W   = g_renderer.width();
      const int   H   = g_renderer.height();

      // optional axis-aligned render region (== full extent when uncropped)
      const real_t rlo[3] = { g_renderer.regionLo(0), g_renderer.regionLo(1),
                              g_renderer.regionLo(2) };
      const real_t rhi[3] = { g_renderer.regionHi(0), g_renderer.regionHi(1),
                              g_renderer.regionHi(2) };
      // per-domain world AABB, clipped to the region
      const auto loc_ext = local_domain->mesh.extent();
      real_t     lo[3]   = { math::max(loc_ext[0].first, rlo[0]),
                             math::max(loc_ext[1].first, rlo[1]),
                             math::max(loc_ext[2].first, rlo[2]) };
      real_t     hi[3]   = { math::min(loc_ext[0].second, rhi[0]),
                             math::min(loc_ext[1].second, rhi[1]),
                             math::min(loc_ext[2].second, rhi[2]) };
      // does this domain intersect the region? if not, render nothing (but still
      // join the collective composite / field-line reduce below).
      const bool in_region = (lo[0] < hi[0]) and (lo[1] < hi[1]) and
                             (lo[2] < hi[2]);

      // global extent (drives the field-line coarse grid, which spans the full
      // field regardless of the crop)
      const auto glob_ext = mesh().extent();
      // fixed world step, identical on all ranks -> seamless. Sized to the region
      // diagonal so `samples` spans the (possibly cropped) view.
      real_t gdiag = ZERO;
      for (auto d { 0 }; d < 3; ++d) {
        const real_t s = rhi[d] - rlo[d];
        gdiag          += s * s;
      }
      gdiag = math::sqrt(gdiag);
      const real_t ds = (g_renderer.stepSize() > ZERO)
                          ? g_renderer.stepSize()
                          : gdiag / static_cast<real_t>(g_renderer.samples());
      const int max_steps = 2 * g_renderer.samples() + 16;

      // region box + depth-occluded spine (opaque box wireframe rendered inline
      // in the march so the volume covers its far edges). The visual width is
      // ~spine_width px; the 0.55*ds floor keeps the thin line gap-free at the
      // current sampling (raise `samples` for a crisper, thinner line).
      real_t       glo[3] = { rlo[0], rlo[1], rlo[2] };
      real_t       ghi[3] = { rhi[0], rhi[1], rhi[2] };
      const real_t px_w   = (cam.half_h * static_cast<real_t>(2)) /
                          static_cast<real_t>(H);
      const real_t spine_radius =
        g_renderer.axes()
          ? math::max(static_cast<real_t>(0.55) * ds,
                      HALF * g_renderer.spineWidth() * px_w)
          : ZERO;
      // contrasting opaque spine color (white on dark bg, black on light)
      const real_t bg_lum = static_cast<real_t>(0.299) * g_renderer.background(0) +
                            static_cast<real_t>(0.587) * g_renderer.background(1) +
                            static_cast<real_t>(0.114) * g_renderer.background(2);
      const real_t sc        = (bg_lum < HALF) ? ONE : ZERO;
      const real_t spine_rgb[3] = { sc, sc, sc };

      // composite order key (depends on the current decomposition offsets)
      const uint64_t order_key = out::compositeOrderKey(
        local_domain->offset_ndomains(),
        ndomains_per_dim(),
        cam.forward);

      auto&     bckp = local_domain->fields.bckp;
      const int ext0 = static_cast<int>(bckp.extent(0));
      const int ext1 = static_cast<int>(bckp.extent(1));
      const int ext2 = static_cast<int>(bckp.extent(2));

      const auto metric = local_domain->mesh.metric;

      // screen-space bounding box of this domain's footprint (same for all
      // scenes); we only ray-march and composite within it.
      int        bx0 = 0, by0 = 0, bw = 0, bh = 0;
      const bool on_screen = in_region and
                             out::screenBBox(cam, W, H, lo, hi, bx0, by0, bw, bh);

      // ---- magnetic-field-line tubes (built once, shared by every scene) --- //
      // Every rank coarsens + replicates the field, traces the SAME global
      // polylines, and keeps only the segments inside its own domain; the
      // ordered cross-domain composite stitches them. Built before the scene
      // loop so an overlay and a standalone tube scene share one geometry pass.
      // NB: all ranks reach this together (cadence is collective), so the
      // Allreduce inside buildCoarseFieldVec is safe.
      const auto&  flc        = g_renderer.fieldlines();
      out::TubeSet tubes      = out::emptyTubeSet();
      out::TubeSet empty      = out::emptyTubeSet();
      bool         have_tubes = false;
      if (flc.enable) {
        const int gN[3] = { static_cast<int>(mesh().n_active(in::x1)),
                            static_cast<int>(mesh().n_active(in::x2)),
                            static_cast<int>(mesh().n_active(in::x3)) };
        int       gnc[3];
        real_t    gorigin[3], gdx[3];
        for (int d = 0; d < 3; ++d) {
          gnc[d]     = std::max(1, (gN[d] + flc.bin - 1) / flc.bin);
          gorigin[d] = glob_ext[d].first;
          gdx[d]     = (glob_ext[d].second - glob_ext[d].first) / gnc[d];
        }
        const char fb = static_cast<char>(
          std::toupper(flc.field.empty() ? 'B' : flc.field[0]));
        out::CoarseField cf = buildCoarseFieldVec<S, M>(
          local_domain->mesh, local_domain->fields, bckp, fb, gorigin, gnc, gdx);
        // seed/tube scale: world units per screen pixel (orthographic frame)
        const real_t wpp = (cam.half_h * TWO) / static_cast<real_t>(H);
        real_t       vlo, vhi;
        auto         lines = out::traceFieldLines(cf, flc, wpp, vlo, vhi);
        if (flc.vmax > flc.vmin) { // explicit color range overrides auto
          vlo = flc.vmin;
          vhi = flc.vmax;
        }
        const real_t tube_world = math::max(flc.tube_px, ONE) * wpp;
        const real_t eff_r = math::max(tube_world,
                                       static_cast<real_t>(0.55) * ds);
        std::size_t n_kept = 0;
        tubes      = out::buildTubeSet(lines, eff_r, flc, vlo, vhi, lo, hi, cf,
                                       n_kept);
        have_tubes = true;
        logger::Checkpoint("field lines: " + std::to_string(lines.size()) +
                             " global lines, " + std::to_string(n_kept) +
                             " local segments",
                           HERE);
      }

      bool rendered_any = false;
      for (const auto& scene : g_renderer.scenes()) {
        // a `field = "fieldlines"` scene renders the tubes standalone (no
        // volume); any other scene may overlay them inside its volume.
        const bool fl_only    = (scene.field == "fieldlines");
        const bool volume_on  = not fl_only;
        const bool show_tubes = scene.show_fieldlines and have_tubes;
        if (volume_on) {
          Kokkos::deep_copy(bckp, ZERO);
          if (not prepareRenderScalar(params, *local_domain, scene.field, bckp)) {
            continue;
          }
          // fill the ghost halo with neighbor active values so trilinear
          // sampling is C0 across domain faces (a halo EXCHANGE, not the
          // sum-into-active that SynchronizeFields performs).
          CommunicateBckp(*local_domain, { 0, 1 });
        } else if (not have_tubes) {
          // standalone field-line scene but tracing produced nothing/disabled
          raise::Warning("output.render: 'fieldlines' scene but no field-line "
                         "geometry; skipping",
                         HERE);
          continue;
        }
        const out::TubeSet& kt = show_tubes ? tubes : empty;
        // a standalone tube scene colors its colorbar by |field|, not by the
        // (unused) volume transfer function
        out::Scene scene_cb = scene;
        if (fl_only) {
          scene_cb.tf.vmin      = tubes.vmin;
          scene_cb.tf.vmax      = tubes.vmax;
          scene_cb.tf.log_scale = tubes.log_scale;
          scene_cb.tf.colormap  = tubes.colormap;
          if (scene_cb.label == "fieldlines") {
            scene_cb.label = "|" + flc.field + "|";
          }
        }

        out::SubImage sub;
        if (on_screen) {
          sub.x0 = bx0;
          sub.y0 = by0;
          sub.w  = bw;
          sub.h  = bh;
          const std::size_t bnpix = static_cast<std::size_t>(bw) *
                                    static_cast<std::size_t>(bh);
          array_t<real_t* [4]>         image { "render_img", bnpix };
          randacc_ndfield_t<M::Dim, 6> Fld { bckp };
          Kokkos::parallel_for(
            "VolumeRayMarch",
            CreateRangePolicy<Dim::_2D>({ 0, 0 },
                                        { static_cast<ncells_t>(bw),
                                          static_cast<ncells_t>(bh) }),
            kernel::VolumeRayMarch_kernel<M>(Fld,
                                             0u,
                                             metric,
                                             cam,
                                             lo,
                                             hi,
                                             ext0,
                                             ext1,
                                             ext2,
                                             W,
                                             H,
                                             bx0,
                                             by0,
                                             bw,
                                             ds,
                                             max_steps,
                                             scene.tf.lut,
                                             scene.tf.n_lut,
                                             scene.tf.vmin,
                                             scene.tf.vmax,
                                             scene.tf.log_scale,
                                             g_renderer.earlyAlpha(),
                                             glo,
                                             ghi,
                                             spine_radius,
                                             spine_rgb,
                                             kt,
                                             volume_on,
                                             image));
          Kokkos::fence();

          // device -> host, into a layout-agnostic pixel-major buffer
          auto image_h = Kokkos::create_mirror_view(image);
          Kokkos::deep_copy(image_h, image);
          sub.rgba.resize(bnpix * 4);
          for (std::size_t p = 0; p < bnpix; ++p) {
            sub.rgba[p * 4 + 0] = image_h(p, 0);
            sub.rgba[p * 4 + 1] = image_h(p, 1);
            sub.rgba[p * 4 + 2] = image_h(p, 2);
            sub.rgba[p * 4 + 3] = image_h(p, 3);
          }
        }
        g_renderer.compositeAndWrite(sub, order_key, scene_cb, current_step,
                                     current_time);
        rendered_any = true;
      }
      return rendered_any;
    } else if constexpr (M::Dim == Dim::_2D) {
      // ---- 2D slice rasterizer (Cartesian or spherical) --------------- //
      // A 2D run has no depth to integrate: each pixel is one inverse-mapped
      // sample, painted opaque. Domains tile the screen disjointly, so the
      // sparse sub-images composite seamlessly regardless of order.
      if (not g_renderer.enabled() or
          not g_renderer.shouldRender(finished_step, finished_time)) {
        return false;
      }
      raise::ErrorIf(l_subdomain_indices().size() != 1,
                     "Renderer supports one subdomain per rank only",
                     HERE);
      auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
      raise::ErrorIf(local_domain->is_placeholder(),
                     "local_domain is a placeholder",
                     HERE);
      logger::Checkpoint("Rendering output (2D slice)", HERE);

      // advance the moving view (region window) to this frame's time before
      // reading region(); collective (same time on all ranks).
      g_renderer.updateForTime(current_time);

      const int  W      = g_renderer.width();
      const int  H      = g_renderer.height();
      const bool mirror = g_renderer.mirror();

      // global slice-plane world window (shared by all ranks -> seamless),
      // taken from the optional render region (== full extent when uncropped).
      // gext (the full extent) is kept for the field-line coarse grid below.
      const auto   gext = mesh().extent();
      const real_t x1lo = g_renderer.regionLo(0), x1hi = g_renderer.regionHi(0);
      const real_t x2lo = g_renderer.regionLo(1), x2hi = g_renderer.regionHi(1);
      real_t       umin, umax, vmin, vmax;
      if constexpr (M::CoordType == Coord::type::Cartesian) {
        umin = x1lo;
        umax = x1hi;
        vmin = x2lo;
        vmax = x2hi;
      } else {
        // meridional (X = r sin th, Z = r cos th) bounding box of the cropped
        // annular wedge r in [x1lo, x1hi], theta in [x2lo, x2hi]. Sample the
        // boundary (arcs + rays) so the bbox is correct for any theta range.
        umin = static_cast<real_t>(1e30);
        umax = static_cast<real_t>(-1e30);
        vmin = static_cast<real_t>(1e30);
        vmax = static_cast<real_t>(-1e30);
        const int NB = 65;
        auto      accXZ = [&](real_t r, real_t th) {
          const real_t X = r * math::sin(th), Z = r * math::cos(th);
          umin = std::min(umin, X);
          umax = std::max(umax, X);
          vmin = std::min(vmin, Z);
          vmax = std::max(vmax, Z);
          if (mirror) {
            umin = std::min(umin, -X);
            umax = std::max(umax, -X);
          }
        };
        for (int k = 0; k < NB; ++k) {
          const real_t t  = static_cast<real_t>(k) / static_cast<real_t>(NB - 1);
          const real_t th = x2lo + (x2hi - x2lo) * t;
          const real_t rr = x1lo + (x1hi - x1lo) * t;
          accXZ(x1lo, th);
          accXZ(x1hi, th);
          accXZ(rr, x2lo);
          accXZ(rr, x2hi);
        }
      }
      // expand the window to the image aspect (centered) so geometry is not
      // stretched
      {
        const real_t waspect = (umax - umin) / (vmax - vmin);
        const real_t iaspect = static_cast<real_t>(W) / static_cast<real_t>(H);
        if (iaspect > waspect) {
          const real_t cu = HALF * (umin + umax);
          const real_t hu = HALF * (vmax - vmin) * iaspect;
          umin = cu - hu;
          umax = cu + hu;
        } else {
          const real_t cv = HALF * (vmin + vmax);
          const real_t hv = HALF * (umax - umin) / iaspect;
          vmin = cv - hv;
          vmax = cv + hv;
        }
      }
      // spherical slices get a background border so the round outline and its
      // R/theta labels are not clipped at the frame edges (Cartesian fills the
      // frame and draws its ticks in dedicated margins, so it needs none).
      if constexpr (M::CoordType != Coord::type::Cartesian) {
        const real_t pad = static_cast<real_t>(1.12);
        const real_t cu = HALF * (umin + umax), hu = HALF * (umax - umin) * pad;
        const real_t cv = HALF * (vmin + vmax), hv = HALF * (vmax - vmin) * pad;
        umin = cu - hu;
        umax = cu + hu;
        vmin = cv - hv;
        vmax = cv + hv;
      }

      // hand the world window + axis names to the (host) axes overlay. Default
      // names follow the coordinate family unless the toml set axis_labels.
      {
        const bool        sph = (M::CoordType != Coord::type::Cartesian);
        const std::string xl  = g_renderer.axisLabelsSet()
                                  ? g_renderer.axisLabel(0)
                                  : (sph ? std::string("X") : std::string("x"));
        const std::string yl  = g_renderer.axisLabelsSet()
                                  ? g_renderer.axisLabel(1)
                                  : (sph ? std::string("Z") : std::string("y"));
        g_renderer.setSliceFrame(umin, umax, vmin, vmax, xl, yl);
        // curvilinear slices get polar axes (R radial + Theta arc); pass the
        // global (r, theta) extent.
        if (sph) {
          g_renderer.setSlicePolar(true, x1lo, x1hi, x2lo, x2hi, mirror);
        } else {
          g_renderer.setSlicePolar(false, ZERO, ONE, ZERO, ONE, mirror);
        }
      }

      auto&      bckp   = local_domain->fields.bckp;
      const int  ext0   = static_cast<int>(bckp.extent(0));
      const int  ext1   = static_cast<int>(bckp.extent(1));
      const auto metric = local_domain->mesh.metric;
      const int  n1 = static_cast<int>(local_domain->mesh.n_active(in::x1));
      const int  n2 = static_cast<int>(local_domain->mesh.n_active(in::x2));

      // screen-space bbox of this domain's footprint (host projection of the
      // boundary; an arc for spherical, a box for Cartesian)
      const auto   le    = local_domain->mesh.extent();
      auto         toPix = [&](real_t u, real_t v, real_t& px, real_t& py) {
        px = (u - umin) / (umax - umin) * static_cast<real_t>(W) - HALF;
        py = (vmax - v) / (vmax - vmin) * static_cast<real_t>(H) - HALF;
      };
      real_t minx = static_cast<real_t>(1e30), miny = static_cast<real_t>(1e30);
      real_t maxx = static_cast<real_t>(-1e30), maxy = static_cast<real_t>(-1e30);
      auto   acc = [&](real_t u, real_t v) {
        real_t px, py;
        toPix(u, v, px, py);
        minx = std::min(minx, px);
        maxx = std::max(maxx, px);
        miny = std::min(miny, py);
        maxy = std::max(maxy, py);
      };
      if constexpr (M::CoordType == Coord::type::Cartesian) {
        acc(le[0].first, le[1].first);
        acc(le[0].second, le[1].first);
        acc(le[0].first, le[1].second);
        acc(le[0].second, le[1].second);
      } else {
        const int    NB = 33;
        const real_t r0 = le[0].first, r1 = le[0].second;
        const real_t a0 = le[1].first, a1 = le[1].second;
        for (int k = 0; k < NB; ++k) {
          const real_t t = static_cast<real_t>(k) / static_cast<real_t>(NB - 1);
          const real_t rr = r0 + (r1 - r0) * t;
          const real_t aa = a0 + (a1 - a0) * t;
          // r-arcs at a0, a1 and theta-rays at r0, r1
          const real_t pts[4][2] = { { r0 * math::sin(aa), r0 * math::cos(aa) },
                                     { r1 * math::sin(aa), r1 * math::cos(aa) },
                                     { rr * math::sin(a0), rr * math::cos(a0) },
                                     { rr * math::sin(a1), rr * math::cos(a1) } };
          for (auto& p : pts) {
            acc(p[0], p[1]);
            if (mirror) {
              acc(-p[0], p[1]);
            }
          }
        }
      }
      const int pad = 2;
      int       x0  = static_cast<int>(std::floor(minx)) - pad;
      int       x1  = static_cast<int>(std::ceil(maxx)) + pad;
      int       y0  = static_cast<int>(std::floor(miny)) - pad;
      int       y1  = static_cast<int>(std::ceil(maxy)) + pad;
      x0            = std::max(0, std::min(W, x0));
      x1            = std::max(0, std::min(W, x1));
      y0            = std::max(0, std::min(H, y0));
      y1            = std::max(0, std::min(H, y1));
      const int bx0 = x0, by0 = y0, bw = x1 - x0, bh = y1 - y0;

      // disjoint tiling -> any consistent total order composites correctly;
      // a lexicographic key over the decomposition offsets is unique per rank.
      const real_t   fwd2d[3]  = { ONE, ONE, ZERO };
      const uint64_t order_key = out::compositeOrderKey(
        local_domain->offset_ndomains(),
        ndomains_per_dim(),
        fwd2d);

      // ---- 2D field lines (built once) ------------------------------------ //
      // Cartesian: iso-contours of the flux function psi. Spherical/Kerr: traced
      // meridional streamlines (nt2py style). Both come from a coarse, MPI-
      // replicated copy of the in-plane field, so the geometry is global and
      // seamless across the disjoint tiles. All ranks reach buildCoarseField2D
      // together (collective Allreduce); M is fixed per run, so every rank takes
      // the same Cartesian/spherical branch.
      const auto&     flc         = g_renderer.fieldlines();
      out::ContourSet contours    = out::emptyContourSet();
      out::ContourSet emptyc      = out::emptyContourSet();
      out::TubeSet    lines2d     = out::emptyTubeSet();
      out::TubeSet    emptyl      = out::emptyTubeSet();
      bool            have_fl     = false;
      real_t          fl_vmin     = ZERO, fl_vmax = ONE;
      std::string     fl_colormap = flc.colormap;
      if (flc.enable) {
        const int gN[2] = { static_cast<int>(mesh().n_active(in::x1)),
                            static_cast<int>(mesh().n_active(in::x2)) };
        int       gnc[2];
        real_t    gorigin[2], gdx[2];
        for (int d = 0; d < 2; ++d) {
          gnc[d]     = std::max(1, (gN[d] + flc.bin - 1) / flc.bin);
          gorigin[d] = gext[d].first;
          gdx[d]     = (gext[d].second - gext[d].first) / gnc[d];
        }
        const char fb = static_cast<char>(
          std::toupper(flc.field.empty() ? 'B' : flc.field[0]));
        // coarse, replicated in-plane field: (Bx,By) for Cartesian, (Br,Bth) for
        // spherical (FieldsToPhys writes the physical components in axis order)
        out::CoarseField2D cf = buildCoarseField2D<S, M>(
          local_domain->mesh, local_domain->fields, bckp, fb, gorigin, gnc, gdx);
        const real_t wpp = (umax - umin) / static_cast<real_t>(W);
        if constexpr (M::CoordType == Coord::type::Cartesian) {
          std::vector<real_t> psi;
          real_t              pmin, pmax, bmin, bmax;
          out::computeFlux2D(cf, psi, pmin, pmax, bmin, bmax);
          const real_t line_half = HALF * math::max(flc.tube_px, ONE);
          contours    = out::buildContourSet(cf, psi, pmin, pmax, bmin, bmax, flc,
                                             line_half, wpp);
          fl_vmin     = contours.vmin;
          fl_vmax     = contours.vmax;
          fl_colormap = contours.colormap;
          logger::Checkpoint("field lines (2D): " + std::to_string(flc.levels) +
                               " psi contours on a " + std::to_string(gnc[0]) +
                               "x" + std::to_string(gnc[1]) + " grid",
                             HERE);
        } else {
          // traced meridional streamlines through the coarse (r, theta) field
          real_t vlo, vhi;
          auto   poly = out::traceFieldLinesMeridional(cf, flc, wpp, mirror, vlo,
                                                       vhi);
          if (flc.vmax > flc.vmin) { // explicit |B| range overrides auto
            vlo = flc.vmin;
            vhi = flc.vmax;
          }
          const real_t eff_r = math::max(flc.tube_px, ONE) * wpp;
          // bucket grid for buildTubeSet: cell ~ a coarse dr (a length), AABB
          // spans the (mirrored) meridional disk, z is a single thin slab at 0
          out::CoarseField bucket_cf;
          bucket_cf.dx[0]    = cf.dx[0];
          bucket_cf.dx[1]    = cf.dx[0];
          bucket_cf.dx[2]    = cf.dx[0];
          const real_t rmax  = gext[0].second;
          const real_t lo[3] = { mirror ? -rmax : ZERO, -rmax, -cf.dx[0] };
          const real_t hi[3] = { rmax, rmax, cf.dx[0] };
          std::size_t  n_kept = 0;
          lines2d     = out::buildTubeSet(poly, eff_r, flc, vlo, vhi, lo, hi,
                                          bucket_cf, n_kept);
          fl_vmin     = lines2d.vmin;
          fl_vmax     = lines2d.vmax;
          fl_colormap = lines2d.colormap;
          logger::Checkpoint("field lines (2D meridional): " +
                               std::to_string(poly.size()) + " lines, " +
                               std::to_string(n_kept) + " segments",
                             HERE);
        }
        have_fl = true;
      }

      bool rendered_any = false;
      for (const auto& scene : g_renderer.scenes()) {
        // standalone `field = "fieldlines"` -> lines only (no heatmap fill); any
        // other scene with `fieldlines = true` overlays them on its heatmap.
        const bool fl_only    = (scene.field == "fieldlines");
        const bool heatmap_on = not fl_only;
        const bool show_lines = scene.show_fieldlines and have_fl;
        if (heatmap_on) {
          Kokkos::deep_copy(bckp, ZERO);
          if (not prepareRenderScalar(params, *local_domain, scene.field, bckp)) {
            continue;
          }
          CommunicateBckp(*local_domain, { 0, 1 });
        } else if (not have_fl) {
          raise::Warning("output.render: 'fieldlines' scene needs a 2D run with "
                         "[output.render.fieldlines]; skipping",
                         HERE);
          continue;
        }
        const out::ContourSet& kc = show_lines ? contours : emptyc;
        const out::TubeSet&    kt = show_lines ? lines2d : emptyl;
        // a standalone field-line scene colors its colorbar by |B|
        out::Scene scene_cb = scene;
        if (fl_only) {
          scene_cb.tf.vmin      = fl_vmin;
          scene_cb.tf.vmax      = fl_vmax;
          scene_cb.tf.log_scale = false;
          scene_cb.tf.colormap  = fl_colormap;
          if (scene_cb.label == "fieldlines") {
            scene_cb.label = "|" + flc.field + "|";
          }
        }

        out::SubImage sub;
        if (bw > 0 and bh > 0) {
          sub.x0 = bx0;
          sub.y0 = by0;
          sub.w  = bw;
          sub.h  = bh;
          const std::size_t bnpix = static_cast<std::size_t>(bw) *
                                    static_cast<std::size_t>(bh);
          array_t<real_t* [4]>         image { "render_img", bnpix };
          randacc_ndfield_t<M::Dim, 6> Fld { bckp };
          Kokkos::parallel_for(
            "Slice2DRaster",
            CreateRangePolicy<Dim::_2D>({ 0, 0 },
                                        { static_cast<ncells_t>(bw),
                                          static_cast<ncells_t>(bh) }),
            kernel::SliceRaster_kernel<M>(Fld,
                                          0u,
                                          metric,
                                          umin,
                                          umax,
                                          vmin,
                                          vmax,
                                          W,
                                          H,
                                          bx0,
                                          by0,
                                          bw,
                                          mirror,
                                          x1lo,
                                          x1hi,
                                          x2lo,
                                          x2hi,
                                          g_renderer.hasRegion(),
                                          n1,
                                          n2,
                                          ext0,
                                          ext1,
                                          scene.tf.lut_opaque,
                                          scene.tf.n_lut,
                                          scene.tf.vmin,
                                          scene.tf.vmax,
                                          scene.tf.log_scale,
                                          kc,
                                          kt,
                                          heatmap_on,
                                          image));
          Kokkos::fence();

          auto image_h = Kokkos::create_mirror_view(image);
          Kokkos::deep_copy(image_h, image);
          sub.rgba.resize(bnpix * 4);
          for (std::size_t p = 0; p < bnpix; ++p) {
            sub.rgba[p * 4 + 0] = image_h(p, 0);
            sub.rgba[p * 4 + 1] = image_h(p, 1);
            sub.rgba[p * 4 + 2] = image_h(p, 2);
            sub.rgba[p * 4 + 3] = image_h(p, 3);
          }
        }
        g_renderer.compositeAndWrite(sub, order_key, scene_cb, current_step,
                                     current_time);
        rendered_any = true;
      }
      return rendered_any;
    } else {
      (void)params;
      (void)current_step;
      (void)finished_step;
      (void)finished_time;
      return false;
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_RENDER(S, M, D)                                             \
  template void Metadomain<S, M<D>>::InitRenderer(const SimulationParams&);    \
  template auto Metadomain<S, M<D>>::Render(const SimulationParams&,           \
                                            timestep_t,                        \
                                            timestep_t,                        \
                                            simtime_t,                         \
                                            simtime_t) -> bool;                \
  template auto Metadomain<S, M<D>>::prepareRenderScalar(                      \
    const SimulationParams&,                                                   \
    Domain<S, M<D>>&,                                                          \
    const std::string&,                                                        \
    ndfield_t<M<D>::Dim, 6>&) const -> bool;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_RENDER)

#undef METADOMAIN_RENDER
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
