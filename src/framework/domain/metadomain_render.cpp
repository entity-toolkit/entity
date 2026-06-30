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
#include "output/render/raymarch.hpp"
#include "output/render/reduce.hpp"
#include "output/render/slice2d.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

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

      const auto& cam = g_renderer.camera();
      const int   W   = g_renderer.width();
      const int   H   = g_renderer.height();

      // per-domain world AABB
      const auto loc_ext = local_domain->mesh.extent();
      real_t     lo[3]   = { loc_ext[0].first, loc_ext[1].first, loc_ext[2].first };
      real_t     hi[3]   = { loc_ext[0].second, loc_ext[1].second, loc_ext[2].second };

      // fixed global world step (identical on all ranks -> seamless)
      const auto glob_ext = mesh().extent();
      real_t     gdiag    = ZERO;
      for (auto d { 0 }; d < 3; ++d) {
        const real_t s = glob_ext[d].second - glob_ext[d].first;
        gdiag          += s * s;
      }
      gdiag = math::sqrt(gdiag);
      const real_t ds = (g_renderer.stepSize() > ZERO)
                          ? g_renderer.stepSize()
                          : gdiag / static_cast<real_t>(g_renderer.samples());
      const int max_steps = 2 * g_renderer.samples() + 16;

      // global box + depth-occluded spine (opaque box wireframe rendered inline
      // in the march so the volume covers its far edges). The visual width is
      // ~spine_width px; the 0.55*ds floor keeps the thin line gap-free at the
      // current sampling (raise `samples` for a crisper, thinner line).
      real_t       glo[3] = { glob_ext[0].first, glob_ext[1].first,
                              glob_ext[2].first };
      real_t       ghi[3] = { glob_ext[0].second, glob_ext[1].second,
                              glob_ext[2].second };
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
      const bool on_screen = out::screenBBox(cam, W, H, lo, hi, bx0, by0, bw, bh);

      bool rendered_any = false;
      for (const auto& scene : g_renderer.scenes()) {
        Kokkos::deep_copy(bckp, ZERO);
        if (not prepareRenderScalar(params, *local_domain, scene.field, bckp)) {
          continue;
        }
        // fill the ghost halo with neighbor active values so trilinear
        // sampling is C0 across domain faces (a halo EXCHANGE, not the
        // sum-into-active that SynchronizeFields performs).
        CommunicateBckp(*local_domain, { 0, 1 });

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
        g_renderer.compositeAndWrite(sub, order_key, scene, current_step);
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

      const int  W      = g_renderer.width();
      const int  H      = g_renderer.height();
      const bool mirror = g_renderer.mirror();

      // global slice-plane world window (shared by all ranks -> seamless)
      const auto gext = mesh().extent();
      real_t     umin, umax, vmin, vmax;
      if constexpr (M::CoordType == Coord::type::Cartesian) {
        umin = gext[0].first;
        umax = gext[0].second;
        vmin = gext[1].first;
        vmax = gext[1].second;
      } else {
        // meridional (X = r sin th, Z = r cos th) bounding box of the arc
        const real_t rmax = gext[0].second;
        const real_t th0  = gext[1].first;
        const real_t th1  = gext[1].second;
        umax = rmax;
        umin = mirror ? -rmax : ZERO;
        vmax = rmax * math::cos(th0);
        vmin = rmax * math::cos(th1);
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
          g_renderer.setSlicePolar(true, gext[0].first, gext[0].second,
                                   gext[1].first, gext[1].second, mirror);
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

      bool rendered_any = false;
      for (const auto& scene : g_renderer.scenes()) {
        Kokkos::deep_copy(bckp, ZERO);
        if (not prepareRenderScalar(params, *local_domain, scene.field, bckp)) {
          continue;
        }
        CommunicateBckp(*local_domain, { 0, 1 });

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
                                          n1,
                                          n2,
                                          ext0,
                                          ext1,
                                          scene.tf.lut_opaque,
                                          scene.tf.n_lut,
                                          scene.tf.vmin,
                                          scene.tf.vmax,
                                          scene.tf.log_scale,
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
        g_renderer.compositeAndWrite(sub, order_key, scene, current_step);
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
