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

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cctype>
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
                      ndfield_t<M::Dim, 6>&   buffer,
                      idx_t                   buff_idx) {
      std::vector<spidx_t> specs;
      for (auto& sp : prtl_species) {
        if (sp.mass() > 0) {
          specs.push_back(sp.index());
        }
      }
      auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
      const auto use_weights = params.get<bool>("particles.use_weights");
      const auto ni2         = mesh.n_active(in::x2);
      const auto inv_n0      = ONE / params.get<real_t>("scales.n0");
      const auto smooth_order = params.get<unsigned short>(
        "output.fields.smoothing.order");
      const auto smooth_method = OutputSmoothingType::from_string(
        params.get<std::string>("output.fields.smoothing.method"));
      const std::vector<uint8_t> components {};
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

  } // namespace

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
    if constexpr (M::Dim == Dim::_3D and M::CoordType == Coord::type::Cartesian) {
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
      logger::Checkpoint("Rendering output", HERE);

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

        if (scene.field == "N") {
          renderMoment<S, M, FldsID::N>(params,
                                        local_domain->mesh,
                                        local_domain->species,
                                        bckp,
                                        0u);
          // sum boundary-crossing particle deposits back into active cells
          // (particles in neighbor domains deposit into our ghost zone)
          SynchronizeFields(*local_domain, Comm::Bckp, { 0, 1 });
        } else {
          // Vector field as a scalar: "<base><selector>" with base in {E, B, J}
          // and selector in {mag, 1/2/3, x/y/z}. A component (e.g. "B1"/"Bx") is
          // signed; a magnitude (e.g. "Bmag") is non-negative.
          const std::string& f    = scene.field;
          const char         base = f.empty()
                                      ? '?'
                                      : static_cast<char>(std::toupper(f[0]));
          bool               ok         = true;
          bool               is_current = false;
          uint8_t            src_base   = 0; // first component of the source field
          PrepareOutputFlags interp     = PrepareOutput::None;
          if (base == 'B') {
            src_base = em::bx1;
            interp   = PrepareOutput::InterpToCellCenterFromFaces;
          } else if (base == 'E') {
            src_base = em::ex1;
            interp   = PrepareOutput::InterpToCellCenterFromEdges;
          } else if (base == 'J') {
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
          if (not ok) {
            raise::Warning("output.render: unknown field '" + scene.field +
                             "' (expected N, {E,B,J}mag, or "
                             "{E,B,J}{1,2,3}|{x,y,z}); skipping",
                           HERE);
            continue;
          }
          // raw vector components into bckp(:, 0..2)
          if (is_current) {
            Kokkos::deep_copy(
              Kokkos::subview(bckp, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                              cell_range_t(0, 3)),
              Kokkos::subview(local_domain->fields.cur, Kokkos::ALL, Kokkos::ALL,
                              Kokkos::ALL, cell_range_t(cur::jx1, cur::jx3 + 1)));
          } else {
            Kokkos::deep_copy(
              Kokkos::subview(bckp, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                              cell_range_t(0, 3)),
              Kokkos::subview(local_domain->fields.em, Kokkos::ALL, Kokkos::ALL,
                              Kokkos::ALL, cell_range_t(src_base, src_base + 3)));
          }
          // interpolate to cell centers + convert to physical basis -> (3,4,5)
          const PrepareOutputFlags prepare = (S == SimEngine::SRPIC)
                                               ? PrepareOutput::ConvertToHat
                                               : PrepareOutput::ConvertToPhysCntrv;
          list_t<uint8_t, 3>       comp_from = { 0, 1, 2 };
          list_t<uint8_t, 3>       comp_to   = { 3, 4, 5 };
          Kokkos::parallel_for(
            "RenderFieldsToPhys",
            local_domain->mesh.rangeActiveCells(),
            kernel::FieldsToPhys_kernel<M, 6, 6>(bckp,
                                                 bckp,
                                                 comp_from,
                                                 comp_to,
                                                 interp | prepare,
                                                 metric));
          // reduce to the scalar to render -> bckp(:, 0)
          auto      bckp_v = bckp;
          const int cc     = comp;
          if (comp == -1) {
            Kokkos::parallel_for(
              "RenderVectorMagnitude",
              local_domain->mesh.rangeActiveCells(),
              Lambda(cellidx_t i1, cellidx_t i2, cellidx_t i3) {
                const real_t v1 = bckp_v(i1, i2, i3, 3);
                const real_t v2 = bckp_v(i1, i2, i3, 4);
                const real_t v3 = bckp_v(i1, i2, i3, 5);
                bckp_v(i1, i2, i3, 0) = math::sqrt(v1 * v1 + v2 * v2 + v3 * v3);
              });
          } else {
            Kokkos::parallel_for(
              "RenderVectorComponent",
              local_domain->mesh.rangeActiveCells(),
              Lambda(cellidx_t i1, cellidx_t i2, cellidx_t i3) {
                bckp_v(i1, i2, i3, 0) = bckp_v(i1, i2, i3, 3 + cc);
              });
          }
        }

        // fill the ghost halo with neighbor active values so trilinear
        // sampling is C0 across domain faces (this is a halo EXCHANGE, not the
        // sum-into-active that SynchronizeFields performs).
        CommunicateBckp(*local_domain, { 0, 1 });

        // ---- launch the ray-march kernel over the screen bbox ---------- //
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
    } else {
      (void)params;
      (void)current_step;
      (void)finished_step;
      (void)current_time;
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
                                            simtime_t) -> bool;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_RENDER)

#undef METADOMAIN_RENDER
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
