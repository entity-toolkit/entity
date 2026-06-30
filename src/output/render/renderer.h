/**
 * @file output/render/renderer.h
 * @brief In-situ volume renderer: configuration, cadence, host composite & PNG
 * @implements
 *   - out::Renderer
 *   - out::CameraDevice
 *   - out::TransferFunction
 *   - out::Scene
 * @cpp:
 *   - render/renderer.cpp
 * @namespaces:
 *   - out::
 * @macros:
 *   - MPI_ENABLED
 *   - OUTPUT_ENABLED
 * @note
 * The Renderer is intentionally NOT templated on the engine/metric: it owns
 * only metric-agnostic, host-side work (config parsing, cadence tracking, the
 * MPI ordered composite and PNG encode). The device ray-march kernel and the
 * field preparation live in the templated `Metadomain<S,M>::Render`, mirroring
 * the `out::Writer` (plain) / `Metadomain::Write` (templated) split.
 */

#ifndef OUTPUT_RENDER_RENDERER_H
#define OUTPUT_RENDER_RENDERER_H

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/tools.h"

#include "framework/parameters/parameters.h"

#include <cstdint>
#include <string>
#include <vector>

namespace out {

  /**
   * @brief Device-friendly POD camera + precomputed per-pixel ray basis.
   * @note Trivially copyable; captured by value into the Kokkos kernel.
   */
  struct CameraDevice {
    real_t eye[3] { ZERO, ZERO, ZERO };
    real_t right[3] { ONE, ZERO, ZERO };
    real_t up[3] { ZERO, ONE, ZERO };
    real_t forward[3] { ZERO, ZERO, -ONE };
    real_t tan_half_fov { ONE };
    real_t aspect { ONE };
    bool   orthographic { true };
    real_t half_w { ONE };
    real_t half_h { ONE };
  };

  /**
   * @brief Per-scene transfer function: premultiplied RGBA device LUT + range.
   */
  struct TransferFunction {
    array_t<real_t* [4]> lut;          // device, (n_lut, 4), premultiplied RGBA
    // opaque variant (alpha == 1 everywhere, so the premultiplied entries are
    // straight RGB) used by the flat 2D slice rasterizer, where a single
    // per-pixel sample should paint a solid heatmap rather than fade by opacity.
    array_t<real_t* [4]> lut_opaque;
    int                  n_lut { 256 };
    real_t               vmin { ZERO };
    real_t               vmax { ONE };
    bool                 log_scale { false };
    std::string          colormap { "viridis" }; // for redrawing the colorbar
  };

  /**
   * @brief Configuration for the magnetic-field-line tube overlay.
   * @note The lines are traced once per frame through a coarse, MPI-replicated
   * copy of the (physical-basis) field, so every rank produces the same global
   * polylines and renders only the segments inside its own domain; the existing
   * ordered cross-domain composite then stitches them. The coarsening (`bin`)
   * is what makes the replicate-and-trace cheap and avoids parallel particle
   * advection. See output/render/fieldlines.h.
   */
  struct FieldLineConfig {
    bool        enable { false };     // build the geometry this run
    std::string field { "B" };        // vector field to trace: "B" | "E" | "J"
    int         bin { 4 };            // coarsening factor (cells/coarse cell), 2..8
    real_t      seed_px { 8 };        // seed lattice spacing in screen pixels
    real_t      tube_px { 2 };        // tube radius in screen pixels
    std::string colormap { "inferno" };
    bool        log_scale { false };
    real_t      vmin { ZERO };        // tube color range; vmin>=vmax => auto |B|
    real_t      vmax { ZERO };
    real_t      step_frac { static_cast<real_t>(0.5) }; // RK4 step / coarse cell
    int         max_steps { 4000 };   // per-direction integration cap
    real_t      max_len_frac { static_cast<real_t>(3) }; // x global box diagonal
    int         seed_max { 4096 };    // hard cap on seed count (spacing grows to fit)
  };

  /**
   * @brief Device-side field-line geometry handed to the ray-march kernel.
   * @note A flat capsule list: each row is (p0xyz, p1xyz, s0, s1) with s the
   * per-vertex scalar (|field|) used to color the tube. Opaque (alpha==1) LUT,
   * so a tube sample paints a solid color and is composited inline exactly like
   * the box spine. Empty (n_seg==0) on ranks no line touches.
   */
  struct TubeSet {
    array_t<real_t* [8]> seg;            // (n_seg, 8): p0, p1, s0, s1 in world coords
    int                  n_seg { 0 };
    real_t               radius { ZERO }; // world-space tube radius (ds floor applied)
    array_t<real_t* [4]> lut;             // premultiplied RGBA, opaque (alpha==1)
    int                  n_lut { 256 };
    real_t               vmin { ZERO }, vmax { ONE };
    bool                 log_scale { false };
    std::string          colormap { "inferno" }; // for the standalone colorbar
    // uniform-grid bucket index (CSR) so a ray sample tests only the few
    // segments in its cell instead of all of them. Bucketing on the coarse
    // grid is exact because the tube radius is << one coarse cell; a segment is
    // registered in every cell its radius-padded AABB overlaps.
    array_t<int*>        cell_start;      // (ncell+1) prefix offsets into seg_idx
    array_t<int*>        seg_idx;         // segment indices, grouped by cell
    int                  gnc[3] { 1, 1, 1 };
    real_t               gorigin[3] { ZERO, ZERO, ZERO };
    real_t               gdx[3] { ONE, ONE, ONE };
  };

  /**
   * @brief One rendered scalar field -> one PNG stream.
   * @note `field == "fieldlines"` is a standalone tube scene: no scalar volume
   * is sampled (the field lines render against the background alone). Any other
   * field with `show_fieldlines` true overlays the tubes inside its volume.
   */
  struct Scene {
    std::string         field;     // "N" | "Bmag" | "Vmag" | "Txy" | "B1" | "fieldlines" ...
    std::string         prefix;    // PNG filename prefix, e.g. "Bmag_"
    std::string         label;     // colorbar title (defaults to field)
    std::vector<real_t> ticks;     // explicit colorbar tick values (optional)
    bool                show_fieldlines { false }; // overlay B-field tubes in the volume
    TransferFunction    tf;
  };

  /**
   * @brief A sparse screen-space sub-image: the bounding box of one domain's
   * projected footprint plus its premultiplied RGBA pixels.
   * @note Each domain covers only a small part of the screen, so compositing
   * these sparse boxes (not full frames) is what lets the renderer scale to
   * thousands of ranks.
   */
  struct SubImage {
    int                 x0 { 0 }, y0 { 0 }; // top-left pixel in the full frame
    int                 w { 0 }, h { 0 };   // bbox size in pixels (0 => empty)
    std::vector<real_t> rgba;                // w*h*4 premultiplied, pixel-major
  };

  class Renderer {
  public:
    Renderer() {}

    ~Renderer() = default;

    Renderer(Renderer&&) = default;

    /**
     * @brief Parse `[output.render.*]` and build the camera + per-scene LUTs.
     * @param params simulation parameters (raw toml read via params.data())
     * @param global_extent global physical box, for default camera framing
     */
    void init(const ntt::SimulationParams& params,
              const boundaries_t<real_t>&  global_extent);

    [[nodiscard]]
    auto shouldRender(timestep_t step, simtime_t time) -> bool {
      return m_enabled and m_tracker.shouldWrite(step, time);
    }

    /**
     * @brief Composite the per-rank sparse sub-image across MPI and write PNG.
     * @param sub this rank's sparse screen-space sub-image (premultiplied RGBA)
     * @param order_key this rank's front-to-back sort key (see composite.h)
     * @param scene the scene being written (prefix, colorbar colormap/range/label)
     * @param step current timestep (for the filename cycle number)
     * @note Uses an order-preserving distributed tree reduce; only the MPI root
     *       rank assembles the full frame and writes the file.
     */
    void compositeAndWrite(const SubImage& sub,
                           uint64_t        order_key,
                           const Scene&    scene,
                           timestep_t      step) const;

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto enabled() const -> bool {
      return m_enabled;
    }

    [[nodiscard]]
    auto width() const -> int {
      return m_width;
    }

    [[nodiscard]]
    auto height() const -> int {
      return m_height;
    }

    [[nodiscard]]
    auto samples() const -> int {
      return m_samples;
    }

    [[nodiscard]]
    auto stepSize() const -> real_t {
      return m_step_size;
    }

    [[nodiscard]]
    auto earlyAlpha() const -> real_t {
      return m_early_alpha;
    }

    [[nodiscard]]
    auto camera() const -> const CameraDevice& {
      return m_camera_dev;
    }

    // 2D slice mode: mirror a spherical half-plane across the axis into a full
    // disk (no effect on Cartesian or 3D rendering).
    [[nodiscard]]
    auto mirror() const -> bool {
      return m_mirror;
    }

    [[nodiscard]]
    auto axes() const -> bool {
      return m_axes;
    }

    [[nodiscard]]
    auto background(int i) const -> real_t {
      return m_background[(i < 0) ? 0 : ((i > 2) ? 2 : i)];
    }

    // target 3D spine line width in pixels
    [[nodiscard]]
    auto spineWidth() const -> real_t {
      return m_spine_width;
    }

    // whether `output.render.axis_labels` was set in the toml (so the 2D path
    // honors it instead of substituting per-metric defaults)
    [[nodiscard]]
    auto axisLabelsSet() const -> bool {
      return m_axis_labels_set;
    }

    [[nodiscard]]
    auto axisLabel(int d) const -> const std::string& {
      return m_axis_labels[(d < 0) ? 0 : ((d > 2) ? 2 : d)];
    }

    // Set the world window + axis names the 2D slice path maps onto the image,
    // so the (host) axes overlay can label spatial coordinates. Called by the
    // templated 2D Render before compositing; the window is constant per run.
    void setSliceFrame(real_t             u0,
                       real_t             u1,
                       real_t             v0,
                       real_t             v1,
                       const std::string& xlabel,
                       const std::string& ylabel) {
      m_slice_win[0]  = u0;
      m_slice_win[1]  = u1;
      m_slice_win[2]  = v0;
      m_slice_win[3]  = v1;
      m_slice_xlabel  = xlabel;
      m_slice_ylabel  = ylabel;
    }

    // Mark the 2D slice as curvilinear (spherical) so the axes are drawn polar:
    // a radial "R" axis on the symmetry axis + a "Theta" arc, with a curvilinear
    // spine. Set per-frame by the templated 2D Render (constant per run).
    void setSlicePolar(bool   polar,
                       real_t rmin,
                       real_t rmax,
                       real_t tmin,
                       real_t tmax,
                       bool   mir) {
      m_slice_polar   = polar;
      m_slice_rmin    = rmin;
      m_slice_rmax    = rmax;
      m_slice_tmin    = tmin;
      m_slice_tmax    = tmax;
      m_slice_pmirror = mir;
    }

    [[nodiscard]]
    auto scenes() const -> const std::vector<Scene>& {
      return m_scenes;
    }

    [[nodiscard]]
    auto fieldlines() const -> const FieldLineConfig& {
      return m_fieldlines;
    }

  private:
    bool m_enabled { false };

    int    m_width { 1024 };
    int    m_height { 1024 };
    int    m_samples { 400 };
    real_t m_step_size { ZERO };  // world units/step; 0 => derive from samples
    real_t m_early_alpha { static_cast<real_t>(0.99) };
    int    m_n_lut { 256 };
    // opaque background composited under the final image (shows through
    // low-alpha pixels); defaults to black.
    real_t m_background[3] { ZERO, ZERO, ZERO };
    // draw a colorbar (gradient + value ticks + label) on each PNG
    bool m_colorbar { true };
    // draw the colorbar in an extended right margin (outside the render region)
    // rather than overlaying it on top of the rendered volume
    bool m_colorbar_outside { true };
    // 2D slice mode (spherical): mirror the meridional half-plane across the
    // symmetry axis to render a full disk from one axisymmetric half
    bool m_mirror { true };
    // draw a spine (frame) + axis ticks/labels around the rendered region
    bool        m_axes { false };
    bool        m_axis_labels_set { false };
    int         m_axis_nticks { 5 };
    real_t      m_spine_width { static_cast<real_t>(2) }; // 3D spine width (px)
    std::string m_axis_labels[3] { "x", "y", "z" };
    // global world box (2 or 3 axes); used to project the 3D axes box and to
    // know the render mode (size 2 => 2D slice, size 3 => 3D volume).
    boundaries_t<real_t> m_global_extent;
    // 2D slice world window + axis names, set per-frame by the templated Render
    real_t      m_slice_win[4] { ZERO, ONE, ZERO, ONE };
    std::string m_slice_xlabel { "x" };
    std::string m_slice_ylabel { "y" };
    // 2D curvilinear (spherical) slice: draw polar axes instead of Cartesian
    bool   m_slice_polar { false };
    real_t m_slice_rmin { ZERO }, m_slice_rmax { ONE };
    real_t m_slice_tmin { ZERO }, m_slice_tmax { ONE };
    bool   m_slice_pmirror { false };

    CameraDevice       m_camera_dev;
    std::vector<Scene> m_scenes;
    FieldLineConfig    m_fieldlines;

    tools::Tracker m_tracker;
    path_t         m_root;
  };

} // namespace out

#endif // OUTPUT_RENDER_RENDERER_H
