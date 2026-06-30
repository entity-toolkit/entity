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
    int                  n_lut { 256 };
    real_t               vmin { ZERO };
    real_t               vmax { ONE };
    bool                 log_scale { false };
    std::string          colormap { "viridis" }; // for redrawing the colorbar
  };

  /**
   * @brief One rendered scalar field -> one PNG stream.
   */
  struct Scene {
    std::string      field;        // "N" | "Bmag" | "Jmag" | "smooth_xyz"
    std::string      prefix;       // PNG filename prefix, e.g. "Bmag_"
    std::string      label;        // colorbar title (defaults to field)
    TransferFunction tf;
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

    [[nodiscard]]
    auto scenes() const -> const std::vector<Scene>& {
      return m_scenes;
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

    CameraDevice       m_camera_dev;
    std::vector<Scene> m_scenes;

    tools::Tracker m_tracker;
    path_t         m_root;
  };

} // namespace out

#endif // OUTPUT_RENDER_RENDERER_H
