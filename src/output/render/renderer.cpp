#include "output/render/renderer.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "output/render/colorbar.h"
#include "output/render/composite.h"
#include "output/render/png.h"
#include "output/render/transfer_fn.h"

#include <toml11/toml.hpp>

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

namespace out {

  namespace {

    inline void cross3(const real_t a[3], const real_t b[3], real_t out[3]) {
      out[0] = a[1] * b[2] - a[2] * b[1];
      out[1] = a[2] * b[0] - a[0] * b[2];
      out[2] = a[0] * b[1] - a[1] * b[0];
    }

    inline auto norm3(const real_t a[3]) -> real_t {
      return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    }

    inline void normalize3(real_t a[3]) {
      const real_t n = norm3(a);
      if (n > static_cast<real_t>(1e-30)) {
        a[0] /= n;
        a[1] /= n;
        a[2] /= n;
      }
    }

    inline auto quantize(real_t v) -> uint8_t {
      const real_t c = (v < ZERO) ? ZERO : ((v > ONE) ? ONE : v);
      return static_cast<uint8_t>(c * static_cast<real_t>(255.0) + HALF);
    }

  } // namespace

  void Renderer::init(const ntt::SimulationParams& params,
                      const boundaries_t<real_t>&  global_extent) {
    m_enabled = false;
    const auto& td = params.data();

    const bool enable = toml::find_or(td, "output", "render", "enable", false);
    if (not enable) {
      return;
    }
    // the renderer is a 3D feature; silently no-op otherwise.
    if (global_extent.size() != 3) {
      raise::Warning("output.render enabled but simulation is not 3D; "
                     "the volume renderer will be inactive",
                     HERE);
      return;
    }

    m_root = path_t(params.get<std::string>("simulation.name"));

    m_width  = toml::find_or<int>(td, "output", "render", "width", 1024);
    m_height = toml::find_or<int>(td, "output", "render", "height", 1024);
    m_samples = toml::find_or<int>(td, "output", "render", "samples", 400);
    m_step_size = toml::find_or<real_t>(td, "output", "render", "step_size", ZERO);
    m_early_alpha = toml::find_or<real_t>(td,
                                          "output",
                                          "render",
                                          "early_term_alpha",
                                          static_cast<real_t>(0.99));
    m_n_lut = toml::find_or<int>(td, "output", "render", "n_lut", 256);

    // opaque background color (shows through low-alpha pixels); default black
    const auto bg = toml::find_or<std::vector<real_t>>(td,
                                                       "output",
                                                       "render",
                                                       "background",
                                                       std::vector<real_t> {});
    if (bg.size() == 3) {
      m_background[0] = bg[0];
      m_background[1] = bg[1];
      m_background[2] = bg[2];
    }

    m_colorbar = toml::find_or<bool>(td, "output", "render", "colorbar", true);
    m_colorbar_outside = toml::find_or<bool>(td,
                                             "output",
                                             "render",
                                             "colorbar_outside",
                                             true);

    // cadence: mirror output.* (interval in steps; interval_time in sim time)
    const auto interval = toml::find_or<timestep_t>(td,
                                                    "output",
                                                    "render",
                                                    "interval",
                                                    0u);
    const auto interval_time = toml::find_or<simtime_t>(td,
                                                        "output",
                                                        "render",
                                                        "interval_time",
                                                        -1.0);
    m_tracker.init("render", interval, interval_time);

    /* ---- camera --------------------------------------------------------- */
    real_t center[3], size[3];
    real_t maxext = ZERO;
    for (int d = 0; d < 3; ++d) {
      center[d] = static_cast<real_t>(0.5) *
                  (global_extent[d].first + global_extent[d].second);
      size[d] = global_extent[d].second - global_extent[d].first;
      maxext  = (size[d] > maxext) ? size[d] : maxext;
    }
    const real_t diag = std::sqrt(size[0] * size[0] + size[1] * size[1] +
                                  size[2] * size[2]);

    const bool ortho = toml::find_or(td,
                                     "output",
                                     "render",
                                     "camera",
                                     "orthographic",
                                     true);
    auto pos = toml::find_or<std::vector<real_t>>(td,
                                                  "output",
                                                  "render",
                                                  "camera",
                                                  "position",
                                                  std::vector<real_t> {});
    auto look = toml::find_or<std::vector<real_t>>(td,
                                                   "output",
                                                   "render",
                                                   "camera",
                                                   "look_at",
                                                   std::vector<real_t> {});
    auto up = toml::find_or<std::vector<real_t>>(td,
                                                 "output",
                                                 "render",
                                                 "camera",
                                                 "up",
                                                 std::vector<real_t> {});
    const real_t fov = toml::find_or<real_t>(td,
                                             "output",
                                             "render",
                                             "camera",
                                             "fov",
                                             static_cast<real_t>(35.0));
    // default covers the box from any view direction (default camera looks
    // down the diagonal), so nothing is clipped without explicit framing.
    (void)maxext;
    const real_t ortho_height = toml::find_or<real_t>(td,
                                                      "output",
                                                      "render",
                                                      "camera",
                                                      "ortho_height",
                                                      diag);

    real_t eye[3], lookat[3], upv[3];
    for (int d = 0; d < 3; ++d) {
      // default eye: box center pushed back along (1,1,1) by ~1.7 diagonals
      eye[d] = (pos.size() == 3)
                 ? pos[d]
                 : center[d] + static_cast<real_t>(1.7) * diag *
                                 static_cast<real_t>(0.57735026919);
      lookat[d] = (look.size() == 3) ? look[d] : center[d];
    }
    upv[0] = (up.size() == 3) ? up[0] : ZERO;
    upv[1] = (up.size() == 3) ? up[1] : ZERO;
    upv[2] = (up.size() == 3) ? up[2] : ONE;

    real_t forward[3] = { lookat[0] - eye[0],
                          lookat[1] - eye[1],
                          lookat[2] - eye[2] };
    normalize3(forward);
    real_t right[3];
    cross3(forward, upv, right);
    normalize3(right);
    real_t up_cam[3];
    cross3(right, forward, up_cam);

    for (int d = 0; d < 3; ++d) {
      m_camera_dev.eye[d]     = eye[d];
      m_camera_dev.forward[d] = forward[d];
      m_camera_dev.right[d]   = right[d];
      m_camera_dev.up[d]      = up_cam[d];
    }
    m_camera_dev.aspect = static_cast<real_t>(m_width) /
                          static_cast<real_t>(m_height);
    m_camera_dev.tan_half_fov = std::tan(static_cast<real_t>(0.5) * fov *
                                         static_cast<real_t>(constant::PI) /
                                         static_cast<real_t>(180.0));
    m_camera_dev.orthographic = ortho;
    m_camera_dev.half_h       = static_cast<real_t>(0.5) * ortho_height;
    m_camera_dev.half_w       = m_camera_dev.half_h * m_camera_dev.aspect;

    /* ---- scenes --------------------------------------------------------- */
    m_scenes.clear();
    const auto scenes_arr = toml::find_or<toml::array>(td,
                                                       "output",
                                                       "render",
                                                       "scenes",
                                                       toml::array {});
    for (const auto& sc : scenes_arr) {
      Scene scene;
      scene.field  = toml::find_or<std::string>(sc, "field", "");
      scene.prefix = toml::find_or<std::string>(sc, "prefix", scene.field + "_");
      if (scene.field.empty()) {
        raise::Warning("output.render scene with no field; skipping", HERE);
        continue;
      }
      scene.label  = toml::find_or<std::string>(sc, "label", scene.field);
      scene.tf.vmin = toml::find_or<real_t>(sc, "min", ZERO);
      scene.tf.vmax = toml::find_or<real_t>(sc, "max", ONE);
      scene.tf.log_scale = toml::find_or<bool>(sc, "log", false);
      scene.tf.n_lut     = m_n_lut;
      const auto colormap = toml::find_or<std::string>(sc, "colormap", "viridis");
      scene.tf.colormap   = colormap;
      // alpha control points: array of [position, alpha] pairs
      const auto alpha_raw = toml::find_or<std::vector<std::vector<real_t>>>(
        sc,
        "alpha",
        std::vector<std::vector<real_t>> {});
      std::vector<std::array<real_t, 2>> alpha_pts;
      for (const auto& p : alpha_raw) {
        if (p.size() >= 2) {
          alpha_pts.push_back({ p[0], p[1] });
        }
      }
      scene.tf.lut = buildLUT(colormap, m_n_lut, alpha_pts);
      m_scenes.push_back(std::move(scene));
    }

    if (m_scenes.empty()) {
      raise::Warning("output.render enabled but no valid scenes; disabling", HERE);
      return;
    }

    m_enabled = true;
    logger::Checkpoint("Volume renderer initialized", HERE);
  }

  void Renderer::compositeAndWrite(const std::vector<real_t>& rgba,
                                   uint64_t                   order_key,
                                   const Scene&               scene,
                                   timestep_t                 step) const {
    const std::size_t npix = static_cast<std::size_t>(m_width) *
                             static_cast<std::size_t>(m_height);
    const std::size_t n = npix * 4;

    auto write_image = [&](const std::vector<real_t>& img) {
      // ensure <name>/renders/ exists
      const auto dir = m_root / path_t("renders");
      try {
        if (not std::filesystem::exists(m_root)) {
          std::filesystem::create_directory(m_root);
        }
        if (not std::filesystem::exists(dir)) {
          std::filesystem::create_directory(dir);
        }
      } catch (const std::exception& e) {
        raise::Warning(e.what(), HERE);
      }
      // composite the premultiplied image over the opaque background:
      // out = src_premult + (1 - src_alpha) * background, alpha = opaque.
      std::vector<uint8_t> bytes(n);
      for (std::size_t p = 0; p < npix; ++p) {
        const real_t a   = img[p * 4 + 3];
        const real_t inv = ONE - a;
        bytes[p * 4 + 0] = quantize(img[p * 4 + 0] + inv * m_background[0]);
        bytes[p * 4 + 1] = quantize(img[p * 4 + 1] + inv * m_background[1]);
        bytes[p * 4 + 2] = quantize(img[p * 4 + 2] + inv * m_background[2]);
        bytes[p * 4 + 3] = 255;
      }
      const auto fname = dir / fmt::format("%s%08lu.png",
                                           scene.prefix.c_str(),
                                           static_cast<unsigned long>(step));
      bool ok = true;
      if (m_colorbar and m_colorbar_outside) {
        // extend the canvas to the right so the colorbar sits in its own margin,
        // outside the rendered volume.
        const int     strip = colorbarBlockWidth(m_height);
        const int     CW    = m_width + strip;
        const uint8_t bR    = quantize(m_background[0]);
        const uint8_t bG    = quantize(m_background[1]);
        const uint8_t bB    = quantize(m_background[2]);
        std::vector<uint8_t> canvas(static_cast<std::size_t>(CW) * m_height * 4);
        for (std::size_t i = 0; i < canvas.size(); i += 4) {
          canvas[i + 0] = bR;
          canvas[i + 1] = bG;
          canvas[i + 2] = bB;
          canvas[i + 3] = 255;
        }
        for (int y = 0; y < m_height; ++y) {
          std::copy_n(&bytes[static_cast<std::size_t>(y) * m_width * 4],
                      static_cast<std::size_t>(m_width) * 4,
                      &canvas[static_cast<std::size_t>(y) * CW * 4]);
        }
        drawColorbar(canvas.data(),
                     CW,
                     m_height,
                     scene.tf.colormap,
                     scene.tf.vmin,
                     scene.tf.vmax,
                     scene.tf.log_scale,
                     scene.label,
                     m_background);
        ok = write_png(fname, CW, m_height, canvas.data());
      } else {
        if (m_colorbar) {
          drawColorbar(bytes.data(),
                       m_width,
                       m_height,
                       scene.tf.colormap,
                       scene.tf.vmin,
                       scene.tf.vmax,
                       scene.tf.log_scale,
                       scene.label,
                       m_background);
        }
        ok = write_png(fname, m_width, m_height, bytes.data());
      }
      if (not ok) {
        raise::Warning(
          fmt::format("failed to write %s", fname.string().c_str()),
          HERE);
      }
    };

#if defined(MPI_ENABLED)
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size == 1) {
      write_image(rgba);
      return;
    }

    std::vector<real_t>            recv;
    std::vector<unsigned long long> keys;
    if (rank == MPI_ROOT_RANK) {
      recv.resize(static_cast<std::size_t>(size) * n);
      keys.resize(static_cast<std::size_t>(size));
    }
    const unsigned long long my_key = static_cast<unsigned long long>(order_key);

    MPI_Gather(rgba.data(),
               static_cast<int>(n),
               mpi::get_type<real_t>(),
               (rank == MPI_ROOT_RANK) ? recv.data() : nullptr,
               static_cast<int>(n),
               mpi::get_type<real_t>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    MPI_Gather(&my_key,
               1,
               MPI_UNSIGNED_LONG_LONG,
               (rank == MPI_ROOT_RANK) ? keys.data() : nullptr,
               1,
               MPI_UNSIGNED_LONG_LONG,
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);

    if (rank != MPI_ROOT_RANK) {
      return;
    }

    // front-to-back order = ranks sorted by ascending composite key
    std::vector<int> order(size);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return keys[a] < keys[b];
    });

    std::vector<real_t> acc(n, ZERO);
    for (const int r : order) {
      const real_t* seg_base = recv.data() + static_cast<std::size_t>(r) * n;
      for (std::size_t p = 0; p < npix; ++p) {
        overComposite(acc.data() + p * 4, seg_base + p * 4);
      }
    }
    write_image(acc);
#else
    (void)order_key;
    write_image(rgba);
#endif
  }

} // namespace out
