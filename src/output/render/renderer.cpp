#include "output/render/renderer.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "output/render/axes.h"
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
    // 2D (slice rasterizer) and 3D Cartesian (volume ray-march) are supported;
    // 1D has nothing to render.
    if (global_extent.size() != 2 and global_extent.size() != 3) {
      raise::Warning("output.render enabled but simulation is 1D; "
                     "the renderer will be inactive",
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
    // 2D slice mode (spherical only): mirror the half-plane into a full disk
    m_mirror = toml::find_or<bool>(td, "output", "render", "mirror", true);

    // axes: spine + ticks + labels around the rendered region
    m_axes = toml::find_or<bool>(td, "output", "render", "axes", false);
    m_axis_nticks = toml::find_or<int>(td, "output", "render", "axis_ticks", 5);
    m_spine_width = toml::find_or<real_t>(td, "output", "render", "spine_width",
                                          static_cast<real_t>(2));
    m_global_extent = global_extent;
    {
      const auto al = toml::find_or<std::vector<std::string>>(
        td, "output", "render", "axis_labels", std::vector<std::string> {});
      m_axis_labels_set = not al.empty();
      for (std::size_t d = 0; d < al.size() and d < 3; ++d) {
        m_axis_labels[d] = al[d];
      }
      // default 2D slice names track the labels (overridden per-metric by Render)
      m_slice_xlabel = m_axis_labels[0];
      m_slice_ylabel = m_axis_labels[1];
    }

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

    /* ---- camera (used by the 3D volume mode; the 2D slice path frames itself
     * and ignores this, so a missing 3rd axis is zero-filled harmlessly) ---- */
    real_t center[3] = { ZERO, ZERO, ZERO }, size[3] = { ZERO, ZERO, ZERO };
    real_t maxext = ZERO;
    for (std::size_t d = 0; d < global_extent.size() and d < 3; ++d) {
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
      scene.ticks  = toml::find_or<std::vector<real_t>>(sc,
                                                        "colorbar_ticks",
                                                        std::vector<real_t> {});
      // overlay the B-field-line tubes inside this scene's volume; a dedicated
      // `field = "fieldlines"` scene renders the tubes standalone (no volume).
      scene.show_fieldlines = toml::find_or<bool>(sc, "fieldlines", false) or
                              (scene.field == "fieldlines");
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
      // opaque companion LUT (alpha == 1) for the flat 2D slice rasterizer
      scene.tf.lut_opaque = buildLUT(colormap,
                                     m_n_lut,
                                     { { ZERO, ONE }, { ONE, ONE } });
      m_scenes.push_back(std::move(scene));
    }

    if (m_scenes.empty()) {
      raise::Warning("output.render enabled but no valid scenes; disabling", HERE);
      return;
    }

    /* ---- magnetic-field-line tube overlay ------------------------------- */
    // The tubes are built whenever the [output.render.fieldlines] section asks
    // for them OR any scene requests the overlay (so a bare `field =
    // "fieldlines"` scene works without a separate enable flag).
    bool any_fl = false;
    for (const auto& s : m_scenes) {
      any_fl = any_fl or s.show_fieldlines;
    }
    m_fieldlines.enable = toml::find_or<bool>(td, "output", "render",
                                              "fieldlines", "enable", false) or
                          any_fl;
    if (m_fieldlines.enable) {
      if (m_global_extent.size() != 3) {
        raise::Warning("output.render.fieldlines is 3D-only; ignoring", HERE);
        m_fieldlines.enable = false;
      } else {
        auto& fl   = m_fieldlines;
        fl.field   = toml::find_or<std::string>(td, "output", "render",
                                                "fieldlines", "field", "B");
        fl.bin     = toml::find_or<int>(td, "output", "render", "fieldlines",
                                        "bin", 4);
        fl.bin     = (fl.bin < 1) ? 1 : ((fl.bin > 16) ? 16 : fl.bin);
        fl.seed_px = toml::find_or<real_t>(td, "output", "render", "fieldlines",
                                           "seed_px", static_cast<real_t>(8));
        fl.tube_px = toml::find_or<real_t>(td, "output", "render", "fieldlines",
                                           "tube_px", static_cast<real_t>(2));
        fl.colormap = toml::find_or<std::string>(td, "output", "render",
                                                 "fieldlines", "colormap",
                                                 "inferno");
        fl.log_scale = toml::find_or<bool>(td, "output", "render", "fieldlines",
                                           "log", false);
        fl.vmin = toml::find_or<real_t>(td, "output", "render", "fieldlines",
                                        "min", ZERO);
        fl.vmax = toml::find_or<real_t>(td, "output", "render", "fieldlines",
                                        "max", ZERO);
        fl.step_frac = toml::find_or<real_t>(td, "output", "render", "fieldlines",
                                             "step_frac", static_cast<real_t>(0.5));
        fl.max_steps = toml::find_or<int>(td, "output", "render", "fieldlines",
                                          "max_steps", 4000);
        fl.max_len_frac = toml::find_or<real_t>(td, "output", "render",
                                                "fieldlines", "max_length",
                                                static_cast<real_t>(3));
        fl.seed_max = toml::find_or<int>(td, "output", "render", "fieldlines",
                                         "seed_max", 4096);
      }
    }

    m_enabled = true;
    logger::Checkpoint("In-situ renderer initialized", HERE);
  }

  void Renderer::compositeAndWrite(const SubImage& sub,
                                   uint64_t        order_key,
                                   const Scene&    scene,
                                   timestep_t      step) const {
    const std::size_t npix = static_cast<std::size_t>(m_width) *
                             static_cast<std::size_t>(m_height);
    const std::size_t n = npix * 4;

    // expand a sparse sub-image into a full transparent frame (premultiplied)
    auto subToFull = [&](const SubImage& s) -> std::vector<real_t> {
      std::vector<real_t> full(n, ZERO);
      for (int y = 0; y < s.h; ++y) {
        for (int x = 0; x < s.w; ++x) {
          const int fx = s.x0 + x;
          const int fy = s.y0 + y;
          if (fx < 0 or fx >= m_width or fy < 0 or fy >= m_height) {
            continue;
          }
          const std::size_t fi = (static_cast<std::size_t>(fy) * m_width + fx) * 4;
          const std::size_t si = (static_cast<std::size_t>(y) * s.w + x) * 4;
          full[fi + 0] = s.rgba[si + 0];
          full[fi + 1] = s.rgba[si + 1];
          full[fi + 2] = s.rgba[si + 2];
          full[fi + 3] = s.rgba[si + 3];
        }
      }
      return full;
    };

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
      std::vector<uint8_t> data(n);
      for (std::size_t p = 0; p < npix; ++p) {
        const real_t a   = img[p * 4 + 3];
        const real_t inv = ONE - a;
        data[p * 4 + 0] = quantize(img[p * 4 + 0] + inv * m_background[0]);
        data[p * 4 + 1] = quantize(img[p * 4 + 1] + inv * m_background[1]);
        data[p * 4 + 2] = quantize(img[p * 4 + 2] + inv * m_background[2]);
        data[p * 4 + 3] = 255;
      }
      const auto fname = dir / fmt::format("%s%08lu.png",
                                           scene.prefix.c_str(),
                                           static_cast<unsigned long>(step));

      auto drawBar = [&](uint8_t* buf, int bw, int bh) {
        if (m_colorbar) {
          drawColorbar(buf, bw, bh, scene.tf.colormap, scene.tf.vmin,
                       scene.tf.vmax, scene.tf.log_scale, scene.label,
                       m_background, scene.ticks);
        }
      };

      // canvas margins: axes (left + bottom) and the colorbar strip (right).
      // The data region sits at (ml, 0); margins/strip are background-filled.
      // The polar (curvilinear) overlay annotates inside the data region (the
      // disk is centered with background around it), so it needs no margins.
      const bool polar = (m_global_extent.size() == 2) and m_slice_polar;
      int        ml = 0, mb = 0;
      out::axesMargins(m_axes and not polar, m_height, ml, mb);
      const int strip = (m_colorbar and m_colorbar_outside)
                          ? colorbarBlockWidth(m_height)
                          : 0;
      const int CW = ml + m_width + strip;
      const int CH = m_height + mb;

      bool ok = true;
      if (CW == m_width and CH == m_height and not m_axes) {
        // no margins, no outside strip, no overlay: colorbar overlays the data
        drawBar(data.data(), m_width, m_height);
        ok = write_png(fname, m_width, m_height, data.data());
      } else {
        const uint8_t bR = quantize(m_background[0]);
        const uint8_t bG = quantize(m_background[1]);
        const uint8_t bB = quantize(m_background[2]);
        std::vector<uint8_t> canvas(static_cast<std::size_t>(CW) * CH * 4);
        for (std::size_t i = 0; i < canvas.size(); i += 4) {
          canvas[i + 0] = bR;
          canvas[i + 1] = bG;
          canvas[i + 2] = bB;
          canvas[i + 3] = 255;
        }
        for (int y = 0; y < m_height; ++y) {
          std::copy_n(
            &data[static_cast<std::size_t>(y) * m_width * 4],
            static_cast<std::size_t>(m_width) * 4,
            &canvas[(static_cast<std::size_t>(y) * CW + ml) * 4]);
        }
        if (m_axes) {
          if (m_global_extent.size() == 3) {
            out::drawAxes3D(canvas.data(), CW, CH, ml, m_width, m_height,
                            m_camera_dev, m_global_extent, m_axis_labels,
                            m_background, m_axis_nticks);
          } else if (polar) {
            out::drawAxesPolar(canvas.data(), CW, CH, ml, m_width, m_height,
                               m_slice_win[0], m_slice_win[1], m_slice_win[2],
                               m_slice_win[3], m_slice_rmin, m_slice_rmax,
                               m_slice_tmin, m_slice_tmax, m_slice_pmirror, "R",
                               "Theta", m_background, m_axis_nticks);
          } else {
            out::drawAxes2D(canvas.data(), CW, CH, ml, m_width, m_height,
                            m_slice_win[0], m_slice_win[1], m_slice_win[2],
                            m_slice_win[3], m_slice_xlabel, m_slice_ylabel,
                            m_background, m_axis_nticks);
          }
        }
        drawBar(canvas.data(), CW, CH);
        ok = write_png(fname, CW, CH, canvas.data());
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
      write_image(subToFull(sub));
      return;
    }

    constexpr int TAG_HDR  = 7301;
    constexpr int TAG_DATA = 7302;

    // Wire format is premultiplied uint8 RGBA (4x less bandwidth than float).
    // Compositing stays in float; only the per-message quantization adds error
    // (~1 LSB through the log(N)-deep tree), so fidelity is effectively that of
    // the final 8-bit PNG.
    auto sendSub = [&](const SubImage& s, int dest) {
      int hdr[4] = { s.x0, s.y0, s.w, s.h };
      MPI_Send(hdr, 4, MPI_INT, dest, TAG_HDR, MPI_COMM_WORLD);
      const int cnt = s.w * s.h * 4;
      if (cnt > 0) {
        std::vector<uint8_t> bytes(static_cast<std::size_t>(cnt));
        for (int i = 0; i < cnt; ++i) {
          bytes[i] = quantize(s.rgba[i]);
        }
        MPI_Send(bytes.data(), cnt, MPI_UNSIGNED_CHAR, dest, TAG_DATA, MPI_COMM_WORLD);
      }
    };
    auto recvSub = [&](int src) -> SubImage {
      int hdr[4];
      MPI_Recv(hdr, 4, MPI_INT, src, TAG_HDR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      SubImage s;
      s.x0          = hdr[0];
      s.y0          = hdr[1];
      s.w           = hdr[2];
      s.h           = hdr[3];
      const int cnt = s.w * s.h * 4;
      if (cnt > 0) {
        std::vector<uint8_t> bytes(static_cast<std::size_t>(cnt));
        MPI_Recv(bytes.data(),
                 cnt,
                 MPI_UNSIGNED_CHAR,
                 src,
                 TAG_DATA,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        s.rgba.resize(static_cast<std::size_t>(cnt));
        const real_t inv255 = ONE / static_cast<real_t>(255);
        for (int i = 0; i < cnt; ++i) {
          s.rgba[i] = static_cast<real_t>(bytes[i]) * inv255;
        }
      }
      return s;
    };

    // Every rank learns the full key vector (one uint64 each: ~tiny) and
    // derives the same global front-to-back order, so no rank needs the others'
    // images to agree on the composite order.
    const unsigned long long my_key = static_cast<unsigned long long>(order_key);
    std::vector<unsigned long long> keys(static_cast<std::size_t>(size));
    MPI_Allgather(&my_key,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  keys.data(),
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_COMM_WORLD);
    std::vector<int> order(size); // order[position] = world rank, front-to-back
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return keys[a] < keys[b];
    });
    std::vector<int> pos(size); // pos[world rank] = front-to-back position
    for (int i = 0; i < size; ++i) {
      pos[order[i]] = i;
    }

    // Order-preserving binary tree reduction over positions. At level `s`, the
    // front of each pair (lower position) receives the back partner's image and
    // composites front OVER back; the back partner sends and drops out. "over"
    // is associative, so this reproduces the sequential front-to-back composite
    // in O(log nranks) rounds with no single-rank bottleneck.
    SubImage  cur = sub;
    const int P   = pos[rank];
    for (int s = 1; s < size; s <<= 1) {
      if ((P % (2 * s)) == 0) {
        const int pp = P + s;
        if (pp < size) {
          const SubImage back = recvSub(order[pp]);
          cur                 = overSub(cur, back); // cur is the front
        }
      } else if ((P % (2 * s)) == s) {
        sendSub(cur, order[P - s]);
        break; // absorbed into the front partner
      }
    }

    // The fully composited image now lives at position 0; deliver it to root.
    if (rank == order[0]) {
      if (rank == MPI_ROOT_RANK) {
        write_image(subToFull(cur));
      } else {
        sendSub(cur, MPI_ROOT_RANK);
      }
    } else if (rank == MPI_ROOT_RANK) {
      write_image(subToFull(recvSub(order[0])));
    }
#else
    (void)order_key;
    write_image(subToFull(sub));
#endif
  }

} // namespace out
