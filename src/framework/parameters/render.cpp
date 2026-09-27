#include "framework/parameters/render.h"

#include "global.h"

#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/parameters/parameters.h"

#include <toml11/toml.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ntt {
  namespace params {

    namespace {
      // `render.<table>.<key>` if present; otherwise nullopt (used for keys whose
      // defaults depend on the domain geometry and are resolved by the renderer)
      template <typename T>
      auto findOpt(const toml::value& toml_data,
                   const std::string& table,
                   const std::string& key) -> std::optional<T> {
        if (toml_data.contains("render") and
            toml_data.at("render").contains(table) and
            toml_data.at("render").at(table).contains(key)) {
          return toml::find<T>(toml_data, "render", table, key);
        }
        return std::nullopt;
      }
    } // namespace

    void Render::read(const toml::value&            toml_data,
                      const SimulationParams* const params) {
      enable = toml::find_or(toml_data, "render", "enable", false);
      if (not enable) {
        return;
      }

      /* cadence -------------------------------------------------------------- */
      interval = toml::find_or<timestep_t>(toml_data, "render", "interval", 0u);
      interval_time = toml::find_or<simtime_t>(toml_data,
                                               "render",
                                               "interval_time",
                                               -1.0);
      if ((interval.value() == 0) and (interval_time.value() == -1.0)) {
        interval      = params->template get<timestep_t>("output.interval");
        interval_time = params->template get<simtime_t>("output.interval_time");
      }

      /* image ---------------------------------------------------------------- */
      width  = toml::find_or<int>(toml_data, "render", "width", 1024);
      height = toml::find_or<int>(toml_data, "render", "height", 1024);
      // `resolution` is a convenience that forces a square frame (width ==
      // height), the natural shape for a dome master.
      const auto resolution = toml::find_or<int>(toml_data, "render", "resolution", 0);
      if (resolution > 0) {
        width  = resolution;
        height = resolution;
      }
      raise::ErrorIf(width.value() <= 0 or height.value() <= 0,
                     "render.width and render.height must be > 0",
                     HERE);
      n_lut      = toml::find_or<int>(toml_data, "render", "n_lut", 256);
      background = toml::find_or<std::vector<real_t>>(
        toml_data,
        "render",
        "background",
        std::vector<real_t> { ZERO, ZERO, ZERO });
      if (background->size() != 3) {
        raise::Warning("render.background must have 3 entries [r, g, b]; "
                       "using black",
                       HERE);
        background = std::vector<real_t> { ZERO, ZERO, ZERO };
      }
      colorbar = toml::find_or(toml_data, "render", "colorbar", true);
      colorbar_outside = toml::find_or(toml_data, "render", "colorbar_outside", true);
      mirror      = toml::find_or(toml_data, "render", "mirror", true);
      time_label  = toml::find_or(toml_data, "render", "time_label", false);
      axes        = toml::find_or(toml_data, "render", "axes", false);
      // empty => unset (the 2D slice then picks per-metric default names)
      axis_labels = toml::find_or<std::vector<std::string>>(
        toml_data,
        "render",
        "axis_labels",
        std::vector<std::string> {});
      axis_ticks  = toml::find_or<int>(toml_data, "render", "axis_ticks", 5);
      spine_width = toml::find_or<real_t>(toml_data,
                                          "render",
                                          "spine_width",
                                          static_cast<real_t>(2));

      /* [render.extent] ------------------------------------------------------ */
      extent.emplace();
      for (const auto& key : { "x1", "x2", "x3" }) {
        auto lim = toml::find_or<std::vector<real_t>>(toml_data,
                                                      "render",
                                                      "extent",
                                                      key,
                                                      std::vector<real_t> {});
        if (not lim.empty() and (lim.size() != 2 or lim[1] <= lim[0])) {
          raise::Warning("render.extent." + std::string(key) +
                           " must be [lo, hi] with hi > lo; ignoring",
                         HERE);
          lim.clear();
        }
        extent->push_back(lim);
      }

      /* [render.volume] ------------------------------------------------------ */
      volume_samples = toml::find_or<int>(toml_data, "render", "volume", "samples", 400);
      volume_step_size        = toml::find_or<real_t>(toml_data,
                                               "render",
                                               "volume",
                                               "step_size",
                                               ZERO);
      volume_early_term_alpha = toml::find_or<real_t>(toml_data,
                                                      "render",
                                                      "volume",
                                                      "early_term_alpha",
                                                      static_cast<real_t>(0.99));

      /* [render.moving_view] ------------------------------------------------- */
      moving_view_velocity = toml::find_or<std::vector<real_t>>(
        toml_data,
        "render",
        "moving_view",
        "velocity",
        std::vector<real_t> {});
      moving_view_start_time = toml::find_or<simtime_t>(toml_data,
                                                        "render",
                                                        "moving_view",
                                                        "start_time",
                                                        0.0);

      /* [render.camera] ------------------------------------------------------ */
      camera_mode = toml::find_or<std::string>(toml_data,
                                               "render",
                                               "camera",
                                               "mode",
                                               "orthographic");
      if (camera_mode.value() != "orthographic" and
          camera_mode.value() != "perspective" and camera_mode.value() != "dome") {
        raise::Warning(
          "render.camera.mode '" + camera_mode.value() +
            "' unknown (want orthographic/perspective/dome); using "
            "orthographic projection",
          HERE);
        camera_mode = "orthographic";
      }
      camera_position    = toml::find_or<std::vector<real_t>>(toml_data,
                                                           "render",
                                                           "camera",
                                                           "position",
                                                           std::vector<real_t> {});
      camera_look_at     = toml::find_or<std::vector<real_t>>(toml_data,
                                                          "render",
                                                          "camera",
                                                          "look_at",
                                                          std::vector<real_t> {});
      camera_up          = toml::find_or<std::vector<real_t>>(toml_data,
                                                     "render",
                                                     "camera",
                                                     "up",
                                                     std::vector<real_t> {});
      camera_fov         = toml::find_or<real_t>(toml_data,
                                         "render",
                                         "camera",
                                         "fov",
                                         static_cast<real_t>(35));
      camera_dome_fov    = toml::find_or<real_t>(toml_data,
                                              "render",
                                              "camera",
                                              "dome_fov",
                                              static_cast<real_t>(180));
      camera_dome_radius = findOpt<real_t>(toml_data, "camera", "dome_radius");
      camera_ortho_height = findOpt<real_t>(toml_data, "camera", "ortho_height");

      /* [render.dome] -------------------------------------------------------- */
      dome_enable = toml::find_or(toml_data, "render", "dome", "enable", false);
      dome_fov    = toml::find_or<real_t>(toml_data,
                                       "render",
                                       "dome",
                                       "fov",
                                       static_cast<real_t>(180));
      dome_radius = findOpt<real_t>(toml_data, "dome", "radius");
      dome_center = toml::find_or<std::vector<real_t>>(toml_data,
                                                       "render",
                                                       "dome",
                                                       "center",
                                                       std::vector<real_t> {});
      if (not dome_center->empty() and dome_center->size() != 2) {
        raise::Warning("render.dome.center must have 2 entries [x, y]; using "
                       "the domain center",
                       HERE);
        dome_center->clear();
      }
      dome_projection = toml::find_or<std::string>(toml_data,
                                                   "render",
                                                   "dome",
                                                   "projection",
                                                   "equidistant");
      if (dome_projection.value() != "equidistant" and
          dome_projection.value() != "gnomonic" and
          dome_projection.value() != "stereographic" and
          dome_projection.value() != "orthographic") {
        raise::Warning("render.dome.projection '" + dome_projection.value() +
                         "' unknown; using 'equidistant'",
                       HERE);
        dome_projection = "equidistant";
      }

      /* [[render.scene]] ----------------------------------------------------- */
      scenes.emplace();
      bool       any_fieldlines = false;
      const auto scenes_arr     = toml::find_or<toml::array>(toml_data,
                                                         "render",
                                                         "scene",
                                                         toml::array {});
      for (const auto& sc : scenes_arr) {
        RenderScene scene;
        scene.field = toml::find_or<std::string>(sc, "field", "");
        if (scene.field.empty()) {
          raise::Warning("render.scene with no field; skipping", HERE);
          continue;
        }
        scene.prefix = toml::find_or<std::string>(sc, "prefix", scene.field + "_");
        scene.label    = toml::find_or<std::string>(sc, "label", scene.field);
        scene.min      = toml::find_or<real_t>(sc, "min", ZERO);
        scene.max      = toml::find_or<real_t>(sc, "max", ONE);
        scene.log      = toml::find_or<bool>(sc, "log", false);
        scene.colormap = toml::find_or<std::string>(sc, "colormap", "viridis");
        // alpha control points: array of [position, alpha] pairs
        scene.alpha    = toml::find_or<std::vector<std::vector<real_t>>>(
          sc,
          "alpha",
          std::vector<std::vector<real_t>> {});
        scene.colorbar_ticks = toml::find_or<std::vector<real_t>>(
          sc,
          "colorbar_ticks",
          std::vector<real_t> {});
        // overlay the field-line tubes inside this scene's volume; a dedicated
        // `field = "fieldlines"` scene renders the tubes standalone (no volume).
        scene.fieldlines = toml::find_or<bool>(sc, "fieldlines", false) or
                           (scene.field == "fieldlines");
        any_fieldlines = any_fieldlines or scene.fieldlines;
        scenes->push_back(scene);
      }
      if (scenes->empty()) {
        raise::Warning("render enabled but no valid scenes; disabling", HERE);
        enable = false;
        return;
      }

      /* [render.fieldlines] -------------------------------------------------- */
      // the field lines are built whenever the section asks for them OR any
      // scene requests the overlay (so a bare `field = "fieldlines"` scene
      // works without a separate enable flag).
      fieldlines_enable = toml::find_or(toml_data,
                                        "render",
                                        "fieldlines",
                                        "enable",
                                        false) or
                          any_fieldlines;
      fieldlines_field = toml::find_or<std::string>(toml_data,
                                                    "render",
                                                    "fieldlines",
                                                    "field",
                                                    "B");
      fieldlines_bin = toml::find_or<int>(toml_data, "render", "fieldlines", "bin", 4);
      fieldlines_bin      = (fieldlines_bin.value() < 1)
                              ? 1
                              : ((fieldlines_bin.value() > 16)
                                   ? 16
                                   : fieldlines_bin.value());
      fieldlines_seed_px  = toml::find_or<real_t>(toml_data,
                                                 "render",
                                                 "fieldlines",
                                                 "seed_px",
                                                 static_cast<real_t>(8));
      fieldlines_seed_max = toml::find_or<int>(toml_data,
                                               "render",
                                               "fieldlines",
                                               "seed_max",
                                               4096);
      fieldlines_levels   = toml::find_or<int>(toml_data,
                                             "render",
                                             "fieldlines",
                                             "levels",
                                             16);
      fieldlines_tube_px  = toml::find_or<real_t>(toml_data,
                                                 "render",
                                                 "fieldlines",
                                                 "tube_px",
                                                 static_cast<real_t>(2));
      fieldlines_colormap = toml::find_or<std::string>(toml_data,
                                                       "render",
                                                       "fieldlines",
                                                       "colormap",
                                                       "inferno");
      // optional monochrome color [r,g,b]; overrides the colormap when set
      fieldlines_color    = toml::find_or<std::vector<real_t>>(toml_data,
                                                            "render",
                                                            "fieldlines",
                                                            "color",
                                                            std::vector<real_t> {});
      if (not fieldlines_color->empty() and fieldlines_color->size() != 3) {
        raise::Warning("render.fieldlines.color must have 3 entries [r, g, b]; "
                       "ignoring",
                       HERE);
        fieldlines_color->clear();
      }
      fieldlines_log = toml::find_or(toml_data, "render", "fieldlines", "log", false);
      fieldlines_min        = toml::find_or<real_t>(toml_data,
                                             "render",
                                             "fieldlines",
                                             "min",
                                             ZERO);
      fieldlines_max        = toml::find_or<real_t>(toml_data,
                                             "render",
                                             "fieldlines",
                                             "max",
                                             ZERO);
      fieldlines_step_frac  = toml::find_or<real_t>(toml_data,
                                                   "render",
                                                   "fieldlines",
                                                   "step_frac",
                                                   static_cast<real_t>(0.5));
      fieldlines_max_steps  = toml::find_or<int>(toml_data,
                                                "render",
                                                "fieldlines",
                                                "max_steps",
                                                4000);
      fieldlines_max_length = toml::find_or<real_t>(toml_data,
                                                    "render",
                                                    "fieldlines",
                                                    "max_length",
                                                    static_cast<real_t>(3));
    }

    void Render::setParams(SimulationParams* params) const {
      params->set("render.enable", enable);
      if (not enable) {
        return;
      }
      params->set("render.interval", interval.value());
      params->set("render.interval_time", interval_time.value());

      params->set("render.width", width.value());
      params->set("render.height", height.value());
      params->set("render.n_lut", n_lut.value());
      params->set("render.background", background.value());
      params->set("render.colorbar", colorbar.value());
      params->set("render.colorbar_outside", colorbar_outside.value());
      params->set("render.mirror", mirror.value());
      params->set("render.time_label", time_label.value());
      params->set("render.axes", axes.value());
      params->set("render.axis_labels", axis_labels.value());
      params->set("render.axis_ticks", axis_ticks.value());
      params->set("render.spine_width", spine_width.value());

      params->set("render.extent.x1", extent.value()[0]);
      params->set("render.extent.x2", extent.value()[1]);
      params->set("render.extent.x3", extent.value()[2]);

      params->set("render.volume.samples", volume_samples.value());
      params->set("render.volume.step_size", volume_step_size.value());
      params->set("render.volume.early_term_alpha",
                  volume_early_term_alpha.value());

      params->set("render.moving_view.velocity", moving_view_velocity.value());
      params->set("render.moving_view.start_time", moving_view_start_time.value());

      params->set("render.camera.mode", camera_mode.value());
      params->set("render.camera.position", camera_position.value());
      params->set("render.camera.look_at", camera_look_at.value());
      params->set("render.camera.up", camera_up.value());
      params->set("render.camera.fov", camera_fov.value());
      params->set("render.camera.dome_fov", camera_dome_fov.value());
      if (camera_dome_radius.has_value()) {
        params->set("render.camera.dome_radius", camera_dome_radius.value());
      }
      if (camera_ortho_height.has_value()) {
        params->set("render.camera.ortho_height", camera_ortho_height.value());
      }

      params->set("render.dome.enable", dome_enable.value());
      params->set("render.dome.fov", dome_fov.value());
      if (dome_radius.has_value()) {
        params->set("render.dome.radius", dome_radius.value());
      }
      params->set("render.dome.center", dome_center.value());
      params->set("render.dome.projection", dome_projection.value());

      params->set("render.fieldlines.enable", fieldlines_enable.value());
      params->set("render.fieldlines.field", fieldlines_field.value());
      params->set("render.fieldlines.bin", fieldlines_bin.value());
      params->set("render.fieldlines.seed_px", fieldlines_seed_px.value());
      params->set("render.fieldlines.seed_max", fieldlines_seed_max.value());
      params->set("render.fieldlines.levels", fieldlines_levels.value());
      params->set("render.fieldlines.tube_px", fieldlines_tube_px.value());
      params->set("render.fieldlines.colormap", fieldlines_colormap.value());
      params->set("render.fieldlines.color", fieldlines_color.value());
      params->set("render.fieldlines.log", fieldlines_log.value());
      params->set("render.fieldlines.min", fieldlines_min.value());
      params->set("render.fieldlines.max", fieldlines_max.value());
      params->set("render.fieldlines.step_frac", fieldlines_step_frac.value());
      params->set("render.fieldlines.max_steps", fieldlines_max_steps.value());
      params->set("render.fieldlines.max_length", fieldlines_max_length.value());

      // scenes are flattened into indexed keys (`render.scene.<i>.<key>`) so
      // that every entry stays a plain (serializable) parameter type
      params->set("render.nscenes", scenes->size());
      for (std::size_t i = 0; i < scenes->size(); ++i) {
        const auto& sc  = scenes.value()[i];
        const auto  pfx = "render.scene." + std::to_string(i) + ".";
        params->set(pfx + "field", sc.field);
        params->set(pfx + "prefix", sc.prefix);
        params->set(pfx + "label", sc.label);
        params->set(pfx + "min", sc.min);
        params->set(pfx + "max", sc.max);
        params->set(pfx + "log", sc.log);
        params->set(pfx + "colormap", sc.colormap);
        params->set(pfx + "alpha", sc.alpha);
        params->set(pfx + "colorbar_ticks", sc.colorbar_ticks);
        params->set(pfx + "fieldlines", sc.fieldlines);
      }
    }

  } // namespace params
} // namespace ntt
