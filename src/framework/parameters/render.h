/**
 * @file framework/parameters/render.h
 * @brief Auxiliary functions for reading in on-the-fly render parameters
 * @implements
 *   - ntt::params::RenderScene
 *   - ntt::params::Render
 * @cpp:
 *   - render.cpp
 * @namespaces:
 *   - ntt::params::
 * @note Only the raw (geometry-independent) configuration is resolved here;
 * defaults that depend on the domain extent (camera framing, dome radius,
 * clamping of the render region to the box) are resolved by out::Renderer.
 * Those keys are only set in SimulationParams when given in the input.
 */
#ifndef FRAMEWORK_PARAMETERS_RENDER_H
#define FRAMEWORK_PARAMETERS_RENDER_H

#include "global.h"

#include "framework/parameters/parameters.h"

#include <toml11/toml.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ntt {
  namespace params {

    struct RenderScene {
      std::string                      field;
      std::string                      prefix;
      std::string                      label;
      real_t                           min;
      real_t                           max;
      bool                             log;
      std::string                      colormap;
      std::vector<std::vector<real_t>> alpha;
      std::vector<real_t>              colorbar_ticks;
      bool                             fieldlines;
    };

    struct Render {
      bool enable { false };

      std::optional<timestep_t> interval;
      std::optional<simtime_t>  interval_time;

      std::optional<int>                      width;
      std::optional<int>                      height;
      std::optional<int>                      n_lut;
      std::optional<std::vector<real_t>>      background;
      std::optional<bool>                     colorbar;
      std::optional<bool>                     colorbar_outside;
      std::optional<bool>                     mirror;
      std::optional<bool>                     time_label;
      std::optional<bool>                     axes;
      std::optional<std::vector<std::string>> axis_labels;
      std::optional<int>                      axis_ticks;
      std::optional<real_t>                   spine_width;

      // [render.extent]: x{1,2,3} -> [lo, hi] or empty (full extent)
      std::optional<std::vector<std::vector<real_t>>> extent;

      // [render.volume]
      std::optional<int>    volume_samples;
      std::optional<real_t> volume_step_size;
      std::optional<real_t> volume_early_term_alpha;

      // [render.moving_view]
      std::optional<std::vector<real_t>> moving_view_velocity;
      std::optional<simtime_t>           moving_view_start_time;

      // [render.camera]
      std::optional<std::string>         camera_mode;
      std::optional<std::vector<real_t>> camera_position;
      std::optional<std::vector<real_t>> camera_look_at;
      std::optional<std::vector<real_t>> camera_up;
      std::optional<real_t>              camera_fov;
      std::optional<real_t>              camera_dome_fov;
      std::optional<real_t>              camera_dome_radius;
      std::optional<real_t>              camera_ortho_height;

      // [render.dome]
      std::optional<bool>                dome_enable;
      std::optional<real_t>              dome_fov;
      std::optional<real_t>              dome_radius;
      std::optional<std::vector<real_t>> dome_center;
      std::optional<std::string>         dome_projection;

      // [render.fieldlines]
      std::optional<bool>                fieldlines_enable;
      std::optional<std::string>         fieldlines_field;
      std::optional<int>                 fieldlines_bin;
      std::optional<real_t>              fieldlines_seed_px;
      std::optional<int>                 fieldlines_seed_max;
      std::optional<int>                 fieldlines_levels;
      std::optional<real_t>              fieldlines_tube_px;
      std::optional<std::string>         fieldlines_colormap;
      std::optional<std::vector<real_t>> fieldlines_color;
      std::optional<bool>                fieldlines_log;
      std::optional<real_t>              fieldlines_min;
      std::optional<real_t>              fieldlines_max;
      std::optional<real_t>              fieldlines_step_frac;
      std::optional<int>                 fieldlines_max_steps;
      std::optional<real_t>              fieldlines_max_length;

      // [[render.scene]]
      std::optional<std::vector<RenderScene>> scenes;

      void read(const toml::value&, const SimulationParams* const);
      void setParams(SimulationParams*) const;
    };

  } // namespace params
} // namespace ntt

#endif // FRAMEWORK_PARAMETERS_RENDER_H
