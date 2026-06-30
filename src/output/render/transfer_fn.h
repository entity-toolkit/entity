/**
 * @file output/render/transfer_fn.h
 * @brief Colormap tables and premultiplied RGBA look-up-table builder
 * @implements
 *   - out::buildLUT
 *   - out::colormapRGB
 * @namespaces:
 *   - out::
 * @note
 * Colormaps are stored as a handful of anchor colors and linearly
 * interpolated; this is visually indistinguishable from the full 256-entry
 * matplotlib tables for volume rendering while keeping the header compact.
 * The LUT is built on the host and deep-copied to a device View of shape
 * (N_LUT, 4) holding premultiplied RGBA (R=r*a, G=g*a, B=b*a, A=a).
 */

#ifndef OUTPUT_RENDER_TRANSFER_FN_H
#define OUTPUT_RENDER_TRANSFER_FN_H

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include <array>
#include <string>
#include <vector>

namespace out {

  namespace cmap_hidden {

    // anchor colors sampled at uniform positions in [0, 1]
    struct Anchors {
      const float (*rgb)[3];
      int n;
    };

    inline constexpr float viridis[9][3] = {
      { 0.267004f, 0.004874f, 0.329415f },
      { 0.282623f, 0.140926f, 0.457517f },
      { 0.253935f, 0.265254f, 0.529983f },
      { 0.206756f, 0.371758f, 0.553117f },
      { 0.163625f, 0.471133f, 0.558148f },
      { 0.127568f, 0.566949f, 0.550556f },
      { 0.134692f, 0.658636f, 0.517649f },
      { 0.477504f, 0.821444f, 0.318195f },
      { 0.993248f, 0.906157f, 0.143936f },
    };

    inline constexpr float inferno[9][3] = {
      { 0.001462f, 0.000466f, 0.013866f },
      { 0.087411f, 0.044556f, 0.224813f },
      { 0.258234f, 0.038571f, 0.406485f },
      { 0.416331f, 0.090203f, 0.432943f },
      { 0.578304f, 0.148039f, 0.404411f },
      { 0.735683f, 0.215906f, 0.330245f },
      { 0.865006f, 0.316822f, 0.226055f },
      { 0.954506f, 0.468744f, 0.099874f },
      { 0.988362f, 0.998364f, 0.644924f },
    };

    inline constexpr float plasma[9][3] = {
      { 0.050383f, 0.029803f, 0.527975f },
      { 0.287076f, 0.010855f, 0.627295f },
      { 0.417642f, 0.000564f, 0.658390f },
      { 0.562738f, 0.051545f, 0.641509f },
      { 0.692840f, 0.165141f, 0.564522f },
      { 0.798216f, 0.280197f, 0.469538f },
      { 0.881443f, 0.392529f, 0.383229f },
      { 0.949217f, 0.517763f, 0.295662f },
      { 0.940015f, 0.975158f, 0.131326f },
    };

    // Moreland cool-to-warm diverging
    inline constexpr float cool2warm[3][3] = {
      { 0.230f, 0.299f, 0.754f },
      { 0.865f, 0.865f, 0.865f },
      { 0.706f, 0.016f, 0.150f },
    };

    inline constexpr float gray[2][3] = {
      { 0.0f, 0.0f, 0.0f },
      { 1.0f, 1.0f, 1.0f },
    };

    inline auto lookup(const std::string& name) -> Anchors {
      if (name == "inferno") {
        return { inferno, 9 };
      } else if (name == "plasma") {
        return { plasma, 9 };
      } else if (name == "cool2warm" or name == "coolwarm") {
        return { cool2warm, 3 };
      } else if (name == "gray" or name == "grey") {
        return { gray, 2 };
      } else {
        // default / "viridis"
        return { viridis, 9 };
      }
    }

  } // namespace cmap_hidden

  /**
   * @brief Sample a named colormap at u in [0, 1], returning RGB in [0, 1].
   */
  inline void colormapRGB(const std::string& name,
                          real_t             u,
                          real_t&            r,
                          real_t&            g,
                          real_t&            b) {
    const auto anchors = cmap_hidden::lookup(name);
    if (u <= ZERO) {
      r = anchors.rgb[0][0];
      g = anchors.rgb[0][1];
      b = anchors.rgb[0][2];
      return;
    }
    if (u >= ONE) {
      r = anchors.rgb[anchors.n - 1][0];
      g = anchors.rgb[anchors.n - 1][1];
      b = anchors.rgb[anchors.n - 1][2];
      return;
    }
    const real_t x   = u * static_cast<real_t>(anchors.n - 1);
    const int    i0  = static_cast<int>(x);
    const int    i1  = (i0 + 1 < anchors.n) ? (i0 + 1) : i0;
    const real_t t   = x - static_cast<real_t>(i0);
    r = static_cast<real_t>(anchors.rgb[i0][0]) * (ONE - t) +
        static_cast<real_t>(anchors.rgb[i1][0]) * t;
    g = static_cast<real_t>(anchors.rgb[i0][1]) * (ONE - t) +
        static_cast<real_t>(anchors.rgb[i1][1]) * t;
    b = static_cast<real_t>(anchors.rgb[i0][2]) * (ONE - t) +
        static_cast<real_t>(anchors.rgb[i1][2]) * t;
  }

  /**
   * @brief Piecewise-linear opacity from sorted (position, alpha) control points.
   */
  inline auto alphaAt(const std::vector<std::array<real_t, 2>>& pts, real_t u)
    -> real_t {
    if (pts.empty()) {
      return u; // sensible default: linear ramp
    }
    if (u <= pts.front()[0]) {
      return pts.front()[1];
    }
    if (u >= pts.back()[0]) {
      return pts.back()[1];
    }
    for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
      if (u >= pts[i][0] and u <= pts[i + 1][0]) {
        const real_t span = pts[i + 1][0] - pts[i][0];
        const real_t t    = (span > ZERO) ? (u - pts[i][0]) / span : ZERO;
        return pts[i][1] * (ONE - t) + pts[i + 1][1] * t;
      }
    }
    return pts.back()[1];
  }

  /**
   * @brief Build a premultiplied RGBA device LUT from a colormap + alpha points.
   * @param colormap name of the colormap
   * @param n_lut number of entries
   * @param alpha_pts sorted (position, alpha) control points in [0,1]x[0,1]
   * @return device View of shape (n_lut, 4), premultiplied RGBA
   */
  inline auto buildLUT(const std::string&                        colormap,
                       int                                       n_lut,
                       const std::vector<std::array<real_t, 2>>& alpha_pts)
    -> array_t<real_t* [4]> {
    array_t<real_t* [4]> lut { "render_lut", static_cast<std::size_t>(n_lut) };
    auto                 lut_h = Kokkos::create_mirror_view(lut);
    for (int i = 0; i < n_lut; ++i) {
      const real_t u = (n_lut > 1)
                         ? static_cast<real_t>(i) / static_cast<real_t>(n_lut - 1)
                         : ZERO;
      real_t r, g, b;
      colormapRGB(colormap, u, r, g, b);
      const real_t a = alphaAt(alpha_pts, u);
      lut_h(i, 0)    = r * a; // premultiplied
      lut_h(i, 1)    = g * a;
      lut_h(i, 2)    = b * a;
      lut_h(i, 3)    = a;
    }
    Kokkos::deep_copy(lut, lut_h);
    return lut;
  }

} // namespace out

#endif // OUTPUT_RENDER_TRANSFER_FN_H
