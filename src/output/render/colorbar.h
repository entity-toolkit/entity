/**
 * @file output/render/colorbar.h
 * @brief Draw a colorbar (gradient strip + value ticks + label) onto an
 *        opaque 8-bit RGBA image, using a self-contained 5x7 bitmap font.
 * @implements
 *   - out::drawColorbar
 * @namespaces:
 *   - out::
 * @note
 * Header-only, host-only. No font dependency: a compact 5x7 ASCII font (digits,
 * sign/exponent symbols, and A-Z) is embedded. Lowercase is mapped to
 * uppercase; unknown glyphs render as blank. Drawn on the MPI root rank after
 * the final composite, so it only ever touches the output buffer.
 */

#ifndef OUTPUT_RENDER_COLORBAR_H
#define OUTPUT_RENDER_COLORBAR_H

#include "global.h"

#include "utils/numeric.h"

#include "output/render/transfer_fn.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace out {

  namespace cbar_hidden {

    // 5x7 glyph: 7 rows, low 5 bits per row, bit 4 = leftmost column.
    inline auto glyph(char c) -> const uint8_t* {
      // map lowercase to uppercase
      if (c >= 'a' and c <= 'z') {
        c = static_cast<char>(c - 'a' + 'A');
      }
      switch (c) {
        case '0': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110 }; return g; }
        case '1': { static const uint8_t g[7] = { 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 }; return g; }
        case '2': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111 }; return g; }
        case '3': { static const uint8_t g[7] = { 0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110 }; return g; }
        case '4': { static const uint8_t g[7] = { 0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010 }; return g; }
        case '5': { static const uint8_t g[7] = { 0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110 }; return g; }
        case '6': { static const uint8_t g[7] = { 0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110 }; return g; }
        case '7': { static const uint8_t g[7] = { 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000 }; return g; }
        case '8': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110 }; return g; }
        case '9': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100 }; return g; }
        case '.': { static const uint8_t g[7] = { 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110 }; return g; }
        case '-': { static const uint8_t g[7] = { 0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000 }; return g; }
        case '+': { static const uint8_t g[7] = { 0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000 }; return g; }
        case '_': { static const uint8_t g[7] = { 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111 }; return g; }
        case ':': { static const uint8_t g[7] = { 0b00000, 0b00110, 0b00110, 0b00000, 0b00110, 0b00110, 0b00000 }; return g; }
        case '/': { static const uint8_t g[7] = { 0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000 }; return g; }
        case 'A': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 }; return g; }
        case 'B': { static const uint8_t g[7] = { 0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110 }; return g; }
        case 'C': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110 }; return g; }
        case 'D': { static const uint8_t g[7] = { 0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100 }; return g; }
        case 'E': { static const uint8_t g[7] = { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111 }; return g; }
        case 'F': { static const uint8_t g[7] = { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000 }; return g; }
        case 'G': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111 }; return g; }
        case 'H': { static const uint8_t g[7] = { 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 }; return g; }
        case 'I': { static const uint8_t g[7] = { 0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 }; return g; }
        case 'J': { static const uint8_t g[7] = { 0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100 }; return g; }
        case 'K': { static const uint8_t g[7] = { 0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001 }; return g; }
        case 'L': { static const uint8_t g[7] = { 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111 }; return g; }
        case 'M': { static const uint8_t g[7] = { 0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001 }; return g; }
        case 'N': { static const uint8_t g[7] = { 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001 }; return g; }
        case 'O': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 }; return g; }
        case 'P': { static const uint8_t g[7] = { 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000 }; return g; }
        case 'Q': { static const uint8_t g[7] = { 0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101 }; return g; }
        case 'R': { static const uint8_t g[7] = { 0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001 }; return g; }
        case 'S': { static const uint8_t g[7] = { 0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110 }; return g; }
        case 'T': { static const uint8_t g[7] = { 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 }; return g; }
        case 'U': { static const uint8_t g[7] = { 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 }; return g; }
        case 'V': { static const uint8_t g[7] = { 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100 }; return g; }
        case 'W': { static const uint8_t g[7] = { 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001 }; return g; }
        case 'X': { static const uint8_t g[7] = { 0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001 }; return g; }
        case 'Y': { static const uint8_t g[7] = { 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100 }; return g; }
        case 'Z': { static const uint8_t g[7] = { 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111 }; return g; }
        default:  { static const uint8_t g[7] = { 0, 0, 0, 0, 0, 0, 0 }; return g; } // blank
      }
    }

    inline void setPx(uint8_t* rgba, int W, int H, int x, int y, uint8_t r,
                      uint8_t g, uint8_t b) {
      if (x < 0 or x >= W or y < 0 or y >= H) {
        return;
      }
      const std::size_t i = (static_cast<std::size_t>(y) * W + x) * 4;
      rgba[i + 0] = r;
      rgba[i + 1] = g;
      rgba[i + 2] = b;
      rgba[i + 3] = 255;
    }

    inline void drawChar(uint8_t* rgba, int W, int H, int x, int y, char c,
                         int s, uint8_t r, uint8_t g, uint8_t b) {
      const uint8_t* gl = glyph(c);
      for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 5; ++col) {
          if (gl[row] & (1u << (4 - col))) {
            for (int dy = 0; dy < s; ++dy) {
              for (int dx = 0; dx < s; ++dx) {
                setPx(rgba, W, H, x + col * s + dx, y + row * s + dy, r, g, b);
              }
            }
          }
        }
      }
    }

    inline void drawText(uint8_t* rgba, int W, int H, int x, int y,
                         const std::string& str, int s, uint8_t r, uint8_t g,
                         uint8_t b) {
      int cx = x;
      for (const char c : str) {
        drawChar(rgba, W, H, cx, y, c, s, r, g, b);
        cx += 6 * s; // 5px glyph + 1px spacing
      }
    }

    inline auto quant(real_t v) -> uint8_t {
      const real_t c = (v < ZERO) ? ZERO : ((v > ONE) ? ONE : v);
      return static_cast<uint8_t>(c * static_cast<real_t>(255.0) + HALF);
    }

    inline auto fmtNum(real_t v) -> std::string {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.3g", static_cast<double>(v));
      return std::string(buf);
    }

    inline auto scale(int H) -> int {
      return std::max(2, H / 400);
    }

  } // namespace cbar_hidden

  /**
   * @brief Pixel width of the colorbar block (bar + gap + labels + padding).
   * @note Depends only on H, so it can size a canvas margin before drawing.
   */
  inline auto colorbarBlockWidth(int H) -> int {
    const int s       = cbar_hidden::scale(H);
    const int char_w  = 6 * s;
    const int bar_w   = std::max(12, H / 50);
    const int gap     = 3 * s;
    const int label_w = 9 * char_w; // room for e.g. "-1.23e+04"
    const int pad     = 4 * s;
    return pad + bar_w + gap + label_w + pad;
  }

  /**
   * @brief Draw a vertical colorbar onto an opaque RGBA buffer.
   * @param rgba    width*height*4 bytes, opaque (alpha forced to 255 on drawn px)
   * @param W,H     image dimensions
   * @param colormap name of the colormap to redraw the gradient
   * @param vmin,vmax value range mapped onto the bar
   * @param log_scale if true, ticks are spaced/labelled logarithmically
   * @param label    title drawn above the bar (e.g. the field name)
   * @param bg       background RGB (to auto-pick contrasting text color)
   */
  inline void drawColorbar(uint8_t*           rgba,
                           int                W,
                           int                H,
                           const std::string& colormap,
                           real_t             vmin,
                           real_t             vmax,
                           bool               log_scale,
                           const std::string& label,
                           const real_t       bg[3]) {
    using namespace cbar_hidden;

    const int s       = scale(H);
    const int char_h  = 8 * s;
    const int bar_w   = std::max(12, H / 50);
    const int bar_h   = H / 2;
    const int gap     = 3 * s;
    const int pad     = 4 * s;
    const int block_w = colorbarBlockWidth(H);

    // place the bar near the right edge of the (possibly extended) canvas
    int bar_x = W - block_w + pad;
    if (bar_x < pad) {
      bar_x = pad;
    }
    const int bar_y = (H - bar_h) / 2;

    // contrasting monochrome for text / frame / ticks
    const real_t  lum = static_cast<real_t>(0.299) * bg[0] +
                       static_cast<real_t>(0.587) * bg[1] +
                       static_cast<real_t>(0.114) * bg[2];
    const uint8_t tc  = (lum < HALF) ? 255 : 0;

    // gradient strip (top = vmax, bottom = vmin)
    for (int j = 0; j < bar_h; ++j) {
      const real_t u = (bar_h > 1)
                         ? ONE - static_cast<real_t>(j) /
                                   static_cast<real_t>(bar_h - 1)
                         : ZERO;
      real_t cr, cg, cb;
      colormapRGB(colormap, u, cr, cg, cb);
      const uint8_t R = quant(cr), G = quant(cg), B = quant(cb);
      for (int i = 0; i < bar_w; ++i) {
        setPx(rgba, W, H, bar_x + i, bar_y + j, R, G, B);
      }
    }

    // frame (thickness s)
    for (int t = 0; t < s; ++t) {
      for (int i = -t; i < bar_w + t; ++i) {
        setPx(rgba, W, H, bar_x + i, bar_y - t, tc, tc, tc);
        setPx(rgba, W, H, bar_x + i, bar_y + bar_h - 1 + t, tc, tc, tc);
      }
      for (int j = -t; j < bar_h + t; ++j) {
        setPx(rgba, W, H, bar_x - t, bar_y + j, tc, tc, tc);
        setPx(rgba, W, H, bar_x + bar_w - 1 + t, bar_y + j, tc, tc, tc);
      }
    }

    // ticks + labels
    const bool   can_log = log_scale and vmin > ZERO and vmax > ZERO;
    const real_t lvmin   = can_log ? math::log10(vmin) : ZERO;
    const real_t lvmax   = can_log ? math::log10(vmax) : ZERO;
    const int    nticks  = 5;
    for (int t = 0; t < nticks; ++t) {
      const real_t u = static_cast<real_t>(t) /
                       static_cast<real_t>(nticks - 1);
      const real_t val = can_log
                           ? math::pow(static_cast<real_t>(10),
                                       lvmin + (lvmax - lvmin) * u)
                           : (vmin + (vmax - vmin) * u);
      const int ty = bar_y +
                     static_cast<int>((ONE - u) *
                                      static_cast<real_t>(bar_h - 1));
      // tick line
      for (int i = 0; i < gap; ++i) {
        for (int w = 0; w < std::max(1, s / 2); ++w) {
          setPx(rgba, W, H, bar_x + bar_w + i, ty + w, tc, tc, tc);
        }
      }
      // label, vertically centered on the tick
      drawText(rgba,
               W,
               H,
               bar_x + bar_w + gap + 2 * s,
               ty - char_h / 2,
               fmtNum(val),
               s,
               tc,
               tc,
               tc);
    }

    // title above the bar, left-aligned to the bar so it stays in the strip
    if (not label.empty()) {
      int ty = bar_y - char_h - 2 * gap;
      if (ty < 0) {
        ty = 0;
      }
      drawText(rgba, W, H, bar_x, ty, label, s, tc, tc, tc);
    }
  }

} // namespace out

#endif // OUTPUT_RENDER_COLORBAR_H
