/**
 * @file output/render/axes.h
 * @brief Draw a spine (frame), axis ticks and labels onto the opaque RGBA
 *        canvas, for both the 2D slice and the 3D volume renders.
 * @implements
 *   - out::axesMargins
 *   - out::drawAxes2D
 *   - out::drawAxes3D
 * @namespaces:
 *   - out::
 * @note
 * Header-only, host-only, drawn on the MPI root rank after compositing (like the
 * colorbar). Reuses the 5x7 bitmap font and helpers from colorbar.h.
 *   - 2D: the data region maps affinely to a world window [u0,u1]x[v0,v1]; a
 *     rectangular spine is drawn around it with linear ticks/labels in the
 *     surrounding margins (so they never overlap the data).
 *   - 3D: the global box is projected with the ray-march camera into a wireframe
 *     "spine"; ticks + labels are placed along the three edges emanating from the
 *     bottom-most projected corner, marks pushed outward from the box centroid.
 */

#ifndef OUTPUT_RENDER_AXES_H
#define OUTPUT_RENDER_AXES_H

#include "global.h"

#include "output/render/colorbar.h"  // glyph, scale, fmtNum
#include "output/render/composite.h" // projectToScreen, CameraDevice

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace out {

  namespace axes_hidden {

    // set one opaque pixel, clipped to the full canvas [0,CW)x[0,CH)
    inline void px(uint8_t* b, int CW, int CH, int x, int y, uint8_t c) {
      if (x < 0 or x >= CW or y < 0 or y >= CH) {
        return;
      }
      const std::size_t i = (static_cast<std::size_t>(y) * CW + x) * 4;
      b[i + 0] = c;
      b[i + 1] = c;
      b[i + 2] = c;
      b[i + 3] = 255;
    }

    inline void thickPx(uint8_t* b, int CW, int CH, int x, int y, int t, uint8_t c) {
      for (int dy = -t; dy <= t; ++dy) {
        for (int dx = -t; dx <= t; ++dx) {
          px(b, CW, CH, x + dx, y + dy, c);
        }
      }
    }

    // Bresenham line, thickness (2t+1)
    inline void line(uint8_t* b, int CW, int CH, int x0, int y0, int x1, int y1,
                     int t, uint8_t c) {
      int dx = std::abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;
      int dy = -std::abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;
      int err = dx + dy;
      while (true) {
        thickPx(b, CW, CH, x0, y0, t, c);
        if (x0 == x1 and y0 == y1) {
          break;
        }
        const int e2 = 2 * err;
        if (e2 >= dy) {
          err += dy;
          x0  += sx;
        }
        if (e2 <= dx) {
          err += dx;
          y0  += sy;
        }
      }
    }

    inline void text(uint8_t* b, int CW, int CH, int x, int y,
                     const std::string& str, int s, uint8_t c) {
      int cx = x;
      for (const char ch : str) {
        const uint8_t* gl = cbar_hidden::glyph(ch);
        for (int row = 0; row < 7; ++row) {
          for (int col = 0; col < 5; ++col) {
            if (gl[row] & (1u << (4 - col))) {
              for (int dy = 0; dy < s; ++dy) {
                for (int dx = 0; dx < s; ++dx) {
                  px(b, CW, CH, cx + col * s + dx, y + row * s + dy, c);
                }
              }
            }
          }
        }
        cx += 6 * s;
      }
    }

    // Rotated bitmap text: the baseline advances along unit (ax, ay); (ox, oy)
    // is the text-local origin (top-left of the first glyph). Each glyph cell is
    // oversampled 2x so rotation leaves no gaps.
    inline void textRot(uint8_t* b, int CW, int CH, real_t ox, real_t oy,
                        const std::string& str, int s, real_t ax, real_t ay,
                        uint8_t c) {
      const real_t dnx = -ay, dny = ax; // glyph "down" (perp. to baseline)
      for (std::size_t ci = 0; ci < str.size(); ++ci) {
        const uint8_t* gl   = cbar_hidden::glyph(str[ci]);
        const real_t   base = static_cast<real_t>(ci) * 6 * s;
        for (int row = 0; row < 7; ++row) {
          for (int col = 0; col < 5; ++col) {
            if (not(gl[row] & (1u << (4 - col)))) {
              continue;
            }
            for (int sy = 0; sy < 2 * s; ++sy) {
              for (int sx = 0; sx < 2 * s; ++sx) {
                const real_t u = base + col * s + static_cast<real_t>(sx) * HALF;
                const real_t v = row * s + static_cast<real_t>(sy) * HALF;
                px(b, CW, CH,
                   static_cast<int>(std::lround(ox + u * ax + v * dnx)),
                   static_cast<int>(std::lround(oy + u * ay + v * dny)), c);
              }
            }
          }
        }
      }
    }

    // rotated text centered on (cxp, cyp), baseline along unit (ax, ay)
    inline void textRotCentered(uint8_t* b, int CW, int CH, real_t cxp,
                                real_t cyp, const std::string& str, int s,
                                real_t ax, real_t ay, uint8_t c) {
      const real_t dnx = -ay, dny = ax;
      const real_t w   = static_cast<real_t>(str.size()) * 6 * s;
      const real_t h   = 7 * s;
      textRot(b, CW, CH, cxp - HALF * w * ax - HALF * h * dnx,
              cyp - HALF * w * ay - HALF * h * dny, str, s, ax, ay, c);
    }

    // orient a screen-space edge direction so text reads naturally (rightward
    // for near-horizontal edges, upward for near-vertical ones)
    inline void readableDir(real_t& ax, real_t& ay) {
      const real_t n = std::sqrt(static_cast<double>(ax * ax + ay * ay));
      if (n < static_cast<real_t>(1e-9)) {
        ax = ONE;
        ay = ZERO;
        return;
      }
      ax /= n;
      ay /= n;
      if (std::fabs(static_cast<double>(ax)) >= std::fabs(static_cast<double>(ay))) {
        if (ax < ZERO) {
          ax = -ax;
          ay = -ay;
        }
      } else if (ay > ZERO) {
        ax = -ax;
        ay = -ay;
      }
    }

    // vertical stack of characters (top to bottom), used for the y-axis name
    inline void textVert(uint8_t* b, int CW, int CH, int x, int y,
                         const std::string& str, int s, uint8_t c) {
      int cy = y;
      for (const char ch : str) {
        const std::string one(1, ch);
        text(b, CW, CH, x, cy, one, s, c);
        cy += 8 * s;
      }
    }

    inline auto textW(const std::string& str, int s) -> int {
      return static_cast<int>(str.size()) * 6 * s;
    }

    inline auto contrast(const real_t bg[3]) -> uint8_t {
      const real_t lum = static_cast<real_t>(0.299) * bg[0] +
                         static_cast<real_t>(0.587) * bg[1] +
                         static_cast<real_t>(0.114) * bg[2];
      return (lum < HALF) ? 255 : 0;
    }

    // a "nice" number close to x (1/2/5 x 10^k)
    inline auto niceNum(real_t x, bool round) -> real_t {
      if (x <= ZERO) {
        return ONE;
      }
      const real_t e = std::floor(std::log10(static_cast<double>(x)));
      const real_t f = x / static_cast<real_t>(std::pow(10.0, e));
      real_t       nf;
      if (round) {
        nf = (f < static_cast<real_t>(1.5))
               ? ONE
               : ((f < static_cast<real_t>(3)) ? static_cast<real_t>(2)
                  : (f < static_cast<real_t>(7))
                    ? static_cast<real_t>(5)
                    : static_cast<real_t>(10));
      } else {
        nf = (f <= ONE) ? ONE
             : (f <= static_cast<real_t>(2))
               ? static_cast<real_t>(2)
               : (f <= static_cast<real_t>(5)) ? static_cast<real_t>(5)
                                               : static_cast<real_t>(10);
      }
      return nf * static_cast<real_t>(std::pow(10.0, e));
    }

    // a tick at k*pi/D, carried with its reduced fraction k/D == n/d
    struct PiTick {
      real_t val;
      int    n, d;
    };

    // format a multiple of pi as "0", "PI", "<n>PI", "PI/<d>", "<n>PI/<d>"
    // (the bitmap font has no greek glyph, so "PI" is spelled out)
    inline auto fmtPi(int n, int d) -> std::string {
      if (n == 0) {
        return "0";
      }
      std::string s;
      if (n < 0) {
        s += "-";
        n = -n;
      }
      if (n != 1) {
        s += std::to_string(n);
      }
      s += "PI";
      if (d != 1) {
        s += "/" + std::to_string(d);
      }
      return s;
    }

    // ticks at nice fractions of pi spanning [lo, hi]
    inline auto piTicks(real_t lo, real_t hi, int nticks) -> std::vector<PiTick> {
      std::vector<PiTick> out;
      const real_t        PI    = static_cast<real_t>(constant::PI);
      const real_t        range = hi - lo;
      if (not(range > ZERO) or nticks < 2) {
        return out;
      }
      const real_t   ideal = static_cast<real_t>(nticks - 1) * PI / range;
      const int      Ds[]  = { 1, 2, 3, 4, 6, 8, 12, 16, 24 };
      int            D     = 4;
      real_t         bestd = static_cast<real_t>(1e30);
      for (const int dd : Ds) {
        const real_t df = std::fabs(static_cast<double>(dd) - ideal);
        if (df < bestd) {
          bestd = df;
          D     = dd;
        }
      }
      const real_t step = PI / static_cast<real_t>(D);
      const int    k0 = static_cast<int>(std::ceil(static_cast<double>(lo / step) -
                                                1e-6));
      const int    k1 = static_cast<int>(
        std::floor(static_cast<double>(hi / step) + 1e-6));
      for (int k = k0; k <= k1; ++k) {
        int       n = k, d = D;
        const int g = std::gcd(std::abs(n), d);
        if (g > 0) {
          n /= g;
          d /= g;
        }
        out.push_back({ static_cast<real_t>(k) * step, n, d });
      }
      return out;
    }

    inline auto niceTicks(real_t lo, real_t hi, int n) -> std::vector<real_t> {
      std::vector<real_t> out;
      if (not(hi > lo) or n < 2) {
        return out;
      }
      const real_t step = niceNum((hi - lo) / static_cast<real_t>(n - 1), true);
      if (step <= ZERO) {
        return out;
      }
      const real_t g0  = std::ceil(static_cast<double>(lo / step)) * step;
      const real_t eps = static_cast<real_t>(1e-6) * step;
      for (real_t v = g0; v <= hi + static_cast<real_t>(0.5) * step; v += step) {
        if (v >= lo - eps and v <= hi + eps) {
          out.push_back((std::fabs(static_cast<double>(v)) < eps) ? ZERO : v);
        }
      }
      return out;
    }

  } // namespace axes_hidden

  /**
   * @brief Left/bottom margin (pixels) that the axes annotation needs.
   * @note Zero when axes are disabled. Depends only on H, so it can size the
   * canvas before drawing.
   */
  inline void axesMargins(bool axes, int H, int& ml, int& mb) {
    if (not axes) {
      ml = 0;
      mb = 0;
      return;
    }
    const int s   = cbar_hidden::scale(H);
    const int cw  = 6 * s;
    const int ch  = 8 * s;
    const int tl  = 5 * s;
    const int gap = 2 * s;
    ml = tl + gap + 7 * cw + gap + cw + gap; // ticks + numbers + y-axis name
    mb = tl + gap + ch + gap + ch + gap;     // ticks + numbers + x-axis name
  }

  /**
   * @brief Draw a 2D spine + ticks + labels around the data region.
   * @param rgba   canvas (CW*CH*4), opaque
   * @param CW,CH  canvas dimensions (includes margins)
   * @param x0     left pixel of the data region (== left margin width)
   * @param W,H    data region dimensions
   * @param u0,u1,v0,v1 world window mapped onto the data region (+v is up)
   * @param xlabel,ylabel axis names
   * @param bg     background RGB (for contrasting text color)
   * @param nticks target number of ticks per axis
   */
  inline void drawAxes2D(uint8_t*           rgba,
                         int                CW,
                         int                CH,
                         int                x0,
                         int                W,
                         int                H,
                         real_t             u0,
                         real_t             u1,
                         real_t             v0,
                         real_t             v1,
                         real_t             du0,
                         real_t             du1,
                         real_t             dv0,
                         real_t             dv1,
                         const std::string& xlabel,
                         const std::string& ylabel,
                         const real_t       bg[3],
                         int                nticks) {
    using namespace axes_hidden;
    const uint8_t c   = contrast(bg);
    const int     s   = cbar_hidden::scale(H);
    const int     ch  = 8 * s;
    const int     tl  = 5 * s;
    const int     gap = 2 * s;
    const int     th  = std::max(0, s / 2 - 1); // spine half-thickness

    // [u0,u1]x[v0,v1] is the world window mapped onto the full data region;
    // [du0,du1]x[dv0,dv1] is the actual data box (the domain/region), a sub-rect
    // when the window was aspect-expanded. The spine + ticks clamp to the DATA
    // box so the empty aspect pad stays outside the frame.
    const int xL = x0, xR = x0 + W - 1, yT = 0, yB = H - 1;
    auto      X  = [&](real_t u) -> int {
      return static_cast<int>(std::lround(
        static_cast<double>(xL) +
        static_cast<double>((u - u0) / (u1 - u0)) * (xR - xL)));
    };
    auto Y = [&](real_t v) -> int {
      return static_cast<int>(std::lround(
        static_cast<double>(yB) -
        static_cast<double>((v - v0) / (v1 - v0)) * (yB - yT)));
    };
    auto clampX = [&](int x) { return (x < xL) ? xL : ((x > xR) ? xR : x); };
    auto clampY = [&](int y) { return (y < yT) ? yT : ((y > yB) ? yB : y); };
    const int xLd = clampX(X(du0)), xRd = clampX(X(du1));
    const int yTd = clampY(Y(dv1)), yBd = clampY(Y(dv0)); // dv1 = top

    // spine around the data box
    line(rgba, CW, CH, xLd, yTd, xRd, yTd, th, c);
    line(rgba, CW, CH, xLd, yBd, xRd, yBd, th, c);
    line(rgba, CW, CH, xLd, yTd, xLd, yBd, th, c);
    line(rgba, CW, CH, xRd, yTd, xRd, yBd, th, c);

    // x ticks (below the data box): marks + labels
    for (const real_t tv : niceTicks(du0, du1, nticks)) {
      const int x = X(tv);
      if (x < xLd or x > xRd) {
        continue;
      }
      line(rgba, CW, CH, x, yBd, x, yBd + tl, 0, c);
      const std::string lab = cbar_hidden::fmtNum(tv);
      text(rgba, CW, CH, x - textW(lab, s) / 2, yBd + tl + gap, lab, s, c);
    }
    // y ticks (left of the data box): marks + right-aligned labels
    for (const real_t tv : niceTicks(dv0, dv1, nticks)) {
      const int y = Y(tv);
      if (y < yTd or y > yBd) {
        continue;
      }
      line(rgba, CW, CH, xLd, y, xLd - tl, y, 0, c);
      const std::string lab = cbar_hidden::fmtNum(tv);
      text(rgba, CW, CH, xLd - tl - gap - textW(lab, s), y - ch / 2, lab, s, c);
    }
    // axis names
    if (not xlabel.empty()) {
      text(rgba, CW, CH, (xLd + xRd) / 2 - textW(xlabel, s) / 2,
           yBd + tl + gap + ch + gap, xlabel, s, c);
    }
    if (not ylabel.empty()) {
      textVert(rgba, CW, CH,
               std::max(gap, xLd - tl - gap - 7 * (6 * s) - gap - 6 * s),
               (yTd + yBd) / 2 - 4 * s * static_cast<int>(ylabel.size()) / 2,
               ylabel, s, c);
    }
  }

  /**
   * @brief Draw polar (curvilinear) axes for a 2D spherical meridional slice.
   * @param x0,W,H data region (the slice maps world (X = r sin th, Z = r cos th)
   *               onto it via the [u0,u1]x[v0,v1] window, aspect-matched)
   * @param rmin,rmax,tmin,tmax global (r, theta) extent
   * @param mirror whether the half-plane is mirrored into a full disk
   * @param rlabel,tlabel names for the radial / angular axes (e.g. "R","Theta")
   * @note Draws (1) a curvilinear spine: the outer & inner arcs plus the two
   * radial edges (or full arcs when mirrored); (2) an "R" radial axis along the
   * symmetry axis (X=0) with R=0 at the center, increasing outward; (3) a
   * "Theta" angular axis with ticks along the outer arc.
   */
  inline void drawAxesPolar(uint8_t*           rgba,
                            int                CW,
                            int                CH,
                            int                x0,
                            int                W,
                            int                H,
                            real_t             u0,
                            real_t             u1,
                            real_t             v0,
                            real_t             v1,
                            real_t             rmin,
                            real_t             rmax,
                            real_t             tmin,
                            real_t             tmax,
                            bool               mirror,
                            const std::string& rlabel,
                            const std::string& tlabel,
                            const real_t       bg[3],
                            int                nticks) {
    using namespace axes_hidden;
    const uint8_t c   = contrast(bg);
    const int     s   = cbar_hidden::scale(H);
    const int     ch  = 8 * s;
    const int     tl  = 5 * s;
    const int     gap = 2 * s;

    auto WX = [&](real_t X) -> real_t {
      return static_cast<real_t>(x0) + (X - u0) / (u1 - u0) * W - HALF;
    };
    auto WZ = [&](real_t Z) -> real_t {
      return (v1 - Z) / (v1 - v0) * H - HALF;
    };
    auto PX = [&](real_t X, real_t Z, int& qx, int& qy) {
      qx = static_cast<int>(std::lround(WX(X)));
      qy = static_cast<int>(std::lround(WZ(Z)));
    };

    const int NA = 160;
    auto      arc = [&](real_t r, real_t sgn) {
      int qx, qy;
      PX(sgn * r * std::sin(static_cast<double>(tmin)),
         r * std::cos(static_cast<double>(tmin)), qx, qy);
      for (int i = 1; i <= NA; ++i) {
        const real_t th = tmin + (tmax - tmin) * i / NA;
        int          rx, ry;
        PX(sgn * r * std::sin(static_cast<double>(th)),
           r * std::cos(static_cast<double>(th)), rx, ry);
        line(rgba, CW, CH, qx, qy, rx, ry, 0, c);
        qx = rx;
        qy = ry;
      }
    };
    auto ray = [&](real_t th) {
      int ax, ay, bx, by;
      PX(rmin * std::sin(static_cast<double>(th)),
         rmin * std::cos(static_cast<double>(th)), ax, ay);
      PX(rmax * std::sin(static_cast<double>(th)),
         rmax * std::cos(static_cast<double>(th)), bx, by);
      line(rgba, CW, CH, ax, ay, bx, by, 0, c);
    };

    // ---- curvilinear spine -------------------------------------------- //
    arc(rmax, ONE);
    arc(rmin, ONE);
    if (mirror) {
      arc(rmax, -ONE);
      arc(rmin, -ONE);
    } else {
      ray(tmin);
      ray(tmax);
    }

    // ---- R axis: along the symmetry axis (X = 0), zero at the center --- //
    const int rmaxlabW = textW(cbar_hidden::fmtNum(rmax), s);
    for (const real_t Rv : niceTicks(ZERO, rmax, nticks)) {
      for (int sg = 1; sg >= -1; sg -= 2) {
        if (sg < 0 and Rv == ZERO) {
          continue; // a single tick at the center
        }
        int ax, ay;
        PX(ZERO, static_cast<real_t>(sg) * Rv, ax, ay);
        line(rgba, CW, CH, ax, ay, ax - tl, ay, 0, c);
        const std::string lab = cbar_hidden::fmtNum(Rv);
        text(rgba, CW, CH, ax - tl - gap - textW(lab, s), ay - ch / 2, lab, s, c);
      }
    }
    if (not rlabel.empty()) {
      int ax, ay;
      PX(ZERO, ZERO, ax, ay);
      const real_t off = tl + gap + rmaxlabW + gap + ch;
      textRotCentered(rgba, CW, CH, ax - off, static_cast<real_t>(ay), rlabel, s,
                      ZERO, -ONE, c);
    }

    // ---- Theta axis: ticks + labels (fractions of pi) along the arc --- //
    // widest tick label, so the "Theta" name can clear them all
    real_t maxlw = static_cast<real_t>(ch);
    for (const auto& tk : piTicks(tmin, tmax, nticks)) {
      maxlw = std::max(maxlw,
                       static_cast<real_t>(textW(fmtPi(tk.n, tk.d), s)));
    }
    for (const auto& tk : piTicks(tmin, tmax, nticks)) {
      const real_t ox = std::sin(static_cast<double>(tk.val));
      const real_t oz = std::cos(static_cast<double>(tk.val));
      int          px0, py0;
      PX(rmax * ox, rmax * oz, px0, py0);
      const real_t dxp = ox, dyp = -oz; // outward pixel direction
      line(rgba, CW, CH, px0, py0,
           static_cast<int>(std::lround(px0 + dxp * tl)),
           static_cast<int>(std::lround(py0 + dyp * tl)), 0, c);
      const std::string lab = fmtPi(tk.n, tk.d);
      // push the (horizontal) label box fully clear of the arc/tick at any
      // angle: offset its center by its own support along the outward direction
      const real_t inset = HALF * (static_cast<real_t>(textW(lab, s)) *
                                     std::fabs(static_cast<double>(dxp)) +
                                   static_cast<real_t>(ch) *
                                     std::fabs(static_cast<double>(dyp)));
      const real_t lo = tl + gap + inset;
      text(rgba, CW, CH,
           static_cast<int>(std::lround(px0 + dxp * lo)) - textW(lab, s) / 2,
           static_cast<int>(std::lround(py0 + dyp * lo)) - ch / 2, lab, s, c);
    }
    if (not tlabel.empty()) {
      const real_t tm = HALF * (tmin + tmax);
      const real_t ox = std::sin(static_cast<double>(tm));
      const real_t oz = std::cos(static_cast<double>(tm));
      int          px0, py0;
      PX(rmax * ox, rmax * oz, px0, py0);
      real_t adx = std::cos(static_cast<double>(tm)); // arc tangent
      real_t ady = std::sin(static_cast<double>(tm));
      readableDir(adx, ady);
      // beyond the tick labels (which reach ~tl+gap+maxlw from the arc)
      const real_t off = tl + gap + maxlw + gap + ch;
      textRotCentered(rgba, CW, CH, px0 + ox * off, py0 - oz * off, tlabel, s,
                      adx, ady, c);
    }
  }

  /**
   * @brief Draw a 3D bounding-box wireframe spine + ticks + labels.
   * @param rgba  canvas (CW*CH*4), opaque
   * @param CW,CH canvas dimensions
   * @param x0    left pixel of the data region (left margin width)
   * @param W,H   data region dimensions (used for the camera projection)
   * @param cam   ray-march camera (inverted to project world -> screen)
   * @param ext   global box extent (3 axes)
   * @param lab   axis names [x, y, z]
   * @param bg    background RGB
   * @param nticks target number of ticks per axis
   */
  inline void drawAxes3D(uint8_t*                    rgba,
                         int                         CW,
                         int                         CH,
                         int                         x0,
                         int                         W,
                         int                         H,
                         const CameraDevice&         cam,
                         const boundaries_t<real_t>& ext,
                         const std::string           lab[3],
                         const real_t                bg[3],
                         int                         nticks) {
    using namespace axes_hidden;
    if (ext.size() < 3) {
      return;
    }
    const uint8_t c  = contrast(bg);
    const int     s  = cbar_hidden::scale(H);
    const int     ch = 8 * s;
    const int     tl = 5 * s;

    auto corner = [&](int m, real_t p[3]) {
      p[0] = (m & 1) ? ext[0].second : ext[0].first;
      p[1] = (m & 2) ? ext[1].second : ext[1].first;
      p[2] = (m & 4) ? ext[2].second : ext[2].first;
    };
    real_t cx[8], cy[8];
    bool   ok[8];
    for (int m = 0; m < 8; ++m) {
      real_t p[3];
      corner(m, p);
      real_t a, b;
      ok[m] = projectToScreen(cam, W, H, p, a, b);
      cx[m] = a + static_cast<real_t>(x0);
      cy[m] = b;
    }
    // box centroid (for outward tick/label direction)
    real_t cen[3] = { HALF * (ext[0].first + ext[0].second),
                      HALF * (ext[1].first + ext[1].second),
                      HALF * (ext[2].first + ext[2].second) };
    real_t ccx = ZERO, ccy = ZERO;
    {
      real_t a, b;
      projectToScreen(cam, W, H, cen, a, b);
      ccx = a + static_cast<real_t>(x0);
      ccy = b;
    }

    (void)ok;
    // The wireframe "spine" is drawn in the ray-march (depth-occluded), so here
    // we only annotate. For each axis, pick one *silhouette* edge (its two
    // adjacent faces face opposite ways) and, among the two candidates, the one
    // whose screen position matches the convention x=bottom, y & z on the left
    // of the default diagonal view.
    auto frontFace = [&](int axis, int side) -> bool {
      const real_t nrm = (side != 0) ? ONE : -ONE; // outward normal sign
      return (nrm * (-cam.forward[axis])) > ZERO;   // points toward the camera?
    };
    for (int d = 0; d < 3; ++d) {
      const int e1 = (d == 0) ? 1 : 0;
      const int e2 = (d == 2) ? 1 : 2;
      int       bs1 = 0, bs2 = 0;
      bool      found = false;
      real_t    best = ZERO;
      for (int s1 = 0; s1 < 2; ++s1) {
        for (int s2 = 0; s2 < 2; ++s2) {
          const int    m0 = (s1 << e1) | (s2 << e2);
          const int    m1 = m0 | (1 << d);
          const real_t mx = HALF * (cx[m0] + cx[m1]);
          const real_t my = HALF * (cy[m0] + cy[m1]);
          // screen-position convention: x on the bottom, y & z on the left edges
          real_t       score = (d == 0) ? my : -mx;
          if (frontFace(e1, s1) != frontFace(e2, s2)) {
            score += static_cast<real_t>(1e6); // strongly prefer silhouette edges
          }
          if (not found or score > best) {
            best  = score;
            bs1   = s1;
            bs2   = s2;
            found = true;
          }
        }
      }
      const int m0 = (bs1 << e1) | (bs2 << e2);
      const int m1 = m0 | (1 << d);
      real_t    o[3];
      corner(m0, o); // perpendicular coords fixed; axis d swept for ticks
      const real_t lo = ext[d].first, hi = ext[d].second;
      // screen-space perpendicular to the edge, flipped to point outward
      real_t ex = cx[m1] - cx[m0], ey = cy[m1] - cy[m0];
      real_t el = std::sqrt(static_cast<double>(ex * ex + ey * ey));
      if (el < ONE) {
        el = ONE;
      }
      ex /= el;
      ey /= el;
      real_t pxd = -ey, pyd = ex;
      {
        const real_t mxv = HALF * (cx[m0] + cx[m1]) - ccx;
        const real_t myv = HALF * (cy[m0] + cy[m1]) - ccy;
        if (pxd * mxv + pyd * myv < ZERO) {
          pxd = -pxd;
          pyd = -pyd;
        }
      }
      // readable text baseline aligned with the edge direction
      real_t adx = ex, ady = ey;
      readableDir(adx, ady);
      const real_t numOff  = static_cast<real_t>(tl) + 5 * s;  // number center
      const real_t nameOff = static_cast<real_t>(tl) + 14 * s; // axis-name center
      // ticks + numeric labels (numbers rotated along the edge for x & y; the
      // vertical z edge keeps horizontal numbers, which read more easily)
      for (const real_t tv : niceTicks(lo, hi, nticks)) {
        real_t p[3] = { o[0], o[1], o[2] };
        p[d]        = tv;
        real_t a, b;
        if (not projectToScreen(cam, W, H, p, a, b)) {
          continue;
        }
        a += static_cast<real_t>(x0);
        const int mx = static_cast<int>(std::lround(a + pxd * tl));
        const int my = static_cast<int>(std::lround(b + pyd * tl));
        line(rgba, CW, CH, static_cast<int>(std::lround(a)),
             static_cast<int>(std::lround(b)), mx, my, 0, c);
        const std::string l2 = cbar_hidden::fmtNum(tv);
        if (d == 2) {
          const int tx = (pxd < ZERO) ? (mx - textW(l2, s)) : mx;
          text(rgba, CW, CH, tx, my - ch / 2, l2, s, c);
        } else {
          textRotCentered(rgba, CW, CH, a + pxd * numOff, b + pyd * numOff, l2,
                          s, adx, ady, c);
        }
      }
      // axis name at the MIDDLE of the edge (near the central tick), aligned
      // with the edge and pushed further outward than the numbers
      if (not lab[d].empty()) {
        real_t mid[3] = { o[0], o[1], o[2] };
        mid[d]        = HALF * (lo + hi);
        real_t a, b;
        if (projectToScreen(cam, W, H, mid, a, b)) {
          a += static_cast<real_t>(x0);
          textRotCentered(rgba, CW, CH, a + pxd * nameOff, b + pyd * nameOff,
                          lab[d], s, adx, ady, c);
        }
      }
    }
  }

} // namespace out

#endif // OUTPUT_RENDER_AXES_H
