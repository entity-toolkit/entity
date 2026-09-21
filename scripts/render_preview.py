#!/usr/bin/env python3
"""
render_preview.py -- fast, data-free preview of the entity in-situ renderer's
SCENE GEOMETRY.

Purpose
-------
Reads a simulation `.toml` and draws the domain box / camera framing / axes /
region crop / field-line seed lattice, WITHOUT any simulation data or
ray-marching. It lets you iterate on camera orientation (e.g. the domain cube of
a 3D turbulence run) and framing without relaunching the simulation.

It reproduces the SAME camera / projection the C++ renderer uses, so the preview
is trustworthy: the box you see here is the box the renderer will draw.

  IMPORTANT: the camera / projection / region / field-line-lattice math below is
  a faithful port of the C++ renderer. If the C++ changes, THIS MUST BE UPDATED
  IN SYNC. The controlling C++ sources (verified line-by-line while writing this)
  are:
    - src/output/render/renderer.cpp
        Renderer::init  -> toml parse, region resolution, default camera,
                           ortho/persp basis, ortho_height default = box diag,
                           eye = center + 1.7*diag*(1,1,1)/sqrt3, fov 35 deg
        Renderer::updateForTime -> moving view (pure pan of region + eye)
    - src/output/render/composite.h
        projectToScreen (~L85-112) -> world -> pixel, ortho & perspective
        screenBBox      (~L121-163)
    - src/output/render/raymarch.hpp
        ray generation (~L308-335) -- the inverse of projectToScreen
    - src/output/render/axes.h
        drawAxes3D / drawAxes2D / drawAxesPolar, niceTicks / niceNum tick style
    - src/framework/domain/metadomain_render.cpp
        2D window derivation (Cartesian window vs. spherical meridional wedge,
        X = r sin th, Z = r cos th, aspect expansion, mirror), field-line setup
    - src/framework/parameters/grid.cpp (~L470-497)
        extent parse + theta,phi auto-fill for non-Cartesian metrics
    - src/output/render/fieldlines.h
        3D seed lattice: spacing = max(seed_px,1)*wpp, grown by
        cbrt(n_seed/seed_max) if over seed_max, ns[d]=floor(size[d]/spacing),
        seeds at cell centers.

Modes
-----
  * 3D (Cartesian only -- the renderer's only 3D mode): projects the domain cube
    (and region crop, if any) with the exact ray-march camera; optional field-
    line SEED lattice scatter (schematic: seeds, not traced lines).
  * 2D Cartesian: aspect-expanded slice window + domain/region box + ticks.
  * 2D spherical/GR: meridional wedge (arcs at r in {rmin,rmax}, rays at
    theta in {tmin,tmax}), mirrored into a full disk if `mirror`.
  * 1D: nothing to render (warns).

Usage
-----
  module load python/3.13.0
  python render_preview.py <toml> [--out preview.png] [--time T] [--scene N]

If --out is omitted, saves to <toml_dir>/<simname>_preview.png.
"""

import argparse
import math
import os
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 and older (e.g. the miniforge3 module)
    import tomli as tomllib  # same load() API

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless cluster: no interactive display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# --------------------------------------------------------------------------- #
#  toml helpers (mirror toml::find_or: return default if any key is missing)   #
# --------------------------------------------------------------------------- #
def find_or(d, default, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------- #
#  metric / extent handling (grid.cpp ~L416-497)                              #
# --------------------------------------------------------------------------- #
CARTESIAN_METRICS = {"minkowski"}
# everything else that entity supports is curvilinear (r-first extent):
#   spherical, qspherical, kerr_schild, kerr_schild_0, qkerr_schild
def is_cartesian(metric_name):
    return metric_name.strip().lower() in CARTESIAN_METRICS


def global_extent(td):
    """Replicates grid.cpp: parse grid.extent, auto-fill theta/phi for
    non-Cartesian metrics. Returns (extent_pairs, dim, cartesian, metric_name).

    extent_pairs is a list of (lo, hi) of length == dim (== len(resolution)).
    """
    resolution = find_or(td, None, "grid", "resolution")
    if resolution is None:
        raise ValueError("grid.resolution missing")
    dim = len(resolution)
    metric_name = find_or(td, "minkowski", "grid", "metric", "metric")
    cart = is_cartesian(metric_name)

    extent = find_or(td, None, "grid", "extent")
    if extent is None:
        raise ValueError("grid.extent missing")
    # deep copy as list of [lo,hi]
    ext = [list(pair) for pair in extent]

    # grid.cpp: if extent has more rows than dim, truncate to dim
    if len(ext) > dim:
        ext = ext[:dim]

    if not cart:
        # non-Cartesian: extent gives only the r-range; append theta,(phi).
        # (grid.cpp errors if >1 row is supplied for non-cartesian; we just
        #  keep the r-row and append.)
        ext = [ext[0]]
        ext.append([0.0, math.pi])            # theta in [0, pi]  (2D and 3D)
        if dim == 3:
            ext.append([0.0, 2.0 * math.pi])  # phi in [0, 2pi]   (3D only)

    if len(ext) != dim:
        raise ValueError(f"inferred grid.extent has {len(ext)} rows, expected {dim}")

    pairs = [(float(p[0]), float(p[1])) for p in ext]
    return pairs, dim, cart, metric_name


# --------------------------------------------------------------------------- #
#  Camera (renderer.cpp Renderer::init, composite.h projectToScreen)          #
# --------------------------------------------------------------------------- #
def _norm3(a):
    n = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if n > 1e-30:
        return (a[0] / n, a[1] / n, a[2] / n)
    return (a[0], a[1], a[2])


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class Camera:
    """POD camera identical to out::CameraDevice, built exactly as in
    Renderer::init. eye/forward/right/up in world coords; ortho/persp framing."""

    def __init__(self, td, region, width, height):
        # region is a list of (lo,hi); may be < 3 axes (zero-filled), mirroring
        # the C++ which sizes center/size over m_region.size() and d<3.
        center = [0.0, 0.0, 0.0]
        size = [0.0, 0.0, 0.0]
        for d in range(min(len(region), 3)):
            center[d] = 0.5 * (region[d][0] + region[d][1])
            size[d] = region[d][1] - region[d][0]
        diag = math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)

        cam_ortho = find_or(td, True, "output", "render", "camera", "orthographic")
        pos = find_or(td, [], "output", "render", "camera", "position")
        look = find_or(td, [], "output", "render", "camera", "look_at")
        up = find_or(td, [], "output", "render", "camera", "up")
        fov = float(find_or(td, 35.0, "output", "render", "camera", "fov"))
        # default ortho_height covers the box from any view -> == box diagonal
        ortho_height = float(
            find_or(td, diag, "output", "render", "camera", "ortho_height"))

        # default eye: box center pushed back along (1,1,1) by ~1.7 diagonals
        eye = [0.0, 0.0, 0.0]
        lookat = [0.0, 0.0, 0.0]
        for d in range(3):
            if len(pos) == 3:
                eye[d] = float(pos[d])
            else:
                eye[d] = center[d] + 1.7 * diag * 0.57735026919
            lookat[d] = float(look[d]) if len(look) == 3 else center[d]
        if len(up) == 3:
            upv = [float(up[0]), float(up[1]), float(up[2])]
        else:
            upv = [0.0, 0.0, 1.0]

        forward = _norm3([lookat[0] - eye[0], lookat[1] - eye[1],
                          lookat[2] - eye[2]])
        right = _norm3(_cross3(forward, upv))
        up_cam = _cross3(right, forward)  # already unit (right,forward unit & perp)

        self.eye = list(eye)
        self.forward = list(forward)
        self.right = list(right)
        self.up = list(up_cam)
        self.aspect = float(width) / float(height)
        self.tan_half_fov = math.tan(0.5 * fov * math.pi / 180.0)
        self.orthographic = bool(cam_ortho)
        self.half_h = 0.5 * ortho_height
        self.half_w = self.half_h * self.aspect
        # keep for the summary line
        self.ortho_height = ortho_height
        self.fov = fov

    def pan(self, shift):
        """Moving view: pure pan of the eye (forward/right/up/half_h unchanged),
        matching Renderer::updateForTime."""
        for d in range(3):
            self.eye[d] += shift[d]

    def project(self, p, W, H):
        """projectToScreen: world point -> (px, py) in pixels (py DOWN,
        image origin top-left). Returns None if behind a perspective camera."""
        dx = p[0] - self.eye[0]
        dy = p[1] - self.eye[1]
        dz = p[2] - self.eye[2]
        cx = dx * self.right[0] + dy * self.right[1] + dz * self.right[2]
        cy = dx * self.up[0] + dy * self.up[1] + dz * self.up[2]
        if self.orthographic:
            fx = cx / self.half_w
            fy = cy / self.half_h
        else:
            cz = dx * self.forward[0] + dy * self.forward[1] + dz * self.forward[2]
            if cz <= 1e-6:
                return None
            fx = (cx / cz) / (self.aspect * self.tan_half_fov)
            fy = (cy / cz) / self.tan_half_fov
        px = (fx + 1.0) * 0.5 * W - 0.5
        py = (1.0 - fy) * 0.5 * H - 0.5
        return (px, py)


# --------------------------------------------------------------------------- #
#  region resolution (renderer.cpp Renderer::init ~L128-158)                  #
# --------------------------------------------------------------------------- #
def resolve_region(td, ext):
    """m_region starts as the global extent; x{d+1}_lim clamps axis d to the box
    if it is a valid [lo,hi] with hi>lo that overlaps. Returns (region, has_region)."""
    region = [list(p) for p in ext]
    has_region = False
    keys = ["x1_lim", "x2_lim", "x3_lim"]
    for d in range(min(len(ext), 3)):
        lim = find_or(td, [], "output", "render", keys[d])
        if not lim:
            continue
        if len(lim) != 2 or lim[1] <= lim[0]:
            print(f"  warning: output.render.{keys[d]} must be [lo,hi] with hi>lo; ignoring")
            continue
        lo = max(float(lim[0]), ext[d][0])
        hi = min(float(lim[1]), ext[d][1])
        if hi > lo:
            region[d] = [lo, hi]
            has_region = True
        else:
            print(f"  warning: output.render.{keys[d]} does not overlap the domain; ignoring")
    return [tuple(p) for p in region], has_region


def apply_moving_view(td, region, ext, cam, time, dim):
    """Renderer::updateForTime: dt=max(0, t-t0); shift=vel*dt; pan region+eye."""
    vel = find_or(td, [], "output", "render", "camera_velocity")
    if not vel:
        return region
    v = [0.0, 0.0, 0.0]
    for d in range(min(len(vel), 3)):
        v[d] = float(vel[d])
    if v == [0.0, 0.0, 0.0]:
        return region
    t0 = float(find_or(td, 0.0, "output", "render", "camera_start_time"))
    dt = max(0.0, time - t0)
    shift = [v[0] * dt, v[1] * dt, v[2] * dt]
    new_region = []
    for d in range(len(region)):
        s = shift[d] if d < 3 else 0.0
        new_region.append((region[d][0] + s, region[d][1] + s))
    cam.pan(shift)  # cam only meaningful for the 3D mode; harmless otherwise
    return new_region


# --------------------------------------------------------------------------- #
#  "nice" ticks (axes.h niceNum / niceTicks) for annotation                    #
# --------------------------------------------------------------------------- #
def _nice_num(x, do_round):
    if x <= 0.0:
        return 1.0
    e = math.floor(math.log10(x))
    f = x / (10.0 ** e)
    if do_round:
        nf = 1.0 if f < 1.5 else (2.0 if f < 3.0 else (5.0 if f < 7.0 else 10.0))
    else:
        nf = 1.0 if f <= 1.0 else (2.0 if f <= 2.0 else (5.0 if f <= 5.0 else 10.0))
    return nf * (10.0 ** e)


def nice_ticks(lo, hi, n):
    out = []
    if not (hi > lo) or n < 2:
        return out
    step = _nice_num((hi - lo) / (n - 1), True)
    if step <= 0.0:
        return out
    g0 = math.ceil(lo / step) * step
    eps = 1e-6 * step
    v = g0
    while v <= hi + 0.5 * step:
        if lo - eps <= v <= hi + eps:
            out.append(0.0 if abs(v) < eps else v)
        v += step
    return out


# --------------------------------------------------------------------------- #
#  3D field-line SEED lattice (fieldlines.h traceFieldLines seed setup)        #
# --------------------------------------------------------------------------- #
def field_line_seeds_3d(td, region, cam, H):
    """Reproduce the seed-lattice geometry (NOT the traced lines).

    world_per_pixel wpp = (cam.half_h*2)/H   (metadomain_render.cpp)
    spacing = max(seed_px,1)*wpp; if n_seed>seed_max: spacing *= cbrt(n_seed/seed_max)
    ns[d]  = max(1, floor(size[d]/spacing)); seeds at cell centers over the
    field-line COARSE grid, which spans the FULL global extent -- BUT here we
    seed over the render region's box (== extent when uncropped), matching the
    coarse grid origin=extent.first in the uncropped case. We use the region box
    the camera frames so the schematic overlays the drawn cube.
    """
    fl_enable = find_or(td, False, "output", "render", "fieldlines", "enable")
    # any scene may also request the overlay
    scenes = find_or(td, [], "output", "render", "scenes")
    any_fl = any(
        (find_or(sc, False, "fieldlines") or find_or(sc, "", "field") == "fieldlines")
        for sc in scenes
    )
    if not (fl_enable or any_fl):
        return None

    seed_px = float(find_or(td, 8.0, "output", "render", "fieldlines", "seed_px"))
    seed_max = int(find_or(td, 4096, "output", "render", "fieldlines", "seed_max"))

    wpp = (cam.half_h * 2.0) / float(H)
    size = [region[d][1] - region[d][0] for d in range(3)]
    origin = [region[d][0] for d in range(3)]

    spacing = max(seed_px, 1.0) * wpp

    def count_seeds(sp):
        ns = []
        tot = 1
        for d in range(3):
            n = max(1, int(math.floor(size[d] / sp))) if sp > 0 else 1
            ns.append(n)
            tot *= n
        return tot, ns

    n_seed, ns = count_seeds(spacing)
    if n_seed > seed_max and seed_max > 0:
        grow = (float(n_seed) / float(seed_max)) ** (1.0 / 3.0)
        spacing *= grow
        n_seed, ns = count_seeds(spacing)

    seeds = []
    for k in range(ns[2]):
        for j in range(ns[1]):
            for i in range(ns[0]):
                seeds.append((
                    origin[0] + (i + 0.5) * size[0] / ns[0],
                    origin[1] + (j + 0.5) * size[1] / ns[1],
                    origin[2] + (k + 0.5) * size[2] / ns[2],
                ))
    return seeds, ns


# --------------------------------------------------------------------------- #
#  cube edges: the 12 edges of an axis-aligned box given by (lo,hi) per axis    #
# --------------------------------------------------------------------------- #
def cube_corners(box):
    """box: list of (lo,hi) for 3 axes. Corner m: bit0->x, bit1->y, bit2->z
    (matches the C++ corner() ordering in axes.h / screenBBox)."""
    corners = []
    for m in range(8):
        corners.append((
            box[0][1] if (m & 1) else box[0][0],
            box[1][1] if (m & 2) else box[1][0],
            box[2][1] if (m & 4) else box[2][0],
        ))
    return corners


# the 12 edges as (corner_i, corner_j) index pairs
CUBE_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),   # x-parallel
    (0, 2), (1, 3), (4, 6), (5, 7),   # y-parallel
    (0, 4), (1, 5), (2, 6), (3, 7),   # z-parallel
]


# --------------------------------------------------------------------------- #
#  3D axis-annotation edge selection (axes.h drawAxes3D)                        #
#                                                                              #
#  A box axis has FOUR parallel edges; the ticks/labels go on exactly one. We  #
#  only ever pick a *silhouette* edge -- one whose two adjacent faces point in  #
#  opposite directions relative to the camera (one toward it, one away). A      #
#  silhouette edge always lies on the drawn OUTLINE of the box, so it is in the #
#  foreground -- never occluded by the volume -- and its ticks, pushed outward  #
#  from the projected centroid, land in empty background. Among the (usually    #
#  two) silhouette candidates we take the one nearest the bottom-left of the    #
#  image (score = pixel_y - pixel_x), the conventional place for annotation.    #
#  This is a faithful port of out::drawAxes3D so the preview matches the        #
#  renderer's choice of which spine carries the axis.                           #
# --------------------------------------------------------------------------- #
def _front_face(cam, axis, side):
    """True if the box face perpendicular to `axis` at `side` (1=high, 0=low)
    faces the camera (its outward normal points back toward the eye)."""
    nrm = 1.0 if side else -1.0
    return (nrm * (-cam.forward[axis])) > 0.0


def select_axis_edge_3d(cam, d, cx, cy, ccx, ccy):
    """Choose the edge parallel to axis d that carries the ticks + label.

    cx,cy are the 8 projected corner pixel coords (cube_corners order); ccx,ccy
    the projected box centroid. Returns (m0, m1, pxd, pyd): the edge's two corner
    indices (m0 has axis d at its low end) and the unit screen-space OUTWARD push
    direction (perpendicular to the edge, pointing away from the centroid)."""
    e1 = 1 if d == 0 else 0            # the two perpendicular axes
    e2 = 1 if d == 2 else 2
    best = None
    for s1 in (0, 1):
        for s2 in (0, 1):
            m0 = (s1 << e1) | (s2 << e2)
            m1 = m0 | (1 << d)
            mx = 0.5 * (cx[m0] + cx[m1])
            my = 0.5 * (cy[m0] + cy[m1])
            score = my - mx  # prefer the foreground (bottom-left) edge
            if _front_face(cam, e1, s1) != _front_face(cam, e2, s2):
                score += 1e6  # strongly prefer silhouette (outline) edges
            if best is None or score > best[0]:
                best = (score, m0, m1)
    _, m0, m1 = best
    ex, ey = cx[m1] - cx[m0], cy[m1] - cy[m0]
    el = math.hypot(ex, ey) or 1.0
    ex, ey = ex / el, ey / el
    pxd, pyd = -ey, ex                 # screen-perpendicular to the edge
    mxv = 0.5 * (cx[m0] + cx[m1]) - ccx
    myv = 0.5 * (cy[m0] + cy[m1]) - ccy
    if pxd * mxv + pyd * myv < 0.0:    # flip to point away from the box centroid
        pxd, pyd = -pxd, -pyd
    return m0, m1, pxd, pyd


# --------------------------------------------------------------------------- #
#  3D preview                                                                  #
# --------------------------------------------------------------------------- #
def draw_3d(td, ext, region, has_region, cam, W, H, out_path, sim_name):
    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)

    def project_box(box, color, lw, label, ls="-"):
        corners = cube_corners(box)
        proj = [cam.project(c, W, H) for c in corners]
        first = True
        for (a, b) in CUBE_EDGES:
            pa, pb = proj[a], proj[b]
            if pa is None or pb is None:
                continue  # edge with a corner behind a perspective camera
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=lw, ls=ls,
                    label=(label if first else None), zorder=3)
            first = False

    # full extent (light gray)
    project_box([ext[0], ext[1], ext[2]], color="0.6", lw=1.2,
                label="full extent")
    # region crop, if distinct
    if has_region:
        project_box([region[0], region[1], region[2]], color="tab:blue",
                    lw=2.0, label="region crop")

    # axes tick labels: for each axis, pick the FOREGROUND (silhouette) edge of
    # the framed box and annotate along it, exactly as out::drawAxes3D does, so
    # the labels never end up on an edge hidden behind the volume.
    axes_on = find_or(td, False, "output", "render", "axes")
    nticks = int(find_or(td, 5, "output", "render", "axis_ticks"))
    frame_box = region if has_region else ext
    axis_names = find_or(td, [], "output", "render", "axis_labels")
    default_names = ["x", "y", "z"]
    if axes_on:
        corners = cube_corners([frame_box[0], frame_box[1], frame_box[2]])
        cx = [0.0] * 8
        cy = [0.0] * 8
        for m in range(8):
            pr = cam.project(corners[m], W, H)
            cx[m] = pr[0] if pr is not None else 0.0
            cy[m] = pr[1] if pr is not None else 0.0
        cen = [0.5 * (frame_box[d][0] + frame_box[d][1]) for d in range(3)]
        prc = cam.project(cen, W, H)
        ccx = prc[0] if prc is not None else 0.0
        ccy = prc[1] if prc is not None else 0.0

        tl = 8.0            # tick-mark length [px]
        num_off = tl + 10.0   # numeric-label center offset from the edge [px]
        name_off = tl + 30.0  # axis-name center offset from the edge [px]
        for d in range(3):
            name = axis_names[d] if d < len(axis_names) else default_names[d]
            m0, m1, pxd, pyd = select_axis_edge_3d(cam, d, cx, cy, ccx, ccy)
            # o = corner(m0): perpendicular coords fixed, axis d swept for ticks
            o = list(corners[m0])
            lo_d, hi_d = frame_box[d][0], frame_box[d][1]
            for tv in nice_ticks(lo_d, hi_d, nticks):
                p = list(o)
                p[d] = tv
                pr = cam.project(p, W, H)
                if pr is None:
                    continue
                a, b = pr
                ax.plot([a, a + pxd * tl], [b, b + pyd * tl],
                        color="0.35", lw=1.0, zorder=4)
                ax.annotate(f"{tv:g}", (a + pxd * num_off, b + pyd * num_off),
                            fontsize=6, color="0.25", ha="center", va="center")
            # axis name at the MIDDLE of the chosen edge, pushed further outward
            mid = list(o)
            mid[d] = 0.5 * (lo_d + hi_d)
            pr = cam.project(mid, W, H)
            if pr is not None:
                a, b = pr
                ax.annotate(name, (a + pxd * name_off, b + pyd * name_off),
                            fontsize=9, color="k", fontweight="bold",
                            ha="center", va="center")

    # field-line SEED lattice (schematic scatter, NOT traced lines)
    fl = field_line_seeds_3d(td, frame_box, cam, H)
    if fl is not None:
        seeds, ns = fl
        pxs, pys = [], []
        for s in seeds:
            pr = cam.project(s, W, H)
            if pr is not None:
                pxs.append(pr[0])
                pys.append(pr[1])
        if pxs:
            ax.scatter(pxs, pys, s=8, c="tab:red", marker="o", alpha=0.6,
                       edgecolors="none", zorder=2,
                       label=f"field-line seeds (schematic, {ns[0]}x{ns[1]}x{ns[2]})")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # inverted y: origin upper-left, matches the PNG
    ax.set_aspect("equal")  # lock width:height 1:1 in pixel space
    ax.set_xlabel("screen x [px]")
    ax.set_ylabel("screen y [px]")
    proj_kind = "orthographic" if cam.orthographic else f"perspective (fov {cam.fov:g})"
    ax.set_title(f"{sim_name}: 3D scene preview ({proj_kind})", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  2D window derivation (metadomain_render.cpp 2D branch)                       #
# --------------------------------------------------------------------------- #
def derive_2d_window(td, ext, region, cartesian, mirror, W, H):
    """Return (umin,umax,vmin,vmax) -- the aspect-expanded world window mapped
    onto the WxH image, exactly as metadomain_render.cpp derives it."""
    x1lo, x1hi = region[0][0], region[0][1]
    x2lo, x2hi = region[1][0], region[1][1]
    if cartesian:
        umin, umax, vmin, vmax = x1lo, x1hi, x2lo, x2hi
    else:
        # meridional (X = r sin th, Z = r cos th) bbox of the wedge, sampling the
        # boundary (arcs at r={x1lo,x1hi}, rays at theta={x2lo,x2hi}).
        umin, umax = 1e30, -1e30
        vmin, vmax = 1e30, -1e30
        NB = 65

        def accXZ(r, th):
            nonlocal umin, umax, vmin, vmax
            X = r * math.sin(th)
            Z = r * math.cos(th)
            umin = min(umin, X); umax = max(umax, X)
            vmin = min(vmin, Z); vmax = max(vmax, Z)
            if mirror:
                umin = min(umin, -X); umax = max(umax, -X)

        for k in range(NB):
            t = k / (NB - 1)
            th = x2lo + (x2hi - x2lo) * t
            rr = x1lo + (x1hi - x1lo) * t
            accXZ(x1lo, th); accXZ(x1hi, th)
            accXZ(rr, x2lo); accXZ(rr, x2hi)

    # expand the window to the image aspect (centered) so geometry isn't stretched
    waspect = (umax - umin) / (vmax - vmin)
    iaspect = float(W) / float(H)
    if iaspect > waspect:
        cu = 0.5 * (umin + umax)
        hu = 0.5 * (vmax - vmin) * iaspect
        umin, umax = cu - hu, cu + hu
    else:
        cv = 0.5 * (vmin + vmax)
        hv = 0.5 * (umax - umin) / iaspect
        vmin, vmax = cv - hv, cv + hv

    # spherical slices get a 1.12x background border so the round outline + labels
    # are not clipped (Cartesian fills the frame and needs none).
    if not cartesian:
        pad = 1.12
        cu = 0.5 * (umin + umax); hu = 0.5 * (umax - umin) * pad
        cv = 0.5 * (vmin + vmax); hv = 0.5 * (vmax - vmin) * pad
        umin, umax = cu - hu, cu + hu
        vmin, vmax = cv - hv, cv + hv

    return umin, umax, vmin, vmax


def draw_2d_cartesian(td, ext, region, has_region, W, H, out_path, sim_name):
    umin, umax, vmin, vmax = derive_2d_window(td, ext, region, True, False, W, H)
    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)

    # aspect-expanded slice window (the background-padded frame)
    ax.add_patch(Rectangle((umin, vmin), umax - umin, vmax - vmin,
                           fill=False, ec="0.7", lw=1.0, ls="--",
                           label="slice window (aspect-expanded)"))
    # full domain box
    ax.add_patch(Rectangle((ext[0][0], ext[1][0]),
                           ext[0][1] - ext[0][0], ext[1][1] - ext[1][0],
                           fill=False, ec="0.4", lw=1.5, label="domain"))
    # region crop
    if has_region:
        ax.add_patch(Rectangle((region[0][0], region[1][0]),
                               region[0][1] - region[0][0],
                               region[1][1] - region[1][0],
                               fill=False, ec="tab:blue", lw=2.0,
                               label="region crop"))

    # ticks (nice numbers over the data box == region)
    axes_on = find_or(td, False, "output", "render", "axes")
    nticks = int(find_or(td, 5, "output", "render", "axis_ticks"))
    if axes_on:
        for tv in nice_ticks(region[0][0], region[0][1], nticks):
            ax.axvline(tv, color="0.85", lw=0.5, zorder=0)
        for tv in nice_ticks(region[1][0], region[1][1], nticks):
            ax.axhline(tv, color="0.85", lw=0.5, zorder=0)

    ax.set_xlim(umin, umax)
    ax.set_ylim(vmin, vmax)  # +v up (Cartesian slice: y is up in world)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{sim_name}: 2D Cartesian slice preview", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def draw_2d_spherical(td, ext, region, has_region, mirror, W, H, out_path, sim_name):
    umin, umax, vmin, vmax = derive_2d_window(td, ext, region, False, mirror, W, H)
    # (r, theta) wedge of the render region
    rmin, rmax = region[0][0], region[0][1]
    tmin, tmax = region[1][0], region[1][1]

    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)

    def wedge_boundary(rmn, rmx, tmn, tmx, sign, color, lw, label=None):
        # outer + inner arcs and two rays, in meridional (X=r sin th, Z=r cos th)
        th = np.linspace(tmn, tmx, 200)
        # outer arc
        ax.plot(sign * rmx * np.sin(th), rmx * np.cos(th), color=color, lw=lw,
                label=label)
        # inner arc
        ax.plot(sign * rmn * np.sin(th), rmn * np.cos(th), color=color, lw=lw)
        # rays at tmin, tmax
        for tt in (tmn, tmx):
            ax.plot([sign * rmn * math.sin(tt), sign * rmx * math.sin(tt)],
                    [rmn * math.cos(tt), rmx * math.cos(tt)], color=color, lw=lw)

    # full extent wedge (light gray)
    wedge_boundary(ext[0][0], ext[0][1], ext[1][0], ext[1][1], 1.0, "0.6", 1.2,
                   label="full extent")
    if mirror:
        wedge_boundary(ext[0][0], ext[0][1], ext[1][0], ext[1][1], -1.0, "0.6", 1.2)

    # region wedge (colored) if cropped
    if has_region:
        wedge_boundary(rmin, rmax, tmin, tmax, 1.0, "tab:blue", 2.0,
                       label="region crop")
        if mirror:
            wedge_boundary(rmin, rmax, tmin, tmax, -1.0, "tab:blue", 2.0)

    # radial ticks along the symmetry axis (X=0)
    axes_on = find_or(td, False, "output", "render", "axes")
    nticks = int(find_or(td, 5, "output", "render", "axis_ticks"))
    if axes_on:
        for Rv in nice_ticks(0.0, ext[0][1], nticks):
            ax.plot(0.0, Rv, marker="+", color="0.3", ms=6)
            ax.annotate(f"{Rv:g}", (0.0, Rv), fontsize=6, color="0.25",
                        xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center")

    ax.set_xlim(umin, umax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("X = r sin(theta)")
    ax.set_ylabel("Z = r cos(theta)")
    m = "mirrored" if mirror else "half-plane"
    ax.set_title(f"{sim_name}: 2D spherical meridional preview ({m})", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Data-free preview of the entity in-situ renderer scene geometry.")
    ap.add_argument("toml", help="simulation .toml file")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <toml_dir>/<simname>_preview.png)")
    ap.add_argument("--time", type=float, default=0.0,
                    help="sim time T for the moving-view pan (default 0)")
    ap.add_argument("--scene", type=int, default=None,
                    help="scene index (accepted for parity; geometry is scene-"
                         "independent, so it only affects the reported label)")
    args = ap.parse_args()

    if not os.path.isfile(args.toml):
        print(f"error: no such file: {args.toml}", file=sys.stderr)
        return 2

    with open(args.toml, "rb") as f:
        td = tomllib.load(f)

    # renderer enabled?
    if not find_or(td, False, "output", "render", "enable"):
        print("note: [output.render].enable is false in this toml; previewing anyway.")

    sim_name = find_or(td, "sim", "simulation", "name")
    width = int(find_or(td, 1024, "output", "render", "width"))
    height = int(find_or(td, 1024, "output", "render", "height"))
    mirror = bool(find_or(td, True, "output", "render", "mirror"))

    ext, dim, cartesian, metric_name = global_extent(td)

    # output path
    if args.out:
        out_path = args.out
    else:
        toml_dir = os.path.dirname(os.path.abspath(args.toml))
        out_path = os.path.join(toml_dir, f"{sim_name}_preview.png")

    # region + camera (region drives the default framing)
    region, has_region = resolve_region(td, ext)
    # camera is only meaningful in 3D, but building it is cheap & shares the pan
    cam = Camera(td, region, width, height)
    region = apply_moving_view(td, region, ext, cam, args.time, dim)
    # NB: the C++ pans the eye but keeps forward/right/up/half_h; we already
    # panned cam.eye, and the ortho_height / basis are region-independent after
    # the initial framing, so cam is now consistent with the panned region.

    # ---- summary line -------------------------------------------------------
    def fmt_pairs(pairs):
        return "[" + ", ".join(f"({p[0]:g},{p[1]:g})" for p in pairs) + "]"

    if dim == 3 and cartesian:
        mode = "3D box (Cartesian volume)"
    elif dim == 3:
        mode = "3D (non-Cartesian: unsupported by renderer)"
    elif dim == 2 and cartesian:
        mode = "2D Cartesian slice"
    elif dim == 2:
        mode = "2D spherical meridional slice"
    else:
        mode = "1D (nothing to render)"

    eye_str = f"({cam.eye[0]:g},{cam.eye[1]:g},{cam.eye[2]:g})"
    print(f"mode={mode} | metric={metric_name} | eye={eye_str} | "
          f"ortho_height={cam.ortho_height:g} | region={fmt_pairs(region)}")
    if args.scene is not None:
        print(f"  (scene index {args.scene} requested; geometry is scene-independent)")

    # ---- dispatch -----------------------------------------------------------
    if dim == 3 and cartesian:
        draw_3d(td, ext, region, has_region, cam, width, height, out_path, sim_name)
    elif dim == 3:
        print("warning: 3D non-Cartesian is not a renderer mode (3D is Cartesian-"
              "only); nothing drawn.")
        return 1
    elif dim == 2 and cartesian:
        draw_2d_cartesian(td, ext, region, has_region, width, height, out_path,
                          sim_name)
    elif dim == 2:
        draw_2d_spherical(td, ext, region, has_region, mirror, width, height,
                          out_path, sim_name)
    else:
        print("warning: 1D run -- the renderer is inactive; nothing to preview.")
        return 1

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
