#!/usr/bin/env python3
"""Recommend the team tile size (T_TILE) for entity's tiled current-deposit kernel.

Each GPU work-group (team) owns a TE^dim scratch tile in shared memory (SLM on Intel,
LDS on AMD, shared mem on NVIDIA), accumulates its particles' currents into it, then
flushes once to global memory.  With

    TE   = T_TILE + 2*HALO
    HALO = stencil_reach + drift   (stencil_reach = shape_order for Esirkepov, 2 for the
                                    O==0 zigzag deposit; drift = the compile-time
                                    `team_policy_drift` CMake knob, NOT the runtime
                                    spatial_sorting_interval -- see currents_deposit.hpp)

the tile size is squeezed by three competing pressures:

  * Shared-memory capacity (HARD): TE^dim * ncomp * sizeof(real) must fit, ideally with
    several work-groups resident per compute unit so latency is hidden.  Binds in 3D /
    double precision / on AMD's 64 KiB LDS.
  * Halo overhead (push LARGER): zero-fill + flush sweep the whole TE^dim tile, so a tiny
    tile is almost all halo (1-(T/TE)^dim wasted).  Big HALO (infrequent sorts) makes this
    worse and forces larger tiles.
  * Particles per tile (push SMALLER): ppc*T^dim particles all atomic-add into one fixed
    scratch tile -> SLM-atomic contention and load imbalance grow with tile size.

Recommendation = the largest tile that respects the particle budget and shared-memory
residency; if that tile would be mostly halo, it is grown (toward lower halo) up to the
shared-memory limit.  This is a first-order model -- confirm by sweeping the entity knobs
  -D team_policy_tile_size=<T> -D team_policy_drift=<D>  and re-profiling (see roofline/).
The team (work-group) size defaults to Kokkos::AUTO; override it at runtime with the
  [algorithms.deposit] team_policy_team_size = <N>   (0 = AUTO)
toml knob -- clamped to the backend maximum at launch (engines/srpic/currents.h).

Two ways to drive it:

  * Interactive TUI (no arguments):
        python ideal_tile_size.py

  * Scriptable CLI (any argument):
        python ideal_tile_size.py pvc --dim 2 --ppc 16
        python ideal_tile_size.py amd --dim 3 --ppc 64
        python ideal_tile_size.py all --dim 2 --ppc 16 --drift 4   # infrequent sorting
"""
import argparse
import curses
import os
import sys
from typing import Callable, List, Optional, Tuple

# Shared-memory budget is the load-bearing number.  Per compute unit (Xe-core / SM / CU);
# smem_wg_max is the largest single-work-group allocation.
ARCH = {
    "pvc":    dict(label="Intel Data Center GPU Max 1550 (PVC, Xe-HPC), per tile",
                   smem_cu=128 * 1024, smem_wg_max=128 * 1024,
                   subgroup=32, max_wg=1024, n_cu=64, cu="Xe-core"),
    "a100":   dict(label="NVIDIA A100 (Ampere)",
                   smem_cu=164 * 1024, smem_wg_max=163 * 1024,
                   subgroup=32, max_wg=1024, n_cu=108, cu="SM",
                   note="shared mem >48 KiB/block needs opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize)"),
    "h100":   dict(label="NVIDIA H100 (Hopper)",
                   smem_cu=228 * 1024, smem_wg_max=227 * 1024,
                   subgroup=32, max_wg=1024, n_cu=132, cu="SM",
                   note="shared mem >48 KiB/block needs opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize)"),
    "gh200":  dict(label="NVIDIA GH200 Grace Hopper (Hopper H100/H200 GPU)",
                   smem_cu=228 * 1024, smem_wg_max=227 * 1024,
                   subgroup=32, max_wg=1024, n_cu=132, cu="SM",
                   note="shared mem >48 KiB/block needs opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize)"),
    "mi250x": dict(label="AMD Instinct MI250X (CDNA2), per GCD",
                   smem_cu=64 * 1024, smem_wg_max=64 * 1024,
                   subgroup=64, max_wg=1024, n_cu=110, cu="CU"),
    "mi300x": dict(label="AMD Instinct MI300X (CDNA3)",
                   smem_cu=64 * 1024, smem_wg_max=64 * 1024,
                   subgroup=64, max_wg=1024, n_cu=304, cu="CU"),
}
ALIAS = {"nvidia": "h100", "amd": "mi300x", "intel": "pvc", "mi250": "mi250x", "mi300": "mi300x",
         "gracehopper": "gh200", "grace-hopper": "gh200", "gh200x": "gh200"}
PRECISION = {"single": 4, "double": 8}

# arch choices offered in the TUI (canonical keys plus the meta-target "all")
ARCH_CHOICES = list(ARCH.keys()) + ["all"]
# "all" expands to one representative of each vendor (matches the CLI behaviour)
ALL_ARCHS = ["pvc", "nvidia", "amd"]


# ============================
# core model (shared by TUI + CLI)
# ============================

class Settings:
    """Tunable inputs for the tile-size model.

    Attribute names match the argparse dest names so `recommend`/`report_lines` accept
    either a Settings instance (TUI) or an argparse namespace (CLI) interchangeably.
    """

    def __init__(self):
        self.arch = "pvc"          # an ARCH key or "all"
        self.dim = 2               # 1 / 2 / 3
        self.ppc = 16.0            # particles per cell (per species)
        self.shape_order = 2       # entity shape_order
        self.precision = "single"  # single / double
        self.components = 3        # current-field components (J has 3)
        self.drift = 1             # team_policy_drift: cells of drift the scratch halo absorbs
                                   # (compile-time CMake knob, independent of spatial_sorting_interval)
        self.target_resident = 2   # work-groups resident per compute unit
        self.npart_cap = 1600.0    # particle-per-tile budget (contention / load-balance proxy)
        self.halo_max = 0.70       # halo fraction above which the tile is grown
        self.grid = 0              # cells per dim (0 disables the GPU-fill check)
        self.balance_factor = 4    # min tiles per compute unit
        self.min_tile = 4          # entity's team_policy_tile_sizes list starts at 4
        self.max_tile = 64


def resolve_arch(name):
    key = ALIAS.get(name.lower(), name.lower())
    if key not in ARCH:
        raise SystemExit("unknown arch '%s'; choose from %s (or aliases %s)"
                         % (name, ", ".join(ARCH), ", ".join(ALIAS)))
    return key, ARCH[key]


def recommend(hw, p):
    """p: Settings or argparse namespace. Returns dict with rows, chosen row, binding."""
    # Matches DepositCurrentsTiled_kernel: STENCIL_REACH = O for Esirkepov (O>=1),
    # 2 for the O==0 zigzag deposit; HALO = STENCIL_REACH + TEAM_POLICY_DRIFT.
    stencil_reach = 2 if p.shape_order == 0 else p.shape_order
    halo = stencil_reach + p.drift
    real = PRECISION[p.precision]
    atoms_pp = p.components * (p.shape_order + 1) ** p.dim   # ~ useful atomics / particle

    rows = []
    for T in range(p.min_tile, p.max_tile + 1, 2):          # entity uses even tile sizes
        TE = T + 2 * halo
        scratch = TE ** p.dim * p.components * real
        npart = p.ppc * T ** p.dim
        halo_frac = 1.0 - (T / TE) ** p.dim
        ovhd = (2.0 * p.components / p.ppc) * (TE / T) ** p.dim / atoms_pp  # zero+flush vs deposit
        resident = int(hw["smem_cu"] // scratch) if scratch else 0
        ntiles = (p.grid / T) ** p.dim if p.grid else None
        rows.append(dict(T=T, TE=TE, scratch=scratch, npart=npart, halo_frac=halo_frac,
                         ovhd=ovhd, resident=resident, ntiles=ntiles,
                         ok_cap=scratch <= hw["smem_wg_max"]))

    capfeas = [r for r in rows if r["ok_cap"]]
    if not capfeas:
        return dict(halo=halo, rows=rows, chosen=None, binding=None, grown=False)

    def largest(pred, default):
        ts = [r["T"] for r in capfeas if pred(r)]
        return max(ts) if ts else default

    T_cap = max(r["T"] for r in capfeas)
    T_res = largest(lambda r: r["resident"] >= p.target_resident, p.min_tile)
    T_np = largest(lambda r: r["npart"] <= p.npart_cap, p.min_tile)
    T_bal = largest(lambda r: r["ntiles"] is None or r["ntiles"] >= p.balance_factor * hw["n_cu"], p.min_tile)
    bounds = {"shared-memory capacity": T_cap, "shared-memory residency": T_res,
              "GPU fill (too few tiles)": T_bal, "particle budget per tile": T_np}

    chosen_T = max(min(bounds.values()), p.min_tile)
    binding = min(bounds, key=lambda k: bounds[k])

    # If the particle-budget pick is mostly halo, grow the tile to cut halo, but never
    # past what shared memory / GPU-fill allow (that just trades halo for contention).
    grown = False
    cur = next(r for r in capfeas if r["T"] == chosen_T)
    if cur["halo_frac"] > p.halo_max:
        halo_ok = [r["T"] for r in capfeas if r["halo_frac"] <= p.halo_max]
        ceil_T = min(T_res, T_bal, T_cap)
        if halo_ok:
            target = max(min(halo_ok), chosen_T)        # smallest tile that clears halo_max
            new_T = min(max(target, chosen_T), ceil_T)
            if new_T > chosen_T:
                chosen_T, grown = new_T, True
                binding = ("halo overhead" if min(halo_ok) <= ceil_T
                           else min({k: v for k, v in bounds.items()
                                     if k != "particle budget per tile"},
                                    key=lambda k: bounds[k]))
        else:
            new_T = min(ceil_T, T_cap)                  # can't clear halo_max at all -> go as big as SLM allows
            if new_T > chosen_T:
                chosen_T, grown = new_T, True
                binding = "shared-memory residency"

    chosen = next(r for r in capfeas if r["T"] == chosen_T)
    return dict(halo=halo, rows=rows, chosen=chosen, binding=binding, grown=grown, atoms_pp=atoms_pp)


def kib(b):
    return "%.1f" % (b / 1024.0)


def report_lines(name, key, hw, p, res):
    """Build the recommendation report as a list of text lines (no printing)."""
    L = []
    L.append("=" * 80)
    L.append("%s   [preset: %s]" % (hw["label"], key))
    L.append("  dim=%d  ppc=%g  shape_order=%d  precision=%s(%dB)  J-components=%d"
             % (p.dim, p.ppc, p.shape_order, p.precision, PRECISION[p.precision], p.components))
    reach = 2 if p.shape_order == 0 else p.shape_order
    reach_kind = "zigzag" if p.shape_order == 0 else "Esirkepov O"
    L.append("  HALO = stencil_reach + drift = %d + %d = %d   ->   TE = T_TILE + %d"
             "   (reach %d = %s; drift = team_policy_drift)"
             % (reach, p.drift, res["halo"], 2 * res["halo"], reach, reach_kind))
    L.append("  shared mem %s KiB/%s (budget %s KiB for %d resident WGs); subgroup=%d, n_cu=%d"
             % (kib(hw["smem_cu"]), hw["cu"], kib(hw["smem_cu"] / p.target_resident),
                p.target_resident, hw["subgroup"], hw["n_cu"]))
    if hw.get("note"):
        L.append("  note: %s" % hw["note"])
    L.append("-" * 80)
    L.append("  T_TILE  TE   scratch  resWG  part/tile  halo%  zero+flush%")
    for r in res["rows"]:
        if not r["ok_cap"]:
            continue
        mark = "  <== recommended" if r is res["chosen"] else ""
        L.append("   %3d   %4d  %6s K  %4d  %9.0f  %4.0f   %7.1f%s"
                 % (r["T"], r["TE"], kib(r["scratch"]), r["resident"], r["npart"],
                    100 * r["halo_frac"], 100 * r["ovhd"], mark))
    L.append("-" * 80)

    c = res["chosen"]
    if c is None:
        L.append("  INFEASIBLE: even T_TILE=%d does not fit %s KiB shared memory."
                 % (p.min_tile, kib(hw["smem_wg_max"])))
        L.append("  -> sort more often (smaller drift), use precision single, lower shape_order,")
        L.append("     or use a non-tiled (global-atomic / ScatterView) deposit on this arch.")
        return L
    L.append("  RECOMMENDED  T_TILE = %d   (limited by: %s%s)"
             % (c["T"], res["binding"], "; tile grown to reduce halo" if res["grown"] else ""))
    L.append("    %.1f KiB scratch/team, %d work-groups resident/%s, %.0f particles/team, %.0f%% halo"
             % (c["scratch"] / 1024.0, c["resident"], hw["cu"], c["npart"], 100 * c["halo_frac"]))
    team = min(hw["max_wg"], 256 - 256 % hw["subgroup"])
    extra = "" if c["T"] <= 16 else "   (entity's team_policy_tile_sizes list stops at 16; extend it)"
    L.append("    entity build:  -D team_policy=ON -D team_policy_tile_size=%d -D team_policy_drift=%d%s"
             % (min(c["T"], 16), p.drift, extra))
    L.append("    team (work-group) size: Kokkos::AUTO by default; to override, set in the toml")
    L.append("    [algorithms.deposit] team_policy_team_size = %d   (0 = AUTO; keep a multiple of"
             % team)
    L.append("    subgroup=%d), then sweep around it and re-profile" % hw["subgroup"])
    # contextual guidance
    if c["halo_frac"] > p.halo_max:
        if p.drift > 1:
            L.append("    !! %.0f%% of the tile is halo, inflated by team_policy_drift=%d; lower it "
                     "(and sort at least that often via spatial_sorting_interval)"
                     % (100 * c["halo_frac"], p.drift))
        else:
            L.append("    !! %.0f%% halo is intrinsic at this size (shared memory caps the tile here)"
                     % (100 * c["halo_frac"]))
    if c["npart"] > p.npart_cap:
        L.append("    !! %.0f particles/team exceeds the %.0f budget -> watch SLM-atomic contention"
                 % (c["npart"], p.npart_cap))
    if c["resident"] < p.target_resident:
        L.append("    !! only %d work-group(s) resident/%s -> limited latency hiding"
                 % (c["resident"], hw["cu"]))
    return L


def archs_for(name):
    """Expand a setting/arg value into the list of arch names to report on."""
    return ALL_ARCHS if name.lower() == "all" else [name]


def build_report(p):
    """Run the model for the selected arch(es) and return the full report as lines."""
    lines = []
    for a in archs_for(p.arch):
        try:
            key, hw = resolve_arch(a)
        except SystemExit as e:
            lines.append(str(e))
            continue
        lines.extend(report_lines(a, key, hw, p, recommend(hw, p)))
    lines.append("=" * 80)
    return lines


# ============================
# colors: edit these
# ============================

COLOR_TITLE_FG = curses.COLOR_BLUE
COLOR_TEXT_FG = curses.COLOR_WHITE
COLOR_SELECTED_FG = curses.COLOR_WHITE
COLOR_SELECTED_BG = curses.COLOR_BLACK
COLOR_HINT_FG = curses.COLOR_YELLOW
COLOR_OK_FG = curses.COLOR_GREEN
COLOR_ERR_FG = curses.COLOR_RED
COLOR_KEY_FG = curses.COLOR_MAGENTA
COLOR_DIM_FG = curses.COLOR_CYAN

PAIR_TITLE = 1
PAIR_TEXT = 2
PAIR_SELECTED = 3
PAIR_HINT = 4
PAIR_OK = 5
PAIR_ERR = 6
PAIR_KEY = 7
PAIR_DIM = 8


class MenuItem:
    def __init__(self, label, hint="", right=None, on_enter=None,
                 on_space=None, disabled=None):
        self.label = label
        self.hint = hint
        self.right = right
        self.on_enter = on_enter
        self.on_space = on_space
        self.disabled = disabled


# ============================
# TUI
# ============================

class App:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.s = Settings()

        self.state = "mainmenu"
        self.stack: List[Tuple[str, int]] = []
        self.selected = 0
        self.scroll = 0
        self.message = "use arrows or j/k"

        self._init_curses()

    def _init_curses(self) -> None:
        curses.curs_set(0)
        self.stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()

        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(PAIR_TITLE, COLOR_TITLE_FG, -1)
            curses.init_pair(PAIR_TEXT, COLOR_TEXT_FG, -1)
            curses.init_pair(PAIR_SELECTED, COLOR_SELECTED_FG, COLOR_SELECTED_BG)
            curses.init_pair(PAIR_HINT, COLOR_HINT_FG, -1)
            curses.init_pair(PAIR_OK, COLOR_OK_FG, -1)
            curses.init_pair(PAIR_ERR, COLOR_ERR_FG, -1)
            curses.init_pair(PAIR_KEY, COLOR_KEY_FG, -1)
            curses.init_pair(PAIR_DIM, COLOR_DIM_FG, -1)

    def cp(self, pair_id: int) -> int:
        return curses.color_pair(pair_id) if curses.has_colors() else 0

    # ----- formatting helpers -----

    def arch_label(self) -> str:
        if self.s.arch == "all":
            return "all (%s)" % " + ".join(ALL_ARCHS)
        try:
            return ARCH[resolve_arch(self.s.arch)[0]]["label"]
        except SystemExit:
            return self.s.arch

    # ----- nav stack -----

    def push(self, st: str) -> None:
        self.stack.append((self.state, self.selected))
        self.state = st
        self.selected = 0
        self.scroll = 0
        self.message = ""

    def pop(self) -> None:
        if self.stack:
            self.state, self.selected = self.stack.pop()
        else:
            self.state, self.selected = "mainmenu", 0
        self.scroll = 0
        self.message = ""

    # ----- drawing -----

    def add(self, y: int, x: int, s: str, attr: int = 0) -> None:
        try:
            self.stdscr.addstr(y, x, s, attr)
        except curses.error:
            pass

    def hline(self, y: int) -> None:
        _, w = self.stdscr.getmaxyx()
        try:
            self.stdscr.hline(y, 0, curses.ACS_HLINE, max(0, w - 1))
        except curses.error:
            pass

    def draw_keybar(self, y: int, x: int, pairs: List[Tuple[str, str]]) -> None:
        cur_x = x
        for key, action in pairs:
            self.add(y, cur_x, key, self.cp(PAIR_KEY) | curses.A_BOLD)
            cur_x += len(key)
            self.add(y, cur_x, " ", self.cp(PAIR_DIM))
            cur_x += 1
            self.add(y, cur_x, action, self.cp(PAIR_HINT))
            cur_x += len(action)
            self.add(y, cur_x, "   ", self.cp(PAIR_DIM))
            cur_x += 3

    def breadcrumb(self) -> str:
        return {
            "mainmenu": "mainmenu",
            "arch": "mainmenu > architecture",
            "physics": "mainmenu > physics & particles",
            "tuning": "mainmenu > tuning knobs",
        }.get(self.state, "mainmenu")

    def draw_menu(self, title: str, prompt: str, items: List[MenuItem]) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        self.add(0, 2, title, self.cp(PAIR_TITLE) | curses.A_BOLD)
        bc = self.breadcrumb()
        self.add(0, max(2, w - 2 - len(bc)), bc, self.cp(PAIR_DIM))

        self.draw_keybar(
            1,
            2,
            [
                ("up/dn/j/k", "move"),
                ("enter", "select"),
                ("space", "toggle/cycle"),
                ("b", "back"),
                ("q", "quit"),
            ],
        )
        self.hline(2)

        status1 = ("arch: %s    dim: %d    precision: %s    ppc: %g"
                   % (self.s.arch, self.s.dim, self.s.precision, self.s.ppc))
        status2 = ("shape_order: %d    drift: %d    components: %d    tile range: %d-%d"
                   % (self.s.shape_order, self.s.drift, self.s.components,
                      self.s.min_tile, self.s.max_tile))
        self.add(3, 2, status1[: w - 4], self.cp(PAIR_TEXT))
        self.add(4, 2, status2[: w - 4], self.cp(PAIR_TEXT))
        self.hline(5)

        self.add(6, 2, prompt[: w - 4], self.cp(PAIR_TEXT) | curses.A_BOLD)

        list_y = 8
        footer_h = 3
        view_h = max(1, h - list_y - footer_h)
        n = len(items)

        if n == 0:
            self.add(list_y, 2, "(empty)", self.cp(PAIR_HINT))
        else:
            self.selected = max(0, min(self.selected, n - 1))

            if self.selected < self.scroll:
                self.scroll = self.selected
            if self.selected >= self.scroll + view_h:
                self.scroll = self.selected - view_h + 1
            self.scroll = max(0, min(self.scroll, max(0, n - view_h)))

            shown = items[self.scroll : self.scroll + view_h]

            for i, it in enumerate(shown):
                idx = self.scroll + i
                sel = idx == self.selected
                dis = bool(it.disabled and it.disabled())

                row_attr = (
                    self.cp(PAIR_SELECTED) | curses.A_BOLD
                    if sel
                    else (self.cp(PAIR_DIM) if dis else self.cp(PAIR_TEXT))
                )
                self.add(list_y + i, 2, ("  %s" % it.label)[: w - 4], row_attr)

                if it.right:
                    rt = (it.right() or "").strip()
                    if rt:
                        rt = rt[: max(0, w - 6)]
                        x = max(2, w - 2 - len(rt))
                        rt_attr = (
                            row_attr
                            if sel
                            else (self.cp(PAIR_HINT) if not dis else self.cp(PAIR_DIM))
                        )
                        self.add(list_y + i, x, rt, rt_attr)

                if sel and it.hint:
                    self.add(
                        list_y + i,
                        min(w - 4, 30),
                        ("    %s" % it.hint)[: w - 4],
                        self.cp(PAIR_HINT),
                    )

        self.hline(h - 3)
        msg = self.message or ""
        if msg:
            is_err = msg.startswith("error")
            attr = (self.cp(PAIR_ERR) if is_err else self.cp(PAIR_OK)) | curses.A_BOLD
            self.add(h - 2, 2, msg[: w - 4], attr)
        self.stdscr.refresh()

    # ----- modals -----

    def input_box(self, title: str, prompt: str, initial: str) -> Optional[str]:
        h, w = self.stdscr.getmaxyx()
        win_h, win_w = 9, min(86, max(46, w - 6))
        top, left = max(0, (h - win_h) // 2), max(0, (w - win_w) // 2)

        win = curses.newwin(win_h, win_w, top, left)
        win.keypad(True)
        win.border()

        win.addstr(1, 2, title[: win_w - 4], self.cp(PAIR_TITLE) | curses.A_BOLD)
        win.addstr(2, 2, prompt[: win_w - 4], self.cp(PAIR_TEXT))

        buf = list(initial)
        curses.curs_set(1)

        while True:
            win.addstr(4, 2, " " * (win_w - 4), self.cp(PAIR_TEXT))
            text = "".join(buf)
            if len(text) > win_w - 4:
                text = text[-(win_w - 4) :]
            win.addstr(4, 2, text, self.cp(PAIR_TEXT) | curses.A_BOLD)
            win.addstr(6, 2, "enter=ok   esc=cancel", self.cp(PAIR_DIM))
            win.refresh()

            ch = win.getch()
            if ch == 27:
                curses.curs_set(0)
                return None
            if ch in (curses.KEY_ENTER, 10, 13):
                curses.curs_set(0)
                return "".join(buf).strip()
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                if buf:
                    buf.pop()
            elif 32 <= ch <= 126:
                buf.append(chr(ch))

    # ----- value editors -----

    def cycle_attr(self, attr: str, options: list) -> None:
        cur = getattr(self.s, attr)
        if cur not in options:
            setattr(self.s, attr, options[0])
        else:
            setattr(self.s, attr, options[(options.index(cur) + 1) % len(options)])

    def edit_int(self, label: str, attr: str, minv: Optional[int] = None) -> None:
        val = self.input_box(label, "enter an integer:", str(getattr(self.s, attr)))
        if val is None or val == "":
            return
        try:
            n = int(val)
        except ValueError:
            self.message = "error: '%s' is not an integer" % val
            return
        if minv is not None and n < minv:
            self.message = "error: %s must be >= %d" % (label, minv)
            return
        setattr(self.s, attr, n)
        self.message = "%s = %d" % (label, n)

    def edit_float(self, label: str, attr: str,
                   minv: Optional[float] = None, maxv: Optional[float] = None) -> None:
        val = self.input_box(label, "enter a number:", str(getattr(self.s, attr)))
        if val is None or val == "":
            return
        try:
            x = float(val)
        except ValueError:
            self.message = "error: '%s' is not a number" % val
            return
        if minv is not None and x < minv:
            self.message = "error: %s must be >= %g" % (label, minv)
            return
        if maxv is not None and x > maxv:
            self.message = "error: %s must be <= %g" % (label, maxv)
            return
        setattr(self.s, attr, x)
        self.message = "%s = %g" % (label, x)

    # ----- report pager -----

    def _pager_attr(self, ln: str) -> int:
        s = ln.strip()
        if "RECOMMENDED" in ln:
            return self.cp(PAIR_OK) | curses.A_BOLD
        if "<== recommended" in ln:
            return self.cp(PAIR_OK)
        if "INFEASIBLE" in ln or "unknown arch" in ln:
            return self.cp(PAIR_ERR) | curses.A_BOLD
        if s.startswith("!!"):
            return self.cp(PAIR_HINT)
        if s.startswith("T_TILE"):
            return self.cp(PAIR_TITLE) | curses.A_BOLD
        if set(s) <= {"=", "-"} and s:
            return self.cp(PAIR_DIM)
        return self.cp(PAIR_TEXT)

    def pager(self, title: str, lines: List[str]) -> None:
        top = 0
        note = ""
        while True:
            self.stdscr.erase()
            h, w = self.stdscr.getmaxyx()

            self.add(0, 2, title, self.cp(PAIR_TITLE) | curses.A_BOLD)
            self.draw_keybar(
                1,
                2,
                [
                    ("up/dn/j/k", "scroll"),
                    ("PgUp/PgDn", "page"),
                    ("g/G", "top/end"),
                    ("w", "save"),
                    ("b/q", "back"),
                ],
            )
            self.hline(2)

            list_y = 4
            view_h = max(1, h - list_y - 2)
            n = len(lines)
            top = max(0, min(top, max(0, n - view_h)))

            for i, ln in enumerate(lines[top : top + view_h]):
                self.add(list_y + i, 2, ln[: w - 3], self._pager_attr(ln))

            self.hline(h - 2)
            footer = "line %d-%d / %d" % (top + 1, min(top + view_h, n), n)
            if note:
                footer += "    " + note
            self.add(h - 1, 2, footer[: w - 4], self.cp(PAIR_DIM))
            self.stdscr.refresh()

            ch = self.stdscr.getch()
            if ch in (ord("q"), ord("Q"), ord("b"), 8, 127):
                return
            if ch in (curses.KEY_UP, ord("k"), ord("K")):
                top -= 1
            elif ch in (curses.KEY_DOWN, ord("j"), ord("J")):
                top += 1
            elif ch in (curses.KEY_PPAGE,):
                top -= view_h
            elif ch in (curses.KEY_NPAGE, ord(" ")):
                top += view_h
            elif ch in (ord("g"),):
                top = 0
            elif ch in (ord("G"),):
                top = n
            elif ch in (ord("w"), ord("W")):
                note = self._save_report(lines)

    def _save_report(self, lines: List[str]) -> str:
        path = os.path.join(os.getcwd(), "ideal_tile_size_report.txt")
        try:
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            return "saved to %s" % path
        except OSError as e:
            return "save failed: %s" % e

    def do_compute(self) -> None:
        self.pager("recommendation  [%s]" % self.s.arch, build_report(self.s))

    def reset(self) -> None:
        self.s = Settings()
        self.message = "reset to defaults"

    # ----- menus -----

    def menu_main(self) -> Tuple[str, str, List[MenuItem]]:
        return (
            "entity tile-size advisor",
            "main menu:",
            [
                MenuItem(
                    "architecture",
                    "choose the target GPU (or 'all')",
                    right=self.arch_label,
                    on_enter=lambda: self.push("arch"),
                ),
                MenuItem(
                    "physics & particles",
                    "dim, ppc, shape order, precision, ...",
                    right=lambda: "dim %d / ppc %g / so %d / %s"
                    % (self.s.dim, self.s.ppc, self.s.shape_order, self.s.precision),
                    on_enter=lambda: self.push("physics"),
                ),
                MenuItem(
                    "tuning knobs",
                    "residency, budgets, tile range, ...",
                    right=lambda: "res %d / cap %g / halo %.2f"
                    % (self.s.target_resident, self.s.npart_cap, self.s.halo_max),
                    on_enter=lambda: self.push("tuning"),
                ),
                MenuItem(
                    "compute recommendation",
                    "run the model and show the report",
                    on_enter=self.do_compute,
                ),
                MenuItem("reset to defaults", "", on_enter=self.reset),
                MenuItem("exit", "", on_enter=lambda: setattr(self, "state", "exit")),
            ],
        )

    def menu_arch(self) -> Tuple[str, str, List[MenuItem]]:
        def choose(name: str):
            self.s.arch = name
            self.pop()

        def label(name: str) -> str:
            mark = "(*)" if name == self.s.arch else "( )"
            return "%s %s" % (mark, name)

        def right(name: str) -> str:
            if name == "all":
                return " + ".join(ALL_ARCHS)
            return ARCH[name]["label"]

        items = [
            MenuItem(
                label(a),
                "select target",
                right=(lambda a=a: right(a)),
                on_enter=(lambda a=a: choose(a)),
            )
            for a in ARCH_CHOICES
        ]
        items.append(MenuItem("back", "return", on_enter=self.pop))
        return ("architecture", "pick the target GPU:", items)

    def menu_physics(self) -> Tuple[str, str, List[MenuItem]]:
        def cyc_dim():
            self.cycle_attr("dim", [1, 2, 3])

        def cyc_prec():
            self.cycle_attr("precision", ["single", "double"])

        return (
            "physics & particles",
            "set physical inputs:",
            [
                MenuItem(
                    "dim",
                    "space cycles: 1 / 2 / 3",
                    right=lambda: str(self.s.dim),
                    on_enter=cyc_dim,
                    on_space=cyc_dim,
                ),
                MenuItem(
                    "ppc",
                    "particles per cell (per species)",
                    right=lambda: "%g" % self.s.ppc,
                    on_enter=lambda: self.edit_float("ppc", "ppc", minv=0.0),
                ),
                MenuItem(
                    "shape_order",
                    "entity particle shape order",
                    right=lambda: str(self.s.shape_order),
                    on_enter=lambda: self.edit_int("shape_order", "shape_order", minv=0),
                ),
                MenuItem(
                    "precision",
                    "space cycles: single / double",
                    right=lambda: self.s.precision,
                    on_enter=cyc_prec,
                    on_space=cyc_prec,
                ),
                MenuItem(
                    "components",
                    "current-field components (J has 3)",
                    right=lambda: str(self.s.components),
                    on_enter=lambda: self.edit_int("components", "components", minv=1),
                ),
                MenuItem(
                    "drift",
                    "team_policy_drift CMake knob: cells the scratch halo absorbs (>= spatial_sorting_interval)",
                    right=lambda: str(self.s.drift),
                    on_enter=lambda: self.edit_int("drift", "drift", minv=0),
                ),
                MenuItem("back", "return", on_enter=self.pop),
            ],
        )

    def menu_tuning(self) -> Tuple[str, str, List[MenuItem]]:
        return (
            "tuning knobs",
            "set the model's budgets and ranges:",
            [
                MenuItem(
                    "target_resident",
                    "work-groups resident per compute unit",
                    right=lambda: str(self.s.target_resident),
                    on_enter=lambda: self.edit_int("target_resident", "target_resident", minv=1),
                ),
                MenuItem(
                    "npart_cap",
                    "particle-per-tile budget (contention proxy)",
                    right=lambda: "%g" % self.s.npart_cap,
                    on_enter=lambda: self.edit_float("npart_cap", "npart_cap", minv=1.0),
                ),
                MenuItem(
                    "halo_max",
                    "halo fraction above which the tile is grown (0..1)",
                    right=lambda: "%.2f" % self.s.halo_max,
                    on_enter=lambda: self.edit_float("halo_max", "halo_max", minv=0.0, maxv=1.0),
                ),
                MenuItem(
                    "grid",
                    "cells per dim (0 disables GPU-fill check)",
                    right=lambda: str(self.s.grid),
                    on_enter=lambda: self.edit_int("grid", "grid", minv=0),
                ),
                MenuItem(
                    "balance_factor",
                    "min tiles per compute unit",
                    right=lambda: str(self.s.balance_factor),
                    on_enter=lambda: self.edit_int("balance_factor", "balance_factor", minv=1),
                ),
                MenuItem(
                    "min_tile",
                    "smallest T_TILE to consider",
                    right=lambda: str(self.s.min_tile),
                    on_enter=lambda: self.edit_int("min_tile", "min_tile", minv=1),
                ),
                MenuItem(
                    "max_tile",
                    "largest T_TILE to consider",
                    right=lambda: str(self.s.max_tile),
                    on_enter=lambda: self.edit_int("max_tile", "max_tile", minv=1),
                ),
                MenuItem("back", "return", on_enter=self.pop),
            ],
        )

    def get_menu(self) -> Tuple[str, str, List[MenuItem]]:
        if self.state == "mainmenu":
            return self.menu_main()
        if self.state == "arch":
            return self.menu_arch()
        if self.state == "physics":
            return self.menu_physics()
        if self.state == "tuning":
            return self.menu_tuning()
        self.state = "mainmenu"
        return self.menu_main()

    # ----- navigation -----

    def is_disabled(self, it: MenuItem) -> bool:
        return bool(it.disabled and it.disabled())

    def move_sel(self, items: List[MenuItem], delta: int) -> None:
        if not items:
            return
        n = len(items)
        start = self.selected
        for _ in range(n):
            self.selected = (self.selected + delta) % n
            if not self.is_disabled(items[self.selected]):
                return
        self.selected = start

    def activate(self, items: List[MenuItem], enter: bool) -> None:
        if not items:
            return
        it = items[self.selected]
        if self.is_disabled(it):
            self.message = "error: option disabled."
            return
        fn = it.on_enter if enter else it.on_space
        if fn:
            fn()

    # ----- loop -----

    def run(self) -> None:
        while True:
            if self.state == "exit":
                return

            title, prompt, items = self.get_menu()
            self.draw_menu(title, prompt, items)

            ch = self.stdscr.getch()

            if ch in (ord("q"), ord("Q")):
                self.state = "exit"
                continue
            if ch in (ord("b"), 8, 127):
                self.pop()
                continue
            if ch in (curses.KEY_UP, ord("k"), ord("K")):
                self.move_sel(items, -1)
                continue
            if ch in (curses.KEY_DOWN, ord("j"), ord("J")):
                self.move_sel(items, +1)
                continue
            if ch in (curses.KEY_ENTER, 10, 13):
                self.activate(items, enter=True)
                continue
            if ch == ord(" "):
                self.activate(items, enter=False)
                continue


def run_tui() -> int:
    try:
        curses.wrapper(lambda stdscr: App(stdscr).run())
    except KeyboardInterrupt:
        return 130
    return 0


# ============================
# CLI (preserved for scripting / sweeps)
# ============================

def run_cli(argv) -> int:
    ap = argparse.ArgumentParser(description="Recommend entity team tile size (T_TILE).")
    ap.add_argument("arch", help="pvc | nvidia | amd  (or a100/h100/gh200/mi250x/mi300x, or 'all')")
    ap.add_argument("--dim", type=int, default=2, choices=(1, 2, 3))
    ap.add_argument("--ppc", type=float, default=16.0, help="particles per cell (per species)")
    ap.add_argument("--shape-order", type=int, default=2, help="particle shape order (entity shape_order)")
    ap.add_argument("--precision", choices=("single", "double"), default="single")
    ap.add_argument("--components", type=int, default=3, help="current-field components (J has 3)")
    ap.add_argument("--drift", type=int, default=1,
                    help="team_policy_drift CMake knob (compile-time): cells of drift the scratch "
                         "halo absorbs; size it >= spatial_sorting_interval")
    ap.add_argument("--target-resident", type=int, default=2, help="work-groups resident per compute unit")
    ap.add_argument("--npart-cap", type=float, default=1600,
                    help="particle-per-tile budget (SLM-atomic-contention / load-balance proxy)")
    ap.add_argument("--halo-max", type=float, default=0.70, help="halo fraction above which the tile is grown")
    ap.add_argument("--grid", type=int, default=0, help="cells per dim (optional; enables a GPU-fill check)")
    ap.add_argument("--balance-factor", type=int, default=4, help="min tiles per compute unit")
    ap.add_argument("--min-tile", type=int, default=4,
                    help="smallest T_TILE to consider (entity's team_policy_tile_sizes starts at 4)")
    ap.add_argument("--max-tile", type=int, default=64)
    p = ap.parse_args(argv)

    for ln in build_report(p):
        print(ln)
    return 0


def main() -> int:
    # no arguments -> interactive TUI; any argument -> scriptable CLI
    if len(sys.argv) > 1:
        return run_cli(sys.argv[1:])
    return run_tui()


if __name__ == "__main__":
    raise SystemExit(main())
