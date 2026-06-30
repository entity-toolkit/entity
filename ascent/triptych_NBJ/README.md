# `turbulence_triptych`

Driven 3D turbulence (re-using the standard
[`turbulence`](../../pgens/turbulence) pgen) with a three-panel in-situ
render via [Ascent](https://ascent.readthedocs.io/):

1. Particle number density `N`.
2. Magnetic-field magnitude `|B|`.
3. Current-density magnitude `|J|`.

Each panel is a separate ray-traced volume render with a shared camera
and cadence. Ascent v0.9.x has no built-in render viewport for tiling
multiple scenes inside one PNG, so each quantity is written to its own
PNG stream with a synchronized cycle index in the filename:

```
N_00000000.png       Bmag_00000000.png       Jmag_00000000.png
N_00000001.png       Bmag_00000001.png       Jmag_00000001.png
...
```

To get a single side-by-side image per cycle, tile them with
ImageMagick (or any other compositor):

```sh
cd turbulence_triptych/plots
for i in $(ls N_*.png | sed 's/N_\([0-9]*\)\.png/\1/'); do
  montage -tile 3x1 -geometry +4+0 \
    N_${i}.png Bmag_${i}.png Jmag_${i}.png \
    triptych_${i}.png
done
```

## Files

- `turbulence_triptych.toml` — input file (3D, periodic, driven antenna).
- `ascent_actions.yaml`      — pipelines for `|B|` and `|J|` plus three
                               volume-render scenes.
- `README.md`                — this file.

The C++ problem generator is reused from `pgens/turbulence/pgen.hpp`,
so no separate `pgen.hpp` is required in this directory.

## Build

Configure Entity with the `turbulence` pgen and Ascent enabled.
`Ascent_DIR` should point to the directory containing
`AscentConfig.cmake`:

```sh
cmake -S . -B build \
      -D pgen=turbulence \
      -D output=ON \
      -D ascent=ON \
      -D Ascent_DIR=/path/to/ascent/install/lib/cmake/ascent
cmake --build build -j
```

## Run

```sh
./build/src/entity.xc -input ascent/triptych_NBJ/turbulence_triptych.toml
```

Ascent writes PNGs into `turbulence_triptych/plots/`.

## Toml options of interest

```toml
[output.fields]
  quantities = ["N", "B", "J"]   # density + B and J vectors
  mom_smooth = 1                 # one smoothing pass for the moment

[output.ascent]
  enable        = true
  actions_file  = "ascent_actions.yaml"
  fields        = ["N", "B1", "B2", "B3", "J1", "J2", "J3"]
  interval_time = 4.0
```

`fields` lists the component scalars Entity publishes to Conduit. The
B1/B2/B3 and J1/J2/J3 entries are also auto-aliased into MCArray vectors
named `B` and `J` (see `src/output/ascent_writer.cpp`), but the
`composite_vector` filter in the actions file is what `vector_magnitude`
consumes here — the proven path for magnitude rendering in the
installed Ascent v0.9.x.

The Ascent cadence (`interval_time = 4.0`) is independent of the ADIOS
field-output cadence (`output.interval_time = 20.0`), so you'll get
many more PNGs than `.bp` dumps.

## Tweaking the render

In `ascent_actions.yaml`:

- `min_value` / `max_value` per scene control the volume-render
  transfer function. `|J|` in particular has a much tighter dynamic
  range than `|B|`; adjust if current sheets wash out or vanish.
- Swap `color_table.name` for any VTK-m colormap (`Cool to Warm`,
  `Spectral`, `Black-Body Radiation`, ...).
- Replace `volume` with `pseudocolor` to surface-render instead of
  ray-cast through the box.
- Camera/`image_width`/`image_height` are duplicated across scenes on
  purpose — keep them consistent so the panels tile cleanly.
