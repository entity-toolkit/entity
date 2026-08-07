# `turbulence_ascent`

Driven 3D MHD-like turbulence (re-using the standard
[`turbulence`](../../turbulence) pgen) with in-situ volume rendering of
the magnetic-field magnitude `|B|` via [Ascent](https://ascent.readthedocs.io/).

The setup is a 64³ periodic Cartesian box driven by the same antenna as
the original 2D `turbulence` example, but now in 3D. Each output cycle
the three magnetic-field components are published to Ascent; an Ascent
*pipeline* combines them into a vector field and derives the scalar
`Bmag = |B|`. A *volume* plot then ray-casts `Bmag` to produce one PNG
per render cycle.

```
turbulence_ascent_Bmag_0000.png
turbulence_ascent_Bmag_0001.png
...
```

## Files

- `turbulence_ascent.toml` — input file (3D, periodic, driven antenna).
- `ascent_actions.yaml`    — Ascent pipeline + volume-render scene.
- `README.md`              — this file.

The C++ problem generator itself is reused from `pgens/turbulence/pgen.hpp`,
so no separate `pgen.hpp` is required in this directory.

## Build

Configure Entity with the existing `turbulence` pgen and Ascent enabled.
`Ascent_DIR` should point to the directory containing `AscentConfig.cmake`:

```sh
cmake -S . -B build \
      -D pgen=turbulence \
      -D output=ON \
      -D ascent=ON \
      -D Ascent_DIR=/path/to/ascent/install/lib/cmake/ascent
cmake --build build -j
```

For a quicker run, drop `ppc0` further or shrink the grid in the toml.

## Run

```sh
./build/src/entity.xc -input pgens/examples/turbulence_ascent/turbulence_ascent.toml
```

Ascent writes PNGs into the `turbulence_ascent/` simulation directory.

## Toml options of interest

```toml
[output.ascent]
  enable        = true
  actions_file  = "ascent_actions.yaml"
  fields        = ["B1", "B2", "B3"]
  interval_time = 4.0
```

`fields` lists the component scalars Entity publishes to Conduit. The
actions file is responsible for any derived quantities (here `Bmag`).
The Ascent cadence (`interval_time = 4.0`) is independent of the ADIOS
field-output cadence (`output.fields.interval_time` falls back to
`output.interval_time = 20.0`), so you'll get many more PNGs than `.bp`
dumps.

## Tweaking the render

To go from a single component to a different derived quantity, edit
`ascent_actions.yaml`. Useful filters in Ascent pipelines include:

- `composite_vector` — combine scalars into a vector field.
- `vector_magnitude` — `|v|` from a vector field.
- `vector_component` — extract one component of a vector field.
- `gradient` — spatial gradient of a scalar.
- `clip` / `slice` — geometric subset before rendering.

To switch the rendering style, change the `plots/p1/type` to e.g.
`pseudocolor` (surface) or keep `volume` and adjust `min_value` /
`max_value` to control transparency, or set `samples` to control the
ray-cast resolution.
