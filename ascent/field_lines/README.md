# `turbulence_lines`

Driven 3D turbulence (re-using the standard
[`turbulence`](../../turbulence) pgen) with in-situ rendering of magnetic
**field lines** via [Ascent](https://ascent.readthedocs.io/).

The same setup as
[`turbulence_ascent`](../turbulence_ascent), but instead of volume-rendering
`|B|`, this example traces the magnetic field with Ascent's `streamline`
filter from a uniform grid of seed points inside the box and rasterizes
the resulting polylines as 3D tubes. One PNG is produced per render
cycle:

```
turbulence_lines_0000.png
turbulence_lines_0001.png
...
```

## Files

- `turbulence_lines.toml` — input file (3D, periodic, driven antenna).
- `ascent_actions.yaml`   — Ascent pipeline (composite_vector → streamline)
                            + ray-traced pseudocolor scene.
- `README.md`             — this file.

The C++ problem generator is reused from `pgens/turbulence/pgen.hpp`.

## Build

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
./build/src/entity.xc -input pgens/examples/turbulence_lines/turbulence_lines.toml
```

Ascent writes PNGs into the `turbulence_lines/` simulation directory.

## Tweaking the field-line render

The interesting knobs all live in `ascent_actions.yaml`:

- `num_steps` × `step_size` controls how far each field line is integrated
  in code units. The default `200 × 0.5 = 100` covers ~1.5× the box width.
- `rendering/tube_size` sets the rendered tube radius. Reduce it
  (e.g. `0.2`) for a finer-looking line, or increase it for a thicker,
  easier-to-see line.
- `seeds.num_seeds_x` / `num_seeds_y` / `num_seeds_z` set how many seeds
  are placed along each axis of the seeding box (for `sampling_type:
  "uniform"`). The example uses a 4×4×4 grid (64 seeds total). Switch to
  `sampling_type: "random"` to draw `num_seeds` random points instead.
- `seeds.extents_x` / `extents_y` / `extents_z` define the seeding box.
- Other supported `seeds.type` values are `point`, `point_list`, `line`
  — see the Ascent `streamline` filter docs.
- To color the tubes by some derived scalar, add another filter to the
  pipeline (e.g. `vector_magnitude` on `B`) before the streamline filter
  and reference the scalar in the plot's `field` key.
