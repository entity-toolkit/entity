# `turbulence_density_lines`

Driven 3D turbulence (re-using the standard
[`turbulence`](../../turbulence) pgen) with a combined in-situ render via
[Ascent](https://ascent.readthedocs.io/) that **overlays** two
visualizations into one image per cycle:

1. A ray-traced **volume** render of the particle number density `N`.
2. **Magnetic field lines** traced from `(B1, B2, B3)` and rasterized
   as 3D tubes on top of the density volume.

Both plots live in the same Ascent scene, so they share a camera and
get composited into a single PNG:

```
turbulence_density_lines_0000.png
turbulence_density_lines_0001.png
...
```

## Files

- `turbulence_density_lines.toml` — input file.
- `ascent_actions.yaml`           — pipeline (composite_vector → streamline)
                                    + two-plot scene (volume + pseudocolor).
- `README.md`                     — this file.

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
./build/src/entity.xc -input pgens/examples/turbulence_density_lines/turbulence_density_lines.toml
```

PNGs are written into the `turbulence_density_lines/` simulation
directory.

## Toml options

```toml
[output.fields]
  quantities = ["B", "N"]    # B is the vector field; N is the moment
  mom_smooth = 1             # one smoothing pass for the density moment

[output.ascent]
  enable        = true
  actions_file  = "ascent_actions.yaml"
  fields        = ["B1", "B2", "B3", "N"]
  interval_time = 4.0
```

The four entries in `output.ascent.fields` are pushed to Ascent under
the matching Conduit field names. `B1`/`B2`/`B3` are then composited
into a vector inside the actions file; `N` is rendered directly.

## Tweaking the look

In `ascent_actions.yaml`:

- `p_density.min_value` / `max_value` — control the transparency
  transfer function for the volume render. Raising `min_value` makes
  weak density invisible (good for letting field lines show through);
  lowering `max_value` saturates the bright cores.
- `pl_lines.f2.params.num_seeds` — number of seeded streamlines.
- `pl_lines.f2.params.tube_size` — radius of the rendered tubes.
- `pl_lines.f2.params.num_steps` × `step_size` — total integration length.
- Use a different scalar to color the tubes: chain another filter
  (e.g. `vector_magnitude` on `B`) before `streamline` and reference
  the resulting scalar in `p_lines.field`.
