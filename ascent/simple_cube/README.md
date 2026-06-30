# `ascent_cube`

Minimal demonstration of the in-situ visualization interface to
[Ascent](https://ascent.readthedocs.io/) shipped with Entity.

The setup is a 3D periodic cube on a Cartesian Minkowski metric. The
problem generator initializes the magnetic field with a non-trivial third
component:

```
B1 = 0
B2 = 0
B3 = B0 * sin(2 pi x / Lx) * sin(2 pi y / Ly)
```

While the simulation runs, Entity publishes the rectilinear mesh and the
selected field components to Ascent every `output.fields.interval_time`.
Ascent reads `ascent_actions.yaml` and renders one PNG per cycle:

```
ascent_cube_B3_0000.png
ascent_cube_B3_0001.png
...
```

## Build

Configure Entity with Ascent enabled. Make sure `Ascent_DIR` points at
your Ascent install (which provides `AscentConfig.cmake`):

```sh
cmake -S . -B build \
      -D pgen=examples/ascent_cube \
      -D output=ON \
      -D ascent=ON \
      -D Ascent_DIR=/path/to/ascent/install/lib/cmake/ascent
cmake --build build -j
```

## Run

```sh
./build/src/entity.xc -input pgens/examples/ascent_cube/ascent_cube.toml
```

The PNGs are written next to the `ascent_cube/` simulation output
directory (controlled by `default_dir` in the writer).

## Toml options

In addition to the usual `[output]` keys, this example uses:

```toml
[output.ascent]
  enable        = true
  actions_file  = "ascent_actions.yaml"
  fields        = ["B3"]
  interval_time = 0.02   # Render cadence (independent of [output.fields])
```

`fields` lists the variable names that Entity should publish to Ascent.
Vector field components are postfixed with the index, so `B1`, `B2`,
`B3` give the three components of the magnetic field; `E1`/`E2`/`E3`
the electric field; `J1`/`J2`/`J3` the current. The names appear
verbatim as Conduit field names, so the actions file should reference
them under `field: "B3"`.

The `interval` / `interval_time` keys control how often Ascent renders;
they are completely independent of the ADIOS field-write cadence in
`[output.fields]`. If both keys are omitted, the global `[output]`
cadence is used.
