# v0.8

- [x] thick layer boundary for the monopole
- [x] test with filters
- [x] add diagnostics for nans in fields and particles
- [x] add gravitationally bound atmosphere
- [x] rewrite UniformInjector with global random pool
- [x] add particle deletion routine
- [x] make more user-friendly and understandable boundary conditions
- [x] refine output
- [ ] fix bugs in `nttiny`
  - [x] delete plot removes all labels
  - [x] add particles to `nttiny`
  - [x] state should also save skip interval
  - [ ] (?) add autosave state
  - [x] why last cell does not work?
- [x] add different moments (momX, momY, momZ, meanGamma)
- [x] add charge
- [x] add per species densities

# v0.9

- [x] add current deposit/filtering for GR
- [ ] add moments for GR
- [ ] add Maxwellian for GR

# v1.0

- [x] MPI configuration in cmake
- [x] MPI domain decomposition
- [x] MPI communications
  - [x] fields
  - [x] particles
  - [x] currents
- [x] MPI-aware output
- [x] MPI-aware diagnostics
- [ ] custom boundary conditions for particles and fields
- [x] better organization of problem generators

### Short term things to do/fix

  - [x] routine for easy side/corner range selection
  - [x] aliases for fields/particles/currents
  - [ ] check allocation of proper fields
  - [x] add a simple current filtering
  - [x] field mirrors
  - [x] unit tests + implement with github actions
  - [x] deployment scripts

### Intermediate term things to do/fix

  - [x] test curvilinear particle pusher
  - [x] particle motion near the axes
  - [x] test curvilinear current deposit
  - [x] deposition near the axes
  - [x] filtering near the axes

### Readability/modulability improvements

  - [x] make kernels accept views instead of meshblocks

### Performance improvements to try

- [ ] removing temporary variables in interpolation
- [ ] passing by value vs const ref in metric
- [ ] return physical coords one-by-one instead of by passing full vector