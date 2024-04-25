# v0.8

- [x] thick layer boundary for the monopole
- [x] test with filters
- [x] add diagnostics for nans in fields and particles
- [x] add gravitationally bound atmosphere
- [x] rewrite UniformInjector with global random pool
- [x] add particle deletion routine
- [x] make more user-friendly and understandable boundary conditions
- [x] refine output
- [x] add different moments (momX, momY, momZ, meanGamma)
- [x] add charge
- [x] add per species densities

# v0.9

- [x] add current deposit/filtering for GR
- [x] add moments for GR
- [x] add Maxwellian for GR

# v1.0.0

- [ ] custom boundary conditions for particles and fields
- [ ] transfer GR from v0.9
- [ ] particle output
- [ ] BUG in MPI particles/currents

### Performance improvements to try

- [ ] removing temporary variables in interpolation
- [ ] passing by value vs const ref in metric
- [ ] return physical coords one-by-one instead of by passing full vector