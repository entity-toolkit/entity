# `Entity` a.k.a. `ntt`
One particle-in-cell code to rule them all. Find our detailed documentation [here](https://haykh.github.io/entity/).

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Dev team (alphabetical)

💁‍♂️ __Alexander Chernoglazov__ {[@SChernoglazov](https://github.com/SChernoglazov): PIC}

🍵 __Benjamin Crinquand__ {[@bcrinquand](https://github.com/bcrinquand): GRPIC, cubed-sphere}

🧋 __Alisa Galishnikova__ {[@alisagk](https://github.com/alisagk): GRPIC}

☕ __Hayk Hakobyan__ {[@haykh](https://github.com/haykh): framework, PIC, GRPIC, cubed-sphere}

🥔 __Jens Mahlmann__ {[@jmahlmann](https://github.com/jmahlmann): framework, MPI, cubed-sphere}

🐬 __Sasha Philippov__ {[@sashaph](https://github.com/sashaph): all-around}

🤷 __Arno Vanthieghem__ {[@vanthieg](https://github.com/vanthieg): framework, PIC}

😺 __Muni Zhou__ {[@munizhou](https://github.com/munizhou): PIC}

## State of things

* Framework 
  - [ ]  Metrics
    - [x]  Minkowski (SR)
    - [x]  Spherical/Qspherical (SR)
    - [x]  Kerr-Schild/QKerr-Schild, zero-mass Kerr-Schild (GR)
    - [ ]  virtual inheritance of metric classes
  - [ ]  Output
    - [x]  Fields/currents (SR/GR)
    - [x]  Moments (SR)
    - [x]  Moments (GR)
    - [x]  Particles (SR/GR)
    - [x]  Energy distributions (SR/GR)
    - [ ]  Particle tracking (SR/GR)
  - [ ]  Extra physics
    - [x]  Radiation (synchrotron/IC)
    - [ ]  gamma+B pair production
    - [ ]  QED
  - [x]  MPI
    - [x]  restructure meshblocks
    - [x]  rewrite fieldsolvers (addressing + ranges)
    - [ ]  improve performance


* SR (minkowski)
  - [x]  fieldsolver (1D/2D/3D)
  - [x]  pusher (1D/2D/3D)
  - [x]  deposit (1D/2D/3D)
  - [x]  filtering (1D/2D/3D)


* SR (spherical)
  - [x]  fieldsolver (2D)
  - [x]  pusher (2D)
  - [x]  deposit (2D)
  - [x]  filtering (2D)


* GR (spherical)
  - [x]  fieldsolver (2D)
  - [x]  pusher (2D)
  - [x]  deposit (2D)
  - [x]  filtering (2D)

### Known bugs / minor issues to fix

  ...

> To keep the code clean, readable and easy to debug we employ some of the `c++` best practices described in details in [the following online manual](https://www.learncpp.com/). Basically, if there is any ambiguity on how to implement something, it's a good start to first look if there is any "best practice" solution offered in that manual.

### Unit testing

A limited number of unit tests are now available. To compile/run them:
```shell
cmake -B build -D TESTS=ON
cmake --build build -j $(nproc)
ctest --test-dir build
```

Tests are automatically run when on pull requests to the `master` branch.

## Third-party libraries

1. [`Kokkos`](https://github.com/kokkos/kokkos/): for CPU/GPU portability
2. [`adios2`](https://github.com/ornladios/ADIOS2): for output

## Dependencies

While we try to keep the code as compatible as possible, there are certain stringent requirements we impose.

1. `cmake>=3.16`: for configuration (verify: `cmake --version`);
2. `icx>=19.1` or `gcc>=8.3.1` with `c++17` support (verify: `[icx|gcc] -std=c++17 -v`; optionally `nvcc` compilers: `nvcc --version`).
