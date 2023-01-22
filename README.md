# `Entity` a.k.a. `ntt`
One particle-in-cell code to rule them all. Find our detailed documentation [here](https://haykh.github.io/entity/).

![image](assets/cover.gif)

## Code contributors (alphabetical)

🍵 __Benjamin Crinquand__ {[@bcrinquand](https://github.com/bcrinquand): GRPIC}

🧋 __Alisa Galishnikova__ {[@alisagk](https://github.com/alisagk): GRPIC}

☕ __Hayk Hakobyan__ {[@haykh](https://github.com/haykh): framework, PIC, GRPIC}

🥔 __Jens Mahlmann__ {[@jmahlmann](https://github.com/jmahlmann): framework, MPI}

🤷 __Arno Vanthieghem__ {[@vanthieg](https://github.com/vanthieg): PIC}

## Development status

### Active to-do

  - [x] change metrics/aux foldername
  - [x] add time as a global parameter
  - [x] use `kokkos` methods for `vis/nttiny.cpp`
  - [x] add disabled indicator for options in `report.cmake`

### Short term things to do/fix

  - [x] routine for easy side/corner range selection
  - [x] aliases for fields/particles/currents
  - [ ] check allocation of proper fields
  - [x] add a simple current filtering
  - [x] field mirrors
  - [ ] unit tests + implement with github actions

### Intermediate term things to do/fix

  - [x] test curvilinear particle pusher
  - [x] particle motion near the axes
  - [x] test curvilinear current deposit
  - [x] deposition near the axes
  - [x] filtering near the axes

### State of things

* PIC
  - [x] spherical/qspherical metrics (2D)
  - [x] minkowski field solver (1D/2D/3D)
  - [x] curvilinear field solver (2D)
  - [x] minkowski particle pusher (Boris; 1D/2D/3D)
  - [x] curvilinear particle pusher (Boris; 2D)
  - [x] minkowski current deposition (1D/2D/3D)
  - [x] curvilinear current deposition (2D)
  - [ ] cubed sphere metric (3D)
* GRPIC
  - [x] spherical/qspherical Kerr-Schild metrics (2D)
  - [x] field solver (2D)
  - [ ] particle pusher (1D/2D/3D)
  - [ ] current deposition (2D)
  - [ ] cartesian Kerr-Schild metrics (1D/2D/3D)

### Known bugs / minor issues to fix

  ...

> To keep the code clean, readable and easy to debug we employ some of the `c++` best practices described in details in [the following online manual](https://www.learncpp.com/). Basically, if there is any ambiguity on how to implement something, it's a good start to first look if there is any "best practice" solution offered in that manual.

### Unit testing

_[under construction]_

### Logging

_[under construction]_

## Third-party libraries

All the third-party libraries are added as git submodules. If the code is cloned recursively, the submodules are automatically included updated.

1. [`Kokkos`](https://github.com/kokkos/kokkos/): for CPU/GPU portability;
2. [`plog`](https://github.com/SergiusTheBest/plog): for runtime logging;
3. [`doctest`](https://github.com/doctest/doctest): for unit testing (header-only);
4. [`toml11`](https://github.com/ToruNiina/toml11): for `toml` file parsing (header-only);

> All the third-party libraries reside in the `extern` directory.

## Dependencies

While we try to keep the code as compatible as possible, there are certain stringent requirements we impose.

1. `cmake>=3.16`: for configuration (verify: `cmake --version`);
2. `icc>=19.1` or `gcc>=8.3.1` with `c++17` support (verify: `[icc|gcc] -std=c++17 -v`; optionally `nvcc` compilers: `nvcc --version`).

> For `apple` users: the default `clang` compilers that ship now with macOS systems have trouble with some of the default math libraries, atomic operations etc. For that reason we highly encourage to use macOS package manager such as `brew` to [install](https://formulae.brew.sh/formula/gcc) the `gnu` compilers. `clang` also does not natively support `OpenMP`, while `gcc` compilers have no problem with that.

---

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
