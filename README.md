# `Entity` a.k.a. `ntt`
One particle-in-cell code to rule them all.

## Contributors

🍵 __Benjamin Crinquand__ {[@bcrinquand](https://github.com/bcrinquand)}  

☕ __Hayk Hakobyan__ {[@haykh](https://github.com/haykh)}

## Getting started

_Clone_ this repository and all its dependencies with the following command:

```sh
git clone --recursive git@github.com:haykh/entity.git
# to update submodules
git submodule update --remote
```

_Configure_ the code by running `configure.py` file with the desired specifications. This will generate a disposable `Makefile` which is used to build the code. Configuration example might look something like this:

```sh
python configure.py -debug --compiler=g++ --precision=single --pgen=unit/boris
```

> To see all the available configuration flags run `python configure.py -h`.

Once the code is configured, and the `Makefile` is generated in the specified path (by default it is `build/`), you may _compile_ the desired code regime by going into the `build` directory and running `make <REGIME>`. Currently we support the following regimes:

* `ntt`: main regime for performance runs;
* `test`: test regime that runs a series of unit tests.

> Running `make all` will compile all the available regimes. `make` or `make help` will show more detailed instruction list.

After the compilation is successful, you will find the corresponding executable called `<REGIME>.exec` in the `bin/` directory (or whatever was specified during the configure). That's it!

> Directories where the temporary compiled objects and executables go can be defined during the configure time using the flags `--build=<DIR>` and `--bin=<DIR>` correspondingly. By default if not specified the configure script assumes `--build=build/` and `--bin=bin/`. Passing the current directory for `--build` is a bad idea, as there are tons of temporary files generated at compile time, especially from `Kokkos` library.  

## Development status

* PIC
  - [x] spherical/qspherical metrics (2D)
  - [x] minkowski field solver (1D/2D/3D)
  - [x] curvilinear field solver (2D)
  - [x] minkowski particle pusher (Boris; 1D/2D/3D)
  - [x] curvilinear particle pusher (Boris; 2D)
  - [ ] minkowski current deposition (1D/2D/3D)
  - [ ] curvilinear current deposition (2D)
  - [ ] cubed sphere metric (3D)
* GRPIC
  - [x] spherical/qspherical Kerr-Schild metrics (2D)
  - [x] field solver (2D)
  - [ ] particle pusher (1D/2D/3D)
  - [ ] current deposition (2D)
  - [ ] cartesian Kerr-Schild metrics (1D/2D/3D)

### Known bugs / minor issues to fix

- [ ] `$(CURDIR)` seems to fail in some instances (need a more robust apprch)
- [ ] check python `subprocess.run` command during the configure stage
- [ ] check if compilation of `glfw` is possible (or if `glfw` is available)
- [ ] same for `freetype`
- [ ] clarify `nttiny_path` w.r.t. what (maybe add an error messages in configure script)

> To keep the code clean, readable and easy to debug we employ some of the `c++` best practices described in details in [the following online manual](https://www.learncpp.com/). Basically, if there is any ambiguity on how to implement something, it's a good start to first look if there is any "best practice" solution offered in that manual.

### Unit testing

_[under construction]_

### Logging

_[under construction]_

## Third-party libraries

1. [`Kokkos`](https://github.com/kokkos/kokkos/): for CPU/GPU portability;
2. [`plog`](https://github.com/SergiusTheBest/plog): for runtime logging;
3. [`acutest`](https://github.com/mity/acutest): for unit testing;
4. [`toml11`](https://github.com/ToruNiina/toml11): for `toml` file parsing.

> All the third-party libraries reside in the `extern` directory.

The header-only libraries can be included in the `.cpp` or `.h` files with `#include "{LIBNAME}/{LIBHEADER}[.h|.hpp]"`. Compiled libraries such as `Kokkos` are included similar to standard libraries, e.g., `#include <Kokkos_Core.hpp>`.

## Dependencies

While we try to keep the code as compatible as possible, there are certain stringent requirements we impose (primarily due to limitations by `Kokkos`).

1. `python3`: for configuration (verify: `python --version`);
2. `GNU Make<=4.2.1`: for compilation (verify: `make -v`);
3. `icc>=19.1` or `gcc>=8.3.1` with `c++17` support (verify: `[icc|gcc] -std=c++17 -v`).

> For `apple` users: the default `clang` compilers that ship now with macOS systems have trouble with some of the default math libraries. For that reason we highly encourage to use macOS package manager such as `brew` to [install](https://formulae.brew.sh/formula/gcc) the `gnu` compilers. `clang` also does not natively support `OpenMP`, while `gcc` compilers have no problem with that.

## To-do architecture-wise

- [ ] offset views are possible with `kokkos` in [experimental branch](https://github.com/kokkos/kokkos/wiki/Offset-View)

---

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
