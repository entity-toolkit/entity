# `Entity` a.k.a. `ntt`
One particle-in-cell code to rule them all.

## Getting started

_Clone_ this repository and all its dependencies with the following command:

```shell
git clone --recursive git@github.com:haykh/entity.git
# to update submodules
git submodule update --remote
```

_Configure_ the code by running `configure.py` file with the desired specifications. This will generate a disposable `Makefile` which is used to build the code. Configuration example might look something like this:

```shell
python configure.py -debug --compiler=icpc --precision=single --pgen=ntt_boris_unit
```

> To see all the available configuration flags run `python configure.py -h`.

Once the code is configured, and the `Makefile` is generated in the specified path (by default it is `build/`), you may _compile_ the desired code regime by going into the `build` directory and running `make <REGIME>`. Currently we support the following regimes:

* `ntt`: main regime for performance runs;
* `test`: test regime that runs a series of unit tests;
* `examples`: build multiple executables that contain typical examples to help guide through the code (see `examples/` directory).

> Running `make all` will compile all the available regimes. `make` or `make help` will show more detailed instruction list.

After the compilation is successful, you will find the corresponding executable called `<REGIME>.exec` in the `bin/` directory (or whatever was specified during the configure). That's it!

> Directories where the temporary compiled objects and executables go can be defined during the configure time using the flags `--build=<DIR>` and `--bin=<DIR>` correspondingly. By default if not specified the configure script assumes `--build=build/` and `--bin=bin/`. Passing the current directory for `--build` is a bad idea, as there are tons of temporary files generated at compile time, especially from `Kokkos` library.  

## Note for developers

To keep the code clean, readable and easy to debug we employ some of the `c++` best practices described in details in [the following online manual](https://www.learncpp.com/). Basically, if there is any ambiguity on how to implement something, it's a good start to first look if there is any "best practice" solution offered in that manual.

To make debugging and styling the code easier, we implement the standard `clang` tools (such as `clang-format` and `clang-tidy`) that will ensure the code satisfies the preset standards. For instructions run `make` from the `build/` directory.

### Unit testing

### Logging

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

## To-do list for the near future

- [ ] offset views are possible with `kokkos` in [experimental branch](https://github.com/kokkos/kokkos/wiki/Offset-View)
- [ ] curvilinear pusher

---

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
