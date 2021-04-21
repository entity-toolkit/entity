# `Entity` a.k.a. `ntt`
One particle-in-cell code to rule them all. 

## Getting started

_Clone_ this repository with the following command:

```shell
git clone --recursive https://github.com/haykh/entity
```

Notice the `--recursive` flag which ensures that all the dependent libraries (submodules) are also downloaded. In case you already have the cloned repository but missed the submodules (this could happen if you `git clone` without `--recursive`) you may get them by doing

```shell
$ git submodule init 
$ git submodule update
```

_Configure_ the code by running `configure.py` file with the desired specifications. This will generat a disposable `Makefile` which is used to build the code. Configuration example might look something like this:

```shell
$ python configure.py -debug --compiler=icc --precision=single
```

> To see all the available configuration flags run `python configure.py -h`.

Once the code is configured, and the `Makefile` is generated, you may _compile_ the desired code regime by running `make <REGIME>`. Currently we support the following regimes:

* `ntt`: main regime for performance runs;
* `test`: test regime that runs a series of unit tests.

> Running `make all` or `make` will compile all the available regimes.

After the compilation is successfull, you will find the corresponding executable called `<REGIME>.exec` in the `build/` directory. That's it!

## Note for developers

To keep the code clean, readable and easy to debug we employ some of the `c++` best practices described in details in [the following online manual](https://www.learncpp.com/). Basically, if there is any ambiguity on how to implement something, it's a good start to first look if there is any "best practice" solution offered in that manual.

### Unit testing

### Logging

## Third-party libraries

All the third-party libraries reside in the `extern` directory. They can be included in the `.cpp` or `.h` files with `#include "<LIBNAME>/<LIB>[.h|.hpp]"`.

1. [`Kokkos`](https://github.com/kokkos/kokkos/): for CPU/GPU portability;
2. [`acutest`](https://github.com/mity/acutest)<sup>h</sup>: for unit testing;
3. [`toml11`](https://github.com/ToruNiina/toml11)<sup>h</sup>: for `toml` file parsing;
4. [`plog`](https://github.com/SergiusTheBest/plog)<sup>h</sup>: for runtime logging.

<sup>h</sup> header-only library

## Dependencies

While we try to keep the code as compatible as possible, there are certain stringent requirements we impose (primarily due to limitations by `Kokkos`). 

1. `python3`: for configuration (verify: `python --version`);
2. `GNU Make<=4.2.1`: for compilation (verify: `make -v`);
3. `icc>=19.1` or `gcc>=8.3.1` with `c++17` support (verify: `[icc|gcc] -std=c++17 -v`). 

## To-do list for the near future

- [ ] parameter reading using `toml`
- [ ] `Kokkos` unit tests
- [ ] `timer` library
