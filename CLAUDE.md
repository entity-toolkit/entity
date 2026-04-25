# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an astrophysical particle-in-cell plasma simulation code which works in arbitrary curvilinear coordinates and supports multiple simulation engines. It is built using the Kokkos performance portability library with C++20. It is parallelized with MPI, and uses the ADIOS2 library for outputting and checkpointing the simulation data.

## Repository Structure

```
entity
├── cmake                        # additional cmake files
│   ├── adios2Config.cmake       #   default configurations for in-tree build of adios2
│   ├── config.cmake             #   compile-time configuration options
│   ├── defaults.cmake           #   default values for configurations
│   ├── dependencies.cmake       #   functions to fetch and build the dependencies
│   ├── kokkosConfig.cmake       #   default configurations for in-tree build of Kokkos
│   ├── report.cmake             #   configuration repoting
│   ├── styling.cmake            #   styling functions
│   └── tests.cmake              #   root cmake for tests
├── dev                          # developer-specific tools
│   ├── nix                      #   nix-shells
│   ├── runners                  #   dockerfiles for github runners on different architectures
│   ├── scripts                  #   developer-specific scripts
│   ├── Dockerfile.common        #   parent docker environment for development
│   ├── Dockerfile.cuda          #   cuda docker environment
│   ├── Dockerfile.rocm          #   rocm docker environment
│   ├── welcome.cuda
│   └── welcome.rocm
├── extern                       # git submodules
│   ├── Kokkos
│   ├── adios2
│   └── entity-pgens
├── include                      # included header-only third-party libraries
│   ├── plog
│   └── toml11
├── minimal                      # set of minimalist programs for testing MPI/Kokkos/adios2
├── pgens                        # problem generators
├── src                          # main code containing all separate submodules
│   ├── archetypes               #   archetypes which can be used by the user in problem generators
│   ├── engines                  #   simulation engines
│   ├── framework                #   main structures, classes and containers
│   ├── global                   #   global definitions and utilities
│   ├── kernels                  #   core kernels (defined as functors)
│   ├── metrics                  #   various metric classes
│   ├── output                   #   functions related to output
│   ├── CMakeLists.txt
│   └── entity.cpp               #   main entry-point
├── tests                        # unit tests for all submodules
├── .clang-format                # code formatting guidelines for clang-format
├── .gitattributes
├── .gitignore
├── .gitmodules
├── .taplo.toml                  # formatting guidelines for toml files
├── CITATION
├── CLAUDE.md
├── CMakeLists.txt               # root cmake file
├── CODE_OF_CONDUCT.md
├── LICENSE
├── README.md
├── conda-entity-nompi.sh
├── dependencies.py              # deployment scripts on various machines
├── docker-compose.yml
└── input.example.toml           # most complete toml file with all possible input options
```

## Testing

The code is tested using the `./dev/scripts/tests.sh` script which compiles and runs all the unit tests using `ctest`:

```sh
./dev/scripts/tests.sh --build build_dir --flags "-D mpi=ON" --with_tests
```

All the unit tests are inside the `tests/` directory each within the respective subdirectory; e.g., tests for `src/kernels` are in `tests/kernels`. When testing, build the tests both with and without MPI and, ideally, with and without GPU (when available).

You can also compile all the problem generators:

```sh
./dev/scripts/tests.sh --build build_dir --flags "-D mpi=ON" --with_pgens
```

## Code guidelines

* Format of the code is enforced using `clang-format` and `cmake-format`. You can run the formatting on all files with `./dev/scripts/format.sh`.

* Best practices are also enforced using `clang-tidy`; to generate recommendations for all the files, run `./dev/scripts/tidy.sh --build build_dir` where `build_dir` is the directory where the code was built, or for specific files: `./dev/scripts/tidy.sh --build build_dir --files "(file1|file2).cpp"` or only for the changed files: `./dev/scripts/tidy.sh --build build_dir --changed`. The recommendations will be in the `tidy/` directory.

* Use `const` and `auto` declarations where possible.

* For real-valued literals, use `ONE`, `ZERO`, `HALF` etc. instead of `1.0`, `0.0`, `0.5` to ensure the compiler will not need to cast. If the value is not defined as a macro, use `static_cast<real_t>(123.4)`.

* Use {} in declarations to signify a null (placeholder) value for the given variable:
  ```cpp
  auto a { -1 }; // <- value of `a` *will* be changed later (-1 is a placeholder)
  auto b = -1; // <- value of `b` is known at the time of declaration (but *may* change later)
  const auto b = -1; // <- value of `b` is not expected to change later
  ```

* Each header file has to have a description at the top, consisting of the following fields:

  * `@file` [required] the name of the file (as it should be included in other files)
  * `@brief` [required] brief description of what the file contains
  * `@implements` list of class/function/macros implementations
    * structs/classes in this section have no prefix (templates are marked with <>)
    * functions are marked with their return type, e.g. -> void
    * type aliases have a prefix type
    * enums or enum-like objects are marked with enum
    * macros have a prefix macro
    * all of the above are also marked with their respective namespaces (if any): namespace::
  * `@cpp`: list of cpp files that implement the header
  * `@namespaces`: list of namespaces defined in the file
  * `@macros`: list of macros that the file depends on
  * `@note` any additional notes (stack as many as necessary)

* `#ifdef`/`#define` macros should be avoided. Use C++20 concept and `if constexpr ()` expressions to specialize functions and classes instead (ideally, specialize them explicitly). `#ifdef`-s are only acceptable in platform/library-specific parts of the code (e.g., `MPI_ENABLED`, `GPU_ENABLED`, `DEBUG`, etc.), or for major shortcuts.

* Header files should start with `#ifndef` ... `#define` ... and end with `#endif`; do not use `#pragma` guards. The name of the macro should be the same as the name of the file in uppercase, with underscores instead of dots and slashes. For example, for `global/utils/formatting.h`, the macro should be `GLOBAL_UTILS_FORMATTING_H`.

* There is no difference between `.h` and `.hpp` files as both indicate C++ header files. As a consistency convention, we use `.h` for common headers which may be included from multiple `.cpp` files (e.g., metrics), while `.hpp` are very specific headers for only a single (or a couple of) .cpp file (e.g. kernels).

* Do assertions on parameters and quantities whenever possible. Outside the kernels, use `raise::Error(message, HERE)` and `raise::ErrorIf(condition, message, HERE)` to throw exceptions. Inside the kernels, use `raise::KernelError(HERE, message, **args)`. To enable compile-time errors, use `static_assert(condition, message)`. The `HERE` keyword is macro that includes the filename and line number in the error message.