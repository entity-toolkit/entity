# Project Overview

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
├── include                      # included header-only third-party libraries
│   ├── plog
│   └── toml11
├── minimal                      # set of minimalist programs for testing MPI/Kokkos/adios2
├── pgens                        # problem generators
├── examples                     # example problem generators with standard use-cases
├── tutorials                    # problem generators from tutorials
├── scripts                      # user-facing helper scripts
│   ├── dependencies.py          #   deployment scripts on various machines
│   ├── generate_template.py     #   renders `input.default.toml` from `entity.schema.json`
│   ├── ideal_tile_size.py       #   recommends the team tile size for the tiled deposit
│   └── render_preview.py        #   previews the in-situ renderer geometry from an input file
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
├── .clang-tidy                  # configurations for the clang-tidy
├── .gitattributes
├── .gitignore
├── .gitmodules
├── .tombi.toml                  # formatting guidelines for toml files + schema association
├── CITATION
├── CODEGUIDE.md                 # this file
├── CMakeLists.txt               # root cmake file
├── CODE_OF_CONDUCT.md
├── LICENSE
├── README.md
├── entity.schema.json           # JSON Schema for the input file: the source of truth
└── input.default.toml           # generated reference input with every option at its default
```

## Testing

The code is tested using the `./dev/scripts/tests.sh` script which compiles and runs all the unit tests using `ctest`:

```sh
./dev/scripts/tests.sh --build build_dir --flags "-D mpi=ON" --with_tests
```

All the unit tests are inside the `tests/` directory each within the respective subdirectory; e.g., tests for `src/kernels` are in `tests/kernels`. When testing, build the tests both with and without MPI and, ideally, with and without GPU (when available).

You can also compile all the problem generators and run the ones from the `examples` directory and automatically create plots to check the validity of the code (requires `nt2py` to be installed via `pip`):

```sh
./dev/scripts/tests.sh --build build_dir --flags "-D mpi=ON" --with_pgens --make_plots
```

## Input configuration

`entity.schema.json` is the single source of truth for the input file. It is a [JSON Schema](https://json-schema.org) (draft 2020-12) describing every table and key the code reads, and it serves two purposes at once:

* editors validate and autocomplete input files against it as you type (see [Formatting](#formatting) below);
* `input.default.toml` -- the annotated reference input listing every option -- is *generated* from it, so the docs cannot drift from what is validated.

Regenerate the reference input after any schema change:

```sh
python scripts/generate_template.py -d -o input.default.toml
```

Dropping `-d` renders the same file with every value left as `""`, i.e. a blank form to fill in rather than a list of defaults. Writing to stdout (the default) is handy for reviewing a change: `diff <(python scripts/generate_template.py -d) input.default.toml`.

### The `x-entity` annotations

Standard JSON Schema keywords (`type`, `enum`, `minimum`, `items`, `prefixItems`, `required`, `default`, `deprecated`, ...) carry everything a validator can check. Everything else lives in an `x-entity` object on the node, and is what the generator turns into the `@`-annotations above each key:

| field | meaning |
| --- | --- |
| `type` | the literal `@type:` string, e.g. `"array<uint> [size 1 :->: 3]"` -- richer than the JSON type |
| `default` | the literal `@default:` text, for defaults the code computes at runtime (`"N_GHOSTS"`, `"1% of the domain size"`) or that need a specific notation (`"1e-4"` rather than `0.0001`) |
| `notes` | ordered `@note:` lines; embedded newlines are kept as hard line breaks |
| `examples` | ordered `@example:` lines |
| `enum` | an *illustrative, non-exhaustive* value list, never validated (e.g. `output.fields.quantities`) |
| `deprecated` | the `@deprecated:` text, paired with the standard `"deprecated": true` |
| `inferred` | see below |

`x-entity.inferred` sits on a **table** and lists quantities the code derives rather than reads -- `grid.dim`, `scales.sigma0`, `checkpoint.start_step`. They are deliberately *not* in `properties`, so `additionalProperties: false` rejects them as input keys, and the generator emits them as an `@inferred:` comment block after that table's own keys.

### Adding a new input parameter

1. Add the key to `entity.schema.json`, in the position you want it to appear in the reference input -- property order is emission order, and scalar keys are emitted before sub-tables regardless.
2. Give it a `description` (the brief line) and an `x-entity.type`; add real constraints (`minimum`, `enum`, `minItems`, ...) wherever they are checkable, and a `default` when it has a literal one.
3. Regenerate `input.default.toml`.
4. Parse it in `src/framework/parameters/`, and register any derived quantity under `x-entity.inferred`.

Three things to keep in mind:

* **String enums are matched case-insensitively by the code** (`fmt::toLower` is applied to `engine`, `metric`, the boundary lists, `pusher`, `log_level`, ...), so a bare `"enum"` would reject perfectly valid input. The convention is `anyOf: [{"enum": [<canonical>]}, {"type": "string", "pattern": "(?i)^(<canonical>|...)$"}]` -- the enum branch drives completion and hover, the pattern branch keeps any casing legal. Note `(?i)` is a Rust/Python regex extension: tombi honours it, JS-based validators do not.
* **Every table is closed.** Set `additionalProperties: false` so typos are caught; tombi's `strict = true` closes objects that omit it anyway. `[setup]` is the one deliberate exception (`additionalProperties: true`), since its keys belong to the problem generator.
* **If a key's documented default is `[]`, the empty array must validate**, which `minItems` would otherwise forbid -- use `anyOf: [{"maxItems": 0}, {<the real shape>}]` (see `output.render.x1_lim`).

## Code guidelines

### Formatting

To maintain coherence throughout the source code, we use `clang-format` to enforce a uniform style. A corresponding `.clang-format` file with all the style-related settings can be found in the root directory of the code. To use this, one needs to have the `clang-format` executable (typically provided with the `llvm` package). After installing the `clang-format` itself (check by running `clang-format --version`), you can use it either manually by running `clang-format .` in the route directory of the code, or attach it to your favorite code editor to run on save. For VSCode, the recommended extension is [`xaver.clang-format`](https://github.com/xaverh/vscode-clang-format), for vim -- [`rhysd/vim-clang-format`](https://vimawesome.com/plugin/vim-clang-format), for nvim -- [`stevearc/conform.nvim`](https://github.com/stevearc/conform.nvim), for [emacs](https://www.vim.org/download.php).

You can run the formatting on all files with `./dev/scripts/format.sh` (this covers C++ and CMake; TOML is handled separately, below).

TOML files are formatted and validated with [`tombi`](https://tombi-toml.github.io/tombi/), which is a formatter, linter and language server in one. The settings live in `.tombi.toml` in the root directory, which also associates `entity.schema.json` with every `.toml` file in the tree -- so input files are checked against the schema as you edit them, with completion and hover documentation for every key. It is provided by the nix shell (`dev/nix`); otherwise install it with `uvx tombi`, `pip install tombi`, `npm i -g tombi` or `brew install tombi`.

From the command line:

```sh
tombi format              # formats the whole project (or pass files/directories)
tombi format --check      # verify only, for CI -- mirrors `format.sh --verify`
tombi lint <file.toml>    # schema validation only
```

In the editor, point it at the `tombi lsp` language server. For VSCode, the extension is [`tombi-toml.tombi`](https://marketplace.visualstudio.com/items?itemName=tombi-toml.tombi); for nvim, `tombi` ships as a built-in `nvim-lspconfig` server, so `vim.lsp.enable('tombi')` is enough. Individual input files can opt into the schema explicitly -- useful outside the repo -- with a directive on the first line:

```toml
#:schema ./entity.schema.json
```

> [!NOTE]
> `tombi` replaces `taplo`, which the project used previously and which is no longer maintained.

Best practices are also enforced using `clang-tidy`; to generate recommendations for all the files, run `./dev/scripts/tidy.sh --build build_dir` where `build_dir` is the directory where the code was built, or for specific files: `./dev/scripts/tidy.sh --build build_dir --files "(file1|file2).cpp"` or only for the changed files: `./dev/scripts/tidy.sh --build build_dir --changed`. The recommendations will be in the `tidy/` directory.

### General guidelines

* Use `const` and `auto` declarations where possible.

* For real-valued literals, use `ONE`, `ZERO`, `HALF` etc. instead of `1.0`, `0.0`, `0.5` to ensure the compiler will not need to cast. If the value is not defined as a macro, use `static_cast<real_t>(123.4)`.

* Use `{}` in declarations to signify a null (placeholder) value for the given variable:
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
