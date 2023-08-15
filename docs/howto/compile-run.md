---
hide:
  - footer
---

## Pre-requisites

To compile the code you need to have the following dependencies installed:

  - [`CMake`](https://cmake.org/) (version >= 3.16; verify by running `cmake --version`).
  - [`GCC`](https://gcc.gnu.org/) (version >= 8.3.1; verify by running `gcc --version`) or [Intel C++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) (version >= 19.1 or higher; verify by running `icx --version`).
  - to compile for GPUs, you need to have [`CUDA`](https://developer.nvidia.com/cuda-toolkit) (version >= 11.0; verify by running `nvcc --version`) installed.

## Compilation workflow

1. _Clone_ the repository with the following command:
  ```shell
  git clone git@github.com:haykh/entity.git
  ```
  
    !!! note
      
        It is highly recommended to use `ssh` for cloning the repository. If you have not set up your github `ssh` yet, please follow the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Alternatively, you can clone the repository with `https`:
        ```shell
        git clone https://github.com/haykh/entity.git
        ```

1. _Configure_ the code from the root directory using `cmake`, e.g.:
  ```shell
  # from the root of the repository
  cmake -B build -D pgen=<PROBLEM_GENERATOR> -D Kokkos_ENABLE_CUDA=ON <...>
  ```
  All the build options are specified using the `-D` flag followed by the argument and its value (as shown above). Boolean options are specified as `ON` or `OFF`. The following are all the options that can be specified:

    | Option | Description | Values | Default |
    | --- | --- | --- | --- |
    | `engine` | simulation type | `pic`, `grpic` | `pic` |
    | `pgen` | problem generator | see `<engine>/pgen/` directory | `dummy` |
    | `metric` | metric | `minkowski`, `spherical`, `qspherical`, `kerr_schild`, `qkerr_schild`, `kerr_schild_0` | `minkowski` for `pic`, and `kerr_schild` for `grpic` |
    | `precision` | floating point precision | `single`, `double` | `single` |
    | `output` | enable output | `ON`, `OFF` | `OFF` |
    | `gui` | enable the `nttiny` GUI | `ON`, `OFF` | `OFF` |
    | `mpi` | enable MPI _(TBR in v1.0)_ | `ON`, `OFF` | `OFF` |
    | `DEBUG` | enable debug mode | `ON`, `OFF` | `OFF` |
    | `BENCHMARKS` | compile the executables for benchmarking | `ON`, `OFF` | `OFF` |
    | `TESTS` | compile the unit tests | `ON`, `OFF` | `OFF` |
    
    Additionally, there are some CMake and other library-specific options (for [Kokkos](https://kokkos.github.io/kokkos-core-wiki/keywords.html) and [ADIOS2](https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html#cmake-options)) that can be specified along with the above ones. While the code picks most of these options for the end-user, some of them can/should be specified manually. In particular:

    | Option | Description | Values | Default |
    | --- | --- | --- | --- |
    | `Kokkos_ENABLE_CUDA` | enable CUDA | `ON`, `OFF` | `OFF` |
    | `Kokkos_ENABLE_OPENMP` | enable OpenMP | `ON`, `OFF` | `OFF` |
    | `Kokkos_ARCH_***` | use particular CPU/GPU architecture | see [Kokkos documentation](https://kokkos.github.io/kokkos-core-wiki/keywords.html#architecture-keywords) | `Kokkos` attempts to determine automatically |


    !!! note
        
        When simply compiling with `-D Kokkos_ENABLE_CUDA=ON` without additional flags, `CMake` will try to deduce the GPU architecture based on the machine you are compiling on. Oftentimes this might not be the same as the architecture of the machine you are planning to run on (and sometimes the former might lack GPU altogether). To be more explicit, you can specify the GPU architecture manually using the `-D Kokkos_ARCH_***=ON` flags. For example, to explicitly compile for `A100` GPUs, you can use `-D Kokkos_ARCH_AMPERE80=ON`. For `V100` -- use `-D Kokkos_ARCH_VOLTA70=ON`.


1. After `cmake` is done configuring the code, a directory named `build` will be created in the root directory. Proceed there and _compile_ the code using `make`:
  ```shell
  make -j
  ```

1. After the compilation is done, you will find the corresponding executable called `<executable>.xc` in the `build/src/` directory. That's it! You can now finally _run_ the code.

## Running

There are two types of executables produced after the compilation is done: `entity.xc` and `entity-GUI.xc` (if compiled with the `nttiny` GUI). In both cases one can run the code with the following command:

```shell
<path/to/executable>.xc -input <path/to/input_file>.toml
```
`entity.xc` runs headlessly, producing a generic diagnostic output. To enable data dumping (output), one needs to compile with the `-D output=ON` flag. 

`entity-GUI.xc` runs the simulation together with the GUI. The simulation lives as long as the GUI window is open. Additionally, `entity-GUI.xc` also accepts the `-scale <S>` flag, where `<S>` is the scale factor for the GUI (e.g. `-scale 2` will make the GUI twice as big; this setting depends on the personal preference and the monitor DPI/resolution used).

!!! note
    
    When running the `entity-GUI.xc` on a remote machine (e.g., via a `vnc` server), one needs to run with `vglrun ./path/to/entity-GUI.xc`. This is because `entity-GUI.xc` uses OpenGL for rendering the GUI, and `vglrun` is a wrapper that enables OpenGL on a remote machine.
      
!!! note "For the Stellar Princeton cluster users"
    
    For convenience we provide precompiled libraries (`kokkos` and `adios2`) for the Stellar users. To use them, run the following:
    ```shell
    # this line can also be added to your ~/.bashrc or ~/.zshrc for auto loading
    module use --append /home/hakobyan/.modules
    # see the new available modules with ...
    module avail
    # load ...
    module load entity/gpu-volta-70
    # or ...
    module load entity/gui
    # or ...
    module load entity/gpu-ampere-80
    # ... depending on the architecture
    # then configuring the code is quite straightforward
    cmake -B build -D pgen=... -D metric=...
    # Kokkos_ARCH_***, Kokkos_ENABLE_CUDA, etc. are already set
    ```

## Docker 

!!! warning
  
    While the Docker environment for the code is available (see `docker` directory in the root of the repository), it is not documented yet.