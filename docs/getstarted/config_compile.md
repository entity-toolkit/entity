---
hide:
  - footer
---

1. _Clone_ the repository with the following command:
  ```shell
  git clone git@github.com:haykh/entity.git
  ```
  
    !!! note
      
        It is highly recommended to use `ssh` for cloning the repository. If you have not set up your github `ssh` yet, please follow the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Alternatively, you can clone the repository with `https`:
        ```shell
        git clone https://github.com/haykh/entity.git
        ```

2. _Configure_ the code from the root directory using `cmake`, e.g.:
  ```shell
  # from the root of the repository
  cmake -B build -D pgen=<PROBLEM_GENERATOR> -D Kokkos_ENABLE_CUDA=ON <...>
  ```
  All the build options are specified using the `-D` flag followed by the argument and its value (as shown above). Boolean options are specified as `ON` or `OFF`. The following are all the options that can be specified:

    | Option | Description | Values | Default |
    | --- | --- | --- | --- |
    | `simtype` | simulation type | `pic`, `grpic` {{config.changes.v09}} | `pic` |
    | `pgen` | problem generator | see `<simtype>/pgen/` directory | `dummy` |
    | `metric` | metric | `minkowski`, `spherical`, `qspherical`, `kerr_schild`, `qkerr_schild` | `minkowski` for `pic`, and `kerr_schild` for `grpic` |
    | `precision` | floating point precision | `single`, `double` | `single` |
    | `output` | enable output | `ON`, `OFF` | `OFF` |
    | `nttiny` | enable `nttiny` GUI | `ON`, `OFF` | `OFF` |
    | `DEBUG` | enable debug mode | `ON`, `OFF` | `OFF` |
    
    Additionally, there are some CMake and other library-specific options (for [Kokkos](https://kokkos.github.io/kokkos-core-wiki/keywords.html) and [ADIOS2](https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html#cmake-options)) that can be specified along with the above ones. While the code picks most of these options for the end-user, some of them can/should be specified manually. In particular:

    | Option | Description | Values | Default |
    | --- | --- | --- | --- |
    | `Kokkos_ENABLE_CUDA` | enable CUDA | `ON`, `OFF` | `OFF` |
    | `Kokkos_ENABLE_OPENMP` | enable OpenMP | `ON`, `OFF` | `OFF` |
    | `CMAKE_CXX_COMPILER` | C++ compiler | whichever compiler is available on your system | CMake/Kokkos will attempt to find automatically |


3. After `cmake` is done configuring the code, a directory named `build` will be created in the root directory. Proceed there and _compile_ the code using `make`:
  ```shell
  make -j
  ```

4. After the compilation is done, you will find the corresponding executable called `<executable>.xc` in the `build/src/` directory. That's it! You can now run the code.

## Docker 

!!! warning
  
    While the Docker environment for the code is available (see `docker` directory in the root of the repository), it is not documented yet.