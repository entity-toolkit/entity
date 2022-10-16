---
hide:
  - footer
lastCodeSync:
  date: 12th of Aug 2022
---

1. Clone the repository and all its dependencies with the following command:
  ```shell
  git clone git@github.com:haykh/entity.git
  git submodule update --recursive --init
  # to update the submodules
  git submodule update --recursive --remote
  ```
2. Configure the code by running `configure.py` file with the desired specifications. This will generate a disposable `Makefile` (in the `build/` directory) which is used to build the code. Configuration example might look something like this:
  ```shell
  python3 configure.py -debug --compiler=g++ --precision=single --pgen=unit/boris
  ```

    !!! hint
        
        To see all the available configuration flags run `python configure.py -h`. Below is an output as of {{page.meta.lastCodeSync.date}}.
        ``` { .lang .hide-overflow-x }
        % python configure.py -h

        ...
                                        
        options:
          -h, --help            show this help message and exit
          -verbose              enable verbose compilation mode
          --build BUILD         specify building directory
          --bin BIN             specify directory for executables
          --compiler COMPILER   choose the compiler
          -debug                compile in `debug` mode
          -nttiny               enable nttiny visualizer compilation
          --nttiny_path NTTINY_PATH
                                specify path for `Nttiny`
          --precision {double,single}
                                code precision (default: `single`)
          --metric {minkowski,spherical,qspherical,kerr_schild,qkerr_schild}
                                select metric to be used (default: `minkowski`)
          --simtype {pic,grpic}
                                select simulation type (default: `pic`)
          --pgen {unit/polar,unit/boris,unit/vertical_gr,unit/monopole_gr,unit/benchmark,unit/deposit,unit/wald_gr,unit/em,dummy}
                                problem generator to be used (default: `ntt_dummy`)
          --kokkos_devices KOKKOS_DEVICES
                                `Kokkos` devices
          --kokkos_arch KOKKOS_ARCH
                                `Kokkos` architecture
          --kokkos_options KOKKOS_OPTIONS
                                `Kokkos` options
          --kokkos_cuda_options KOKKOS_CUDA_OPTIONS
                                `Kokkos` CUDA options
        ```

3. Once the code is configured, and the `Makefile` is generated in the specified path (by default it is `build/`), you may compile the desired target by going into the `build` directory and running `make <TARGET>`. Currently we support the following targets:

      * `ntt`: main target for performance runs;
      * `vis`: target for a runtime visualization;
      * `test`: test regime that runs a series of unit tests.

    !!! note
      
        `make` or `make help` will show more detailed instruction list. `make demo` will demonstrate the full compilation and linking command with all the flags and dependencies.

4. After the compilation is successful, you will find the corresponding executable called `<REGIME>.exec` in the `bin/` directory (or whatever was specified during the configure). That's it! You can now run the code.

    !!! note 
        
        Directories where the temporary compiled objects and executables go can be defined during the configure time using the flags `--build=<DIR>` and `--bin=<DIR>` correspondingly. By default if not specified the configure script assumes `--build=build/` and `--bin=bin/`. Passing the current directory for `--build` is a bad idea, as there are tons of temporary files generated at compile time, especially from `Kokkos` library.  

## Docker