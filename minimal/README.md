# Minimal third-party tests

These minimal tests are designed to test the third-party libraries outside of the `Entity` scope. These tests will show whether there is an issue with the way third-party are installed (or the cluster is set up).

To compile:

```sh
cmake -B build -D MODES="MPI;MPI_SIMPLE;ADIOS2_NOMPI;ADIOS2_MPI"
cmake --build build -j
```

This will produce executables, one for each test, in the `build` directory.

The `MODES` flag determines the tests it will generate and can be a subset of the following (separated with a `;`):

- `MPI` test of pure MPI + Kokkos (can also add `-D GPU_AWARE_MPI=OFF` to disable the GPU-aware MPI explicitly);
- `MPI_SIMPLE` a simpler test of pure MPI + Kokkos;
- `ADIOS2_NOMPI` test of ADIOS2 library without MPI;
- `ADIOS2_MPI` same but with MPI.

All tests also use `Kokkos`. To build `ADIOS2` or `Kokkos` in-tree, you may pass the regular `-D Kokkos_***` and `-D ADIOS2_***` flags to cmake`.
