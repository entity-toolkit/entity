---
name: Problems at compile-/runtime
about: report any issues or errors thrown when compiling/running the code
title: "[ERROR | QUESTION]"
labels: ''
assignees: ''

---

**describe the issue**
ideally, what are the steps to reproduce the error. at which stage does it occur (compilation, running, first timestep, etc.)?

**code version**
ideally, the git hash which is shown both at compile time and printed during runtime to the `.info` file

**compiler/library versions**
versions of both host (gcc/clang) and gpu (cuda/rocm) compilers, version of hdf5/mpi (if applicable) and adios2/kokkos

**cmake configuration command**
which flags are being used by cmake to configure/compile the code?

**applicable files**
if the issue occurs at runtime, attach the `.toml` input file, as well as `.diag` and `.info` files written during runtime.
