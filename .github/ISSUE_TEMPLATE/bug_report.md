---
name: Bug report
about: template for reporting an unexpected behavior or bugs
title: "[BUG]"
labels: bug
assignees: ''

---

**describe the bug**
a clear and concise description of what the bug is

**code version**
ideally, the git hash which is shown both at compile time and printed during runtime to the `.info` file

**compiler/library versions**
versions of both host (gcc/clang) and gpu (cuda/rocm) compilers, version of hdf5/mpi (if applicable) and adios2/kokkos

**cmake configuration command**
which flags are being used by cmake to configure/compile the code?
