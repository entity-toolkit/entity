#include "global.h"

#include <Kokkos_Core.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#if defined(HIP_ENABLED)
  #include <hip/hip_runtime.h>

  #include <cstdint>
#endif // HIP_ENABLED

namespace {
#if defined(HIP_ENABLED)
  // Turn the ROCm stream-ordered allocator into a caching arena.
  //
  // This Kokkos build uses hipMallocAsync/hipFreeAsync (Kokkos option
  // IMPL_HIP_MALLOC_ASYNC). The default memory pool has a release
  // threshold of 0, so every freed block is handed back to the driver
  // at the next stream sync. With ~50 GB of particle SoA permanently
  // pinned and only ~14 GB free, the per-step churn of dozens of
  // large, differently-sized sort/comm scratch buffers fragments that
  // free space: allocation cost grows monotonically (ParticleSort
  // slowdown) until no contiguous mid-size block remains and BinSort's
  // `sorted_values` allocation fails (OOM). Raising the release
  // threshold to "unlimited" makes the pool retain and recycle freed
  // blocks instead, which stabilizes the working set and removes both
  // the slowdown and the OOM.
  void ConfigureHipMemPool() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
      return;
    }
    hipMemPool_t pool = nullptr;
    if (hipDeviceGetDefaultMemPool(&pool, device) != hipSuccess or
        pool == nullptr) {
      return;
    }
    uint64_t threshold = UINT64_MAX;
    (void)hipMemPoolSetAttribute(pool,
                                 hipMemPoolAttrReleaseThreshold,
                                 &threshold);
  }
#endif // HIP_ENABLED
} // namespace

void ntt::GlobalInitialize(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
#if defined(HIP_ENABLED)
  ConfigureHipMemPool();
#endif // HIP_ENABLED
#if defined(MPI_ENABLED)
  MPI_Init(&argc, &argv);
#endif // MPI_ENABLED
}

void ntt::GlobalFinalize() {
#if defined(MPI_ENABLED)
  MPI_Finalize();
#endif // MPI_ENABLED
  Kokkos::finalize();
}
