#include "global.h"

#include <Kokkos_Core.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

void ntt::GlobalInitialize(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
#if defined(MPI_ENABLED)
  int required = MPI_THREAD_MULTIPLE;
  int provided;
  MPI_Init_thread(&argc,
                  &argv,
                  required,
                  &provided);
  if (provided != required) {
    std::cerr << "MPI_Init_thread() did not provide the requested threading support." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  //MPI_Init(&argc, &argv);
#endif // MPI_ENABLED
}

void ntt::GlobalFinalize() {
#if defined(MPI_ENABLED)
  MPI_Finalize();
#endif // MPI_ENABLED
  Kokkos::finalize();
}
