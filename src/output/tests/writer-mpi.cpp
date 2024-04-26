#include "enums.h"
#include "global.h"

#include "utils/formatting.h"

#include "output/fields.h"
#include "output/writer.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <mpi.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void cleanup() {
  namespace fs = std::filesystem;
  // fs::path tempfile_path { "test.h5" };
  // fs::remove(tempfile_path);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    using namespace ntt;
    auto writer = out::Writer("hdf5");
    writer.defineMeshLayout({ static_cast<unsigned long>(size) * 10 },
                            { static_cast<unsigned long>(rank) * 10 },
                            { 10 },
                            false,
                            Coord::Cart);
    writer.defineFieldOutputs(SimEngine::SRPIC, { "E" });

    ndfield_t<Dim::_1D, 3> field { "fld", 10 + 2 * N_GHOSTS };
    Kokkos::parallel_for(
      "fill",
      CreateRangePolicy<Dim::_1D>({ N_GHOSTS }, { 10 + N_GHOSTS }),
      Lambda(index_t i1) {
        field(i1, 0) = i1;
        field(i1, 1) = -(real_t)(i1);
        field(i1, 2) = i1 / 2;
      });
    std::vector<std::string> names;
    std::vector<std::size_t> addresses;
    for (auto i = 0; i < 3; ++i) {
      names.push_back(writer.fieldWriters()[0].name(i));
      addresses.push_back(i);
    }
    writer.beginWriting("test", 0, 0.0);
    writer.writeField<Dim::_1D, 3>(names, field, addresses);
    writer.endWriting();

    writer.beginWriting("test", 1, 0.1);
    writer.writeField<Dim::_1D, 3>(names, field, addresses);
    writer.endWriting();

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    cleanup();
    MPI_Finalize();
    Kokkos::finalize();
    return 1;
  }
  cleanup();
  MPI_Finalize();
  Kokkos::finalize();
  return 0;
}
