#include "enums.h"
#include "global.h"

#include "utils/formatting.h"

#include "output/fields.h"
#include "output/writer.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void cleanup() {
  namespace fs = std::filesystem;
  fs::path tempfile_path { "test.h5" };
  fs::remove(tempfile_path);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {

    using namespace ntt;
    auto writer = out::Writer("hdf5");
    writer.defineMeshLayout({ 10, 10, 10 }, { 0, 0, 0 }, { 10, 10, 10 }, false, Coord::Cart);
    writer.defineFieldOutputs(SimEngine::SRPIC, { "E", "B", "Rho_1_3", "N_2" });

    ndfield_t<Dim::_3D, 3> field { "fld",
                                   10 + 2 * N_GHOSTS,
                                   10 + 2 * N_GHOSTS,
                                   10 + 2 * N_GHOSTS };
    Kokkos::parallel_for(
      "fill",
      CreateRangePolicy<Dim::_3D>({ N_GHOSTS, N_GHOSTS, N_GHOSTS },
                                  { 10 + N_GHOSTS, 10 + N_GHOSTS, 10 + N_GHOSTS }),
      Lambda(index_t i1, index_t i2, index_t i3) {
        field(i1, i2, i3, 0) = i1 + i2 + i3;
        field(i1, i2, i3, 1) = i1 * i2 / i3;
        field(i1, i2, i3, 2) = i1 / i2 * i3;
      });
    std::vector<std::string> names;
    std::vector<std::size_t> addresses;
    for (auto i = 0; i < 3; ++i) {
      names.push_back(writer.fieldWriters()[0].name(i));
      addresses.push_back(i);
    }
    writer.beginWriting("test", 0, 0.0);
    writer.writeField<Dim::_3D, 3>(names, field, addresses);
    writer.endWriting();

    writer.beginWriting("test", 1, 0.1);
    writer.writeField<Dim::_3D, 3>(names, field, addresses);
    writer.endWriting();

    {
      // read
      adios2::ADIOS adios;
      adios2::IO    io = adios.DeclareIO("read-test");
      io.SetEngine("hdf5");
      adios2::Engine reader = io.Open("test.h5", adios2::Mode::Read);

      std::size_t step { 0 };
      long double time { 0.0 };
      reader.Get(io.InquireVariable<std::size_t>("Step"), step);
      reader.Get(io.InquireVariable<long double>("Time"), time);
      raise::ErrorIf(step != 0, "Step is not 0", HERE);
      raise::ErrorIf(time != 0.0, "Time is not 0.0", HERE);

      for (std::size_t step = 0; reader.BeginStep() == adios2::StepStatus::OK;
           ++step) {
        std::size_t                   step_read;
        adios2::Variable<std::size_t> stepVar = io.InquireVariable<std::size_t>(
          "Step");
        reader.Get(stepVar, step_read);

        long double time_read;
        reader.Get(io.InquireVariable<long double>("Time"), time_read);
        raise::ErrorIf(step_read != step, "Step is not correct", HERE);
        raise::ErrorIf((float)time_read != (float)step / 10.0f,
                       "Time is not correct",
                       HERE);

        for (const auto& name : names) {
          auto data = io.InquireVariable<real_t>(name);
          raise::ErrorIf(data.Shape().size() != 3,
                         fmt::format("%s is not 3D", name.c_str()),
                         HERE);

          auto        dims = data.Shape();
          std::size_t nx1  = dims[0];
          std::size_t nx2  = dims[1];
          std::size_t nx3  = dims[2];
          raise::ErrorIf((nx1 != 10) || (nx2 != 10) || (nx3 != 10),
                         fmt::format("%s is not 10x10x10", name.c_str()),
                         HERE);
        }
        reader.EndStep();
      }
      reader.Close();
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    cleanup();
    Kokkos::finalize();
    return 1;
  }
  cleanup();
  Kokkos::finalize();
  return 0;
}
