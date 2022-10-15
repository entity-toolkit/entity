#include "wrapper.h"
#include "simulation.h"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <vector>
#include <string>

namespace ntt {
  

  // template <Dimension D, SimulationType S>
  // void Simulation<D, S>::WriteFields(const std::string& fname, const std::string& engine) {
  //   auto  params = *(this->params());
  //   auto& mblock = this->meshblock;
  //   this->SynchronizeHostDevice();

  //   tuple_t<std::size_t, D> resolution;
  //   for (short d = 0; d < (short)D; ++d) {
  //     resolution[d] = params.resolution()[d];
  //   }

  //   adios2::ADIOS adios;
  //   adios2::IO    io = adios.DeclareIO("WriteKokkos");
  //   io.SetEngine(engine);
  //   adios2::Engine writer = io.Open(fname, adios2::Mode::Write);

  //   adios2::Dims             shape, start, count;
  //   std::vector<std::size_t> imin((short)D);
  //   std::vector<std::size_t> imax((short)D);

  //   for (short d = 0; d < (short)D; ++d) {
  //     shape.push_back(params.resolution()[d]);
  //     count.push_back(params.resolution()[d]);
  //     start.push_back(0);
  //   }

  //   for (short d = 0; d < (short)D; ++d) {
  //     imin[d] = 0;
  //     imax[d] = params.resolution()[d];
  //   }
  //   adios2::Box<adios2::Dims> sel(imin, imax);
  //   auto                      data = io.DefineVariable<real_t>("dummy", shape, start, count);
  //   data.SetSelection(sel);

  //   writer.BeginStep();
  //   if constexpr (D == Dim1) {
  //     h_ndfield_t<D> dummy("dummy", resolution[0]);
  //     writer.Put(data, dummy);
  //   } else if constexpr (D == Dim2) {
  //     h_ndfield_t<D> dummy("dummy", resolution[0], resolution[1]);
  //     writer.Put(data, dummy);
  //   } else if constexpr (D == Dim3) {
  //     h_ndfield_t<D> dummy("dummy", resolution[0], resolution[1], resolution[2]);
  //     writer.Put(data, dummy);
  //   }
  //   writer.EndStep();

  //   writer.Close();
  // }
} // namespace ntt

// template class ntt::Simulation<ntt::Dim1, ntt::TypePIC>;
// template class ntt::Simulation<ntt::Dim2, ntt::TypePIC>;
// template class ntt::Simulation<ntt::Dim3, ntt::TypePIC>;