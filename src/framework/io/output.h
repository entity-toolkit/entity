#ifndef FRAMEWORK_IO_OUTPUT_H
#define FRAMEWORK_IO_OUTPUT_H

// #include <adios2.h>
// #include <adios2/cxx11/KokkosView.h>
// // #include 

// int WriteFields(const std::string& fname, const std::string& engine = "BP5") {
//   adios2::ADIOS adios;
//   adios2::IO    io = adios.DeclareIO("WriteKokkos");
//   io.SetEngine(engine);

//   // Declare an array for the ADIOS data of size (NumOfProcesses * N)
//   const adios2::Dims shape {static_cast<size_t>(size * N)};
//   const adios2::Dims start {static_cast<size_t>(rank * N)};
//   const adios2::Dims count {N};
//   auto               data = io.DefineVariable<float>("data", {}, {0, 0}, count);

//   adios2::Engine bpWriter = io.Open(fname, adios2::Mode::Write);

//   // // Simulation steps
//   // for (size_t step = 0; step < nSteps; ++step) {
//   //   // Make a 1D selection to describe the local dimensions of the
//   //   // variable we write and its offsets in the global spaces
//     adios2::Box<adios2::Dims> sel({0}, {N});
//     data.SetSelection(sel);

//     // Start IO step every write step
//     bpWriter.BeginStep();
//     bpWriter.Put(data, gpuSimData);
//     bpWriter.EndStep();

//   //   // Update values in the simulation data
//   //   Kokkos::parallel_for(
//   //     "updateBuffer",
//   //     Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
//   //     KOKKOS_LAMBDA(int i) { gpuSimData(i) += 5; });
//   // }

//   bpWriter.Close();
//   return 0;
// }

#endif // FRAMEWORK_IO_OUTPUT_H