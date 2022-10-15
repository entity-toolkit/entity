#include "wrapper.h"
#include "writer.h"
#include "fields.h"
#include "meshblock.h"
#include "sim_params.h"
#include "utils.h"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <plog/Log.h>
#include <fmt/core.h>

#include <string>
#include <map>
#include <vector>
#include <algorithm>

namespace ntt {

  template <Dimension D, SimulationType S>
  Writer<D, S>::Writer(const SimulationParams& params, const Meshblock<D, S>& mblock) {
    m_io = m_adios.DeclareIO("WriteKokkos");
    m_io.SetEngine("HDF5");
    writer = m_io.Open("flds.h5", adios2::Mode::Write);

    adios2::Dims shape, start, count;
    for (short d = 0; d < (short)D; ++d) {
      shape.push_back(params.resolution()[d]);
      count.push_back(params.resolution()[d]);
      start.push_back(0);
    }
    std::reverse(shape.begin(), shape.end());
    std::reverse(count.begin(), count.end());

    m_vars_i.emplace("step", m_io.DefineVariable<int>("step"));
    m_vars_r.emplace("time", m_io.DefineVariable<real_t>("time"));

    if constexpr (D == Dim1 || D == Dim2 || D == Dim3) {
      m_io.DefineAttribute<real_t>("x1_min", mblock.metric.x1_min);
      m_io.DefineAttribute<real_t>("x1_max", mblock.metric.x1_max);
    }
    if constexpr (D == Dim2 || D == Dim3) {
      m_io.DefineAttribute<real_t>("x2_min", mblock.metric.x2_min);
      m_io.DefineAttribute<real_t>("x2_max", mblock.metric.x2_max);
    }
    if constexpr (D == Dim3) {
      m_io.DefineAttribute<real_t>("x3_min", mblock.metric.x3_min);
      m_io.DefineAttribute<real_t>("x3_max", mblock.metric.x3_max);
    }

    m_io.DefineAttribute<real_t>("dt", mblock.timestep());

    for (auto& var : {"ex1", "ex2", "ex3", "bx1", "bx2", "bx3"}) {
      m_vars_r.emplace(var, m_io.DefineVariable<real_t>(var, shape, start, count));
    }
  }

  template <Dimension D, SimulationType S>
  Writer<D, S>::~Writer() {
    writer.Close();
  }

  template <Dimension D, SimulationType S>
  void Writer<D, S>::WriteFields(const Meshblock<D, S>& mblock,
                                 const real_t&          time,
                                 const std::size_t&     tstep) {
    writer.BeginStep();

    int step = (int)tstep;
    writer.Put<int>(m_vars_i["step"], &step);
    writer.Put<real_t>(m_vars_r["time"], &time);

    std::vector<std::string> field_names = {"ex1", "ex2", "ex3", "bx1", "bx2", "bx3"};
    std::vector<em>          fields = {em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3};
    for (auto&& [var_st, var] : c9::zip(field_names, fields)) {
      if constexpr (D == Dim1) {
        writer.Put<real_t>(m_vars_r[var_st],
                           Kokkos::subview(mblock.em_h,
                                           std::make_pair(N_GHOSTS, mblock.Ni(0) - N_GHOSTS),
                                           (int)var));
      } else if constexpr (D == Dim2) {
        writer.Put<real_t>(m_vars_r[var_st],
                           Kokkos::subview(mblock.em_h,
                                           std::make_pair(N_GHOSTS, mblock.Ni(0) - N_GHOSTS),
                                           std::make_pair(N_GHOSTS, mblock.Ni(1) - N_GHOSTS),
                                           (int)var));
      } else if constexpr (D == Dim3) {
        writer.Put<real_t>(m_vars_r[var_st],
                           Kokkos::subview(mblock.em_h,
                                           std::make_pair(N_GHOSTS, mblock.Ni(0) - N_GHOSTS),
                                           std::make_pair(N_GHOSTS, mblock.Ni(1) - N_GHOSTS),
                                           std::make_pair(N_GHOSTS, mblock.Ni(2) - N_GHOSTS),
                                           (int)var));
      }
    }
    writer.EndStep();
    PLOGD << "Wrote fields to file.";
  }

} // namespace ntt

template class ntt::Writer<ntt::Dim1, ntt::TypePIC>;
template class ntt::Writer<ntt::Dim2, ntt::TypePIC>;
template class ntt::Writer<ntt::Dim3, ntt::TypePIC>;

// if constexpr (D == Dim1) {
//   extent = fmt::format("{} {} 0 0 0 0", params.extent()[0], params.extent()[1]);
// } else if constexpr (D == Dim2) {
//   extent = fmt::format("{} {} {} {} 0 0",
//                        params.extent()[0],
//                        params.extent()[1],
//                        params.extent()[2],
//                        params.extent()[3]);
// } else if constexpr (D == Dim3) {
//   extent = fmt::format("{} {} {} {} {} {}",
//                        params.extent()[0],
//                        params.extent()[1],
//                        params.extent()[2],
//                        params.extent()[3],
//                        params.extent()[4],
//                        params.extent()[5]);
// }

// const std::string vtk_xml = R"(
//   <?xml version="1.0"?>
//   <VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
//     <RectilinearGrid WholeExtent=")"
//                             + extent + R"(>"
//         <Piece Extent=")" + extent
//                             + R"(">
//           <CellData Scalars="DUMMY">
//             <DataArray Name="DUMMY"/>
//             <DataArray Name="TIME">
//               step
//             </DataArray>
//           </CellData>
//         </Piece>
//       </RectilinearGrid>
//   </VTKFile>)";

// if constexpr (D == Dim1) {
//   extent = fmt::format("0 {} 0 0 0 0", params.resolution()[0] + 1);
// } else if constexpr (D == Dim2) {
//   extent
//     = fmt::format("0 {} 0 {} 0 0", params.resolution()[0] + 1, params.resolution()[1] +
//     1);
// } else if constexpr (D == Dim3) {
//   extent = fmt::format("0 {} 0 {} 0 {}",
//                        params.resolution()[0] + 1,
//                        params.resolution()[1] + 1,
//                        params.resolution()[2] + 1);
// }
// const std::string vtk_xml = R"(
//   <?xml version="1.0"?>
//   <VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
//     <ImageData WholeExtent=")"
//                             + extent + R"(" Origin="0 0 0" Spacing="1 1 1">
//       <Piece Extent=")" + extent
//                             + R"(">
//         <CellData Scalars="DUMMY">
//           <DataArray Name="DUMMY"/>
//           <DataArray Name="TIME">
//             time
//           </DataArray>
//         </CellData>
//       </Piece>
//     </ImageData>
//   </VTKFile>)";

// std::cout << vtk_xml << std::endl;

// m_io.DefineAttribute<std::string>("vtk.xml", vtk_xml);