
if constexpr (D == Dim1) {
  extent = fmt::format("{} {} 0 0 0 0", params.extent()[0], params.extent()[1]);
} else if constexpr (D == Dim2) {
  extent = fmt::format("{} {} {} {} 0 0",
                       params.extent()[0],
                       params.extent()[1],
                       params.extent()[2],
                       params.extent()[3]);
} else if constexpr (D == Dim3) {
  extent = fmt::format("{} {} {} {} {} {}",
                       params.extent()[0],
                       params.extent()[1],
                       params.extent()[2],
                       params.extent()[3],
                       params.extent()[4],
                       params.extent()[5]);
}

const std::string vtk_xml = R"(
  <?xml version="1.0"?>
  <VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
    <RectilinearGrid WholeExtent=")"
                            + extent + R"(>"
        <Piece Extent=")" + extent
                            + R"(">
          <CellData Scalars="DUMMY">
            <DataArray Name="DUMMY"/>
            <DataArray Name="TIME">
              step
            </DataArray>
          </CellData>
        </Piece>
      </RectilinearGrid>
  </VTKFile>)";

if constexpr (D == Dim1) {
  extent = fmt::format("0 {} 0 0 0 0", params.resolution()[0] + 1);
} else if constexpr (D == Dim2) {
  extent
    = fmt::format("0 {} 0 {} 0 0", params.resolution()[0] + 1, params.resolution()[1] +
    1);
} else if constexpr (D == Dim3) {
  extent = fmt::format("0 {} 0 {} 0 {}",
                       params.resolution()[0] + 1,
                       params.resolution()[1] + 1,
                       params.resolution()[2] + 1);
}
const std::string vtk_xml = R"(
  <?xml version="1.0"?>
  <VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
    <ImageData WholeExtent=")"
                            + extent + R"(" Origin="0 0 0" Spacing="1 1 1">
      <Piece Extent=")" + extent
                            + R"(">
        <CellData Scalars="DUMMY">
          <DataArray Name="DUMMY"/>
          <DataArray Name="TIME">
            time
          </DataArray>
        </CellData>
      </Piece>
    </ImageData>
  </VTKFile>)";

std::cout << vtk_xml << std::endl;

m_io.DefineAttribute<std::string>("vtk.xml", vtk_xml);
