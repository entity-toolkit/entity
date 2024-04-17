#include "output/writer.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/param_container.h"

#include <Kokkos_Core.hpp>

#include <any>
#include <string>
#include <type_traits>
#include <vector>

namespace out {

  Writer::Writer(const std::string& engine) : m_engine { engine } {
    m_io = m_adios.DeclareIO("Entity::ADIOS2");
    m_io.SetEngine(engine);

    m_io.DefineVariable<std::size_t>("Step");
    m_io.DefineVariable<real_t>("Time");
  }

  void Writer::writeAttrs(const prm::Parameters&) {
    // todo!()
  }

  void Writer::defineFieldLayout(const std::vector<std::size_t>& glob_shape,
                                 const std::vector<std::size_t>& loc_corner,
                                 const std::vector<std::size_t>& loc_shape,
                                 bool                            incl_ghosts) {
    m_flds_ghosts = incl_ghosts;

    m_flds_g_shape  = glob_shape;
    m_flds_l_corner = loc_corner;
    m_flds_l_shape  = loc_shape;

    if constexpr (std::is_same<typename ndfield_t<Dim::_3D, 6>::array_layout,
                               Kokkos::LayoutRight>::value) {
      m_io.DefineAttribute("LayoutRight", 1);
    } else {
      std::reverse(m_flds_g_shape.begin(), m_flds_g_shape.end());
      std::reverse(m_flds_l_corner.begin(), m_flds_l_corner.end());
      std::reverse(m_flds_l_shape.begin(), m_flds_l_shape.end());
      m_io.DefineAttribute("LayoutRight", 0);
    }
  }

  void Writer::defineFieldOutputs(const SimEngine&                S,
                                  const std::vector<std::string>& flds_out) {
    m_flds_writers.clear();
    raise::ErrorIf((m_flds_g_shape.size() == 0) || (m_flds_l_corner.size() == 0) ||
                     (m_flds_l_shape.size() == 0),
                   "Fields layout must be defined before output fields",
                   HERE);
    for (const auto& fld : flds_out) {
      m_flds_writers.emplace_back(S, fld);
    }
    for (const auto& fld : m_flds_writers) {
      if (fld.comp.size() == 0) {
        m_io.DefineVariable<real_t>(fld.name(),
                                    m_flds_g_shape,
                                    m_flds_l_corner,
                                    m_flds_l_shape,
                                    adios2::ConstantDims);
      } else {
        for (std::size_t i { 0 }; i < fld.comp.size(); ++i) {
          m_io.DefineVariable<real_t>(fld.name(i),
                                      m_flds_g_shape,
                                      m_flds_l_corner,
                                      m_flds_l_shape,
                                      adios2::ConstantDims);
        }
      }
    }
  }

  template <Dimension D, int N>
  void WriteField(adios2::IO&            io,
                  adios2::Engine&        writer,
                  const std::string&     varname,
                  const ndfield_t<D, N>& field,
                  std::size_t            comp,
                  bool                   ghosts) {
    auto       var      = io.InquireVariable<real_t>(varname);
    const auto gh_zones = ghosts ? 0 : N_GHOSTS;

    if constexpr (D == Dim::_1D) {
      auto slice_i1     = range_tuple_t(gh_zones, field.extent(0) - gh_zones);
      auto slice        = Kokkos::subview(field, slice_i1, comp);
      auto output_field = array_t<real_t*>("output_field", slice.extent(0));
      Kokkos::deep_copy(output_field, slice);
      auto output_field_host = Kokkos::create_mirror_view(output_field);
      Kokkos::deep_copy(output_field_host, output_field);
      writer.Put(var, output_field_host);
    } else if constexpr (D == Dim::_2D) {
      auto slice_i1     = range_tuple_t(gh_zones, field.extent(0) - gh_zones);
      auto slice_i2     = range_tuple_t(gh_zones, field.extent(1) - gh_zones);
      auto slice        = Kokkos::subview(field, slice_i1, slice_i2, comp);
      auto output_field = array_t<real_t**>("output_field",
                                            slice.extent(0),
                                            slice.extent(1));
      Kokkos::deep_copy(output_field, slice);
      auto output_field_host = Kokkos::create_mirror_view(output_field);
      Kokkos::deep_copy(output_field_host, output_field);
      writer.Put(var, output_field_host);
    } else if constexpr (D == Dim::_3D) {
      auto slice_i1 = range_tuple_t(gh_zones, field.extent(0) - gh_zones);
      auto slice_i2 = range_tuple_t(gh_zones, field.extent(1) - gh_zones);
      auto slice_i3 = range_tuple_t(gh_zones, field.extent(2) - gh_zones);
      auto slice = Kokkos::subview(field, slice_i1, slice_i2, slice_i3, comp);
      auto output_field = array_t<real_t***>("output_field",
                                             slice.extent(0),
                                             slice.extent(1),
                                             slice.extent(2));
      Kokkos::deep_copy(output_field, slice);
      auto output_field_host = Kokkos::create_mirror_view(output_field);
      Kokkos::deep_copy(output_field_host, output_field);
      writer.Put(var, output_field_host);
    }
  }

  template <Dimension D, int N>
  void Writer::writeField(const std::vector<std::string>& names,
                          const ndfield_t<D, N>&          fld,
                          const std::vector<std::size_t>& addresses) {
    raise::ErrorIf(addresses.size() > N,
                   "addresses vector size must be less than N",
                   HERE);
    raise::ErrorIf(names.size() != addresses.size(),
                   "# of names != # of addresses ",
                   HERE);
    for (std::size_t i { 0 }; i < addresses.size(); ++i) {
      WriteField<D, N>(m_io, m_writer, names[i], fld, addresses[i], m_flds_ghosts);
    }
  }

  void Writer::beginWriting(const std::string& fname, std::size_t tstep, real_t time) {
    m_writer = m_io.Open(fname + (m_engine == "hdf5" ? ".h5" : ".bp"), m_mode);
    m_mode   = adios2::Mode::Append;
    m_writer.BeginStep();
    std::size_t step = tstep;
    m_writer.Put(m_io.InquireVariable<std::size_t>("Step"), &step);
    m_writer.Put(m_io.InquireVariable<real_t>("Time"), &time);
  }

  void Writer::endWriting() {
    m_writer.EndStep();
    m_writer.Close();
  }

  template void Writer::writeField<Dim::_1D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_1D, 3>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_1D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_1D, 6>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_2D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_2D, 3>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_2D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_2D, 6>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_3D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_3D, 3>&,
                                                const std::vector<std::size_t>&);
  template void Writer::writeField<Dim::_3D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_3D, 6>&,
                                                const std::vector<std::size_t>&);

  template void WriteField<Dim::_1D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_1D, 3>&,
                                        std::size_t,
                                        bool);
  template void WriteField<Dim::_1D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_1D, 6>&,
                                        std::size_t,
                                        bool);
  template void WriteField<Dim::_2D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_2D, 3>&,
                                        std::size_t,
                                        bool);
  template void WriteField<Dim::_2D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_2D, 6>&,
                                        std::size_t,
                                        bool);
  template void WriteField<Dim::_3D, 3>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_3D, 3>&,
                                        std::size_t,
                                        bool);
  template void WriteField<Dim::_3D, 6>(adios2::IO&,
                                        adios2::Engine&,
                                        const std::string&,
                                        const ndfield_t<Dim::_3D, 6>&,
                                        std::size_t,
                                        bool);

} // namespace out