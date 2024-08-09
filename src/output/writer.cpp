#include "output/writer.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/param_container.h"

#include <Kokkos_Core.hpp>

#include <string>
#include <vector>

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif

namespace out {

  Writer::Writer(const std::string& engine) : m_engine { engine } {
    m_io = m_adios.DeclareIO("Entity::ADIOS2");
    m_io.SetEngine(engine);

    m_io.DefineVariable<std::size_t>("Step");
    m_io.DefineVariable<long double>("Time");
  }

  void Writer::addTracker(const std::string& type,
                          std::size_t        interval,
                          long double        interval_time) {
    m_trackers.insert(std::pair<std::string, out::Tracker>(
      { type, Tracker(type, interval, interval_time) }));
  }

  auto Writer::shouldWrite(const std::string& type,
                           std::size_t        step,
                           long double        time) -> bool {
    if (m_trackers.find(type) != m_trackers.end()) {
      return m_trackers.at(type).shouldWrite(step, time);
    } else {
      raise::Error("Tracker type not found", HERE);
      return false;
    }
  }

  void Writer::defineMeshLayout(const std::vector<std::size_t>& glob_shape,
                                const std::vector<std::size_t>& loc_corner,
                                const std::vector<std::size_t>& loc_shape,
                                bool                            incl_ghosts,
                                Coord                           coords) {
    m_flds_ghosts   = incl_ghosts;
    m_flds_g_shape  = glob_shape;
    m_flds_l_corner = loc_corner;
    m_flds_l_shape  = loc_shape;

    m_io.DefineAttribute("NGhosts", incl_ghosts ? N_GHOSTS : 0);
    m_io.DefineAttribute("Dimension", m_flds_g_shape.size());
    m_io.DefineAttribute("Coordinates", std::string(coords.to_string()));

    for (std::size_t i { 0 }; i < m_flds_g_shape.size(); ++i) {
      // cell-centers
      adios2::Dims g_shape  = { m_flds_g_shape[i] };
      adios2::Dims l_corner = { m_flds_l_corner[i] };
      adios2::Dims l_shape  = { m_flds_l_shape[i] };
      m_io.DefineVariable<real_t>("X" + std::to_string(i + 1),
                                  g_shape,
                                  l_corner,
                                  l_shape,
                                  adios2::ConstantDims);
      // cell-edges
      const auto   is_last  = (m_flds_l_corner[i] + m_flds_l_shape[i] ==
                            m_flds_g_shape[i]);
      adios2::Dims g_shape1 = { m_flds_g_shape[i] + 1 };
      adios2::Dims l_shape1 = { m_flds_l_shape[i] + (is_last ? 1 : 0) };
      m_io.DefineVariable<real_t>("X" + std::to_string(i + 1) + "e",
                                  g_shape1,
                                  l_corner,
                                  l_shape1,
                                  adios2::ConstantDims);
    }

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
                   "Mesh layout must be defined before field output",
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

  void Writer::defineParticleOutputs(Dimension                          dim,
                                     const std::vector<unsigned short>& specs) {
    m_prtl_writers.clear();
    for (const auto& s : specs) {
      m_prtl_writers.emplace_back(s);
    }
    for (const auto& prtl : m_prtl_writers) {
      for (auto d { 0u }; d < dim; ++d) {
        m_io.DefineVariable<real_t>(prtl.name("X", d + 1),
                                    {},
                                    {},
                                    { adios2::UnknownDim });
      }
      for (auto d { 0u }; d < Dim::_3D; ++d) {
        m_io.DefineVariable<real_t>(prtl.name("U", d + 1),
                                    {},
                                    {},
                                    { adios2::UnknownDim });
      }
      m_io.DefineVariable<real_t>(prtl.name("W", 0), {}, {}, { adios2::UnknownDim });
    }
  }

  void Writer::defineSpectraOutputs(const std::vector<unsigned short>& specs) {
    m_spectra_writers.clear();
    for (const auto& s : specs) {
      m_spectra_writers.emplace_back(s);
    }
    m_io.DefineVariable<real_t>("sEbn", {}, {}, { adios2::UnknownDim });
    for (const auto& sp : m_spectra_writers) {
      m_io.DefineVariable<real_t>(sp.name(), {}, {}, { adios2::UnknownDim });
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

  void Writer::writeParticleQuantity(const array_t<real_t*>& array,
                                     const std::string&      varname) {
    auto var = m_io.InquireVariable<real_t>(varname);
    var.SetSelection(adios2::Box<adios2::Dims>({}, { array.extent(0) }));
    auto array_h = Kokkos::create_mirror_view(array);
    Kokkos::deep_copy(array_h, array);
    m_writer.Put<real_t>(var, array_h);
  }

  void Writer::writeSpectrum(const array_t<real_t*>& counts,
                             const std::string&      varname) {
    auto counts_h = Kokkos::create_mirror_view(counts);
    Kokkos::deep_copy(counts_h, counts);
#if defined(MPI_ENABLED)
    array_t<real_t*> counts_all { "counts_all", counts.extent(0) };
    auto             counts_h_all = Kokkos::create_mirror_view(counts_all);
    int              rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(counts_h.data(),
               counts_h_all.data(),
               counts_h.extent(0),
               mpi::get_type<real_t>(),
               MPI_SUM,
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank == MPI_ROOT_RANK) {
      auto var = m_io.InquireVariable<real_t>(varname);
      var.SetSelection(adios2::Box<adios2::Dims>({}, { counts.extent(0) }));
      m_writer.Put<real_t>(var, counts_h_all);
    }
#else
    auto var = m_io.InquireVariable<real_t>(varname);
    var.SetSelection(adios2::Box<adios2::Dims>({}, { counts.extent(0) }));
    m_writer.Put<real_t>(var, counts_h);
#endif
  }

  void Writer::writeSpectrumBins(const array_t<real_t*>& e_bins,
                                 const std::string&      varname) {
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != MPI_ROOT_RANK) {
      return;
    }
#endif
    auto var = m_io.InquireVariable<real_t>(varname);
    var.SetSelection(adios2::Box<adios2::Dims>({}, { e_bins.extent(0) }));
    auto e_bins_h = Kokkos::create_mirror_view(e_bins);
    Kokkos::deep_copy(e_bins_h, e_bins);
    m_writer.Put<real_t>(var, e_bins_h);
  }

  void Writer::writeMesh(unsigned short          dim,
                         const array_t<real_t*>& xc,
                         const array_t<real_t*>& xe) {
    auto varc = m_io.InquireVariable<real_t>("X" + std::to_string(dim + 1));
    auto vare = m_io.InquireVariable<real_t>("X" + std::to_string(dim + 1) + "e");
    auto xc_h = Kokkos::create_mirror_view(xc);
    auto xe_h = Kokkos::create_mirror_view(xe);
    Kokkos::deep_copy(xc_h, xc);
    Kokkos::deep_copy(xe_h, xe);
    m_writer.Put(varc, xc_h);
    m_writer.Put(vare, xe_h);
  }

  void Writer::beginWriting(const std::string& fname,
                            std::size_t        tstep,
                            long double        time) {
    m_adios.ExitComputationBlock();
    try {
      m_writer = m_io.Open(fname + (m_engine == "hdf5" ? ".h5" : ".bp"), m_mode);
    } catch (std::exception& e) {
      raise::Fatal(e.what(), HERE);
    }
    m_mode = adios2::Mode::Append;
    m_writer.BeginStep();
    m_writer.Put(m_io.InquireVariable<std::size_t>("Step"), &tstep);
    m_writer.Put(m_io.InquireVariable<long double>("Time"), &time);
  }

  void Writer::endWriting() {
    m_writer.EndStep();
    m_writer.Close();
    m_adios.EnterComputationBlock();
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
