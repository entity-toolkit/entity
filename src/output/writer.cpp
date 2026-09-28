#include "output/writer.h"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/param_container.h"
#include "utils/tools.h"

#include "output/utils/tuning.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx/KokkosView.h>

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif

#include <algorithm>
#include <any>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace out {

  void Writer::init(adios2::ADIOS*        ptr_adios,
                    const std::string&    engine,
                    const std::string&    title,
                    const out::Bp5Tuning& bp5) {
    m_engine = fmt::toLower(engine);
    p_adios  = ptr_adios;

    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);

    m_io = p_adios->DeclareIO("Entity::Output");
    m_io.SetEngine(engine);

    out::ApplyBp5Tuning(m_io, m_engine, bp5);

    m_io.DefineVariable<timestep_t>("Step");
    m_io.DefineVariable<simtime_t>("Time");
    m_root = path_t(title);
  }

  void Writer::addTracker(const std::string& type,
                          timestep_t         interval,
                          simtime_t          interval_time) {
    m_trackers.insert({ type, tools::Tracker(type, interval, interval_time) });
  }

  auto Writer::shouldWrite(const std::string& type, timestep_t step, simtime_t time)
    -> bool {
    if (m_trackers.find(type) != m_trackers.end()) {
      return m_trackers.at(type).shouldWrite(step, time);
    } else {
      raise::Error(fmt::format("Tracker type %s not found", type.c_str()), HERE);
      return false;
    }
  }

  void Writer::setMode(adios2::Mode mode) {
    m_mode = mode;
  }

  void Writer::setLocalLayout(const std::vector<ncells_t>& loc_corner,
                              const std::vector<ncells_t>& loc_shape) {
    raise::ErrorIf(loc_corner.size() != m_flds_l_corner.size() or
                     loc_shape.size() != m_flds_l_shape.size(),
                   "setLocalLayout dim mismatch with the original layout",
                   HERE);
    m_flds_l_corner = loc_corner;
    m_flds_l_shape  = loc_shape;
    m_flds_l_corner_dwn.clear();
    m_flds_l_shape_dwn.clear();
    m_flds_l_first.clear();
    for (auto i { 0u }; i < m_flds_g_shape.size(); ++i) {
      const double d = static_cast<double>(m_dwn[i]);
      const double l = static_cast<double>(loc_corner[i]);
      const double n = static_cast<double>(loc_shape[i]);
      const double f = math::ceil(l / d) * d - l;
      m_flds_l_corner_dwn.push_back(static_cast<ncells_t>(math::ceil(l / d)));
      m_flds_l_first.push_back(static_cast<ncells_t>(f));
      m_flds_l_shape_dwn.push_back(static_cast<ncells_t>(math::ceil((n - f) / d)));
    }
    if constexpr (not std::is_same<typename ndfield_t<Dim::_3D, 6>::array_layout,
                                   Kokkos::LayoutRight>::value) {
      std::reverse(m_flds_l_corner_dwn.begin(), m_flds_l_corner_dwn.end());
      std::reverse(m_flds_l_shape_dwn.begin(), m_flds_l_shape_dwn.end());
    }
  }

  void Writer::defineMeshLayout(
    const std::vector<ncells_t>&                 glob_shape,
    const std::vector<ncells_t>&                 loc_corner,
    const std::vector<ncells_t>&                 loc_shape,
    const std::pair<unsigned int, unsigned int>& domain_idx,
    const std::vector<unsigned int>&             dwn,
    bool                                         incl_ghosts,
    Coord                                        coords) {
    m_flds_ghosts = incl_ghosts;
    m_dwn         = dwn;

    m_flds_g_shape  = glob_shape;
    m_flds_l_corner = loc_corner;
    m_flds_l_shape  = loc_shape;

    for (auto i { 0u }; i < glob_shape.size(); ++i) {
      raise::ErrorIf(dwn[i] != 1 && incl_ghosts,
                     "Downsampling with ghosts not supported",
                     HERE);

      const double g = static_cast<double>(glob_shape[i]);
      const double d = static_cast<double>(m_dwn[i]);
      const double l = static_cast<double>(loc_corner[i]);
      const double n = static_cast<double>(loc_shape[i]);
      const double f = math::ceil(l / d) * d - l;
      m_flds_g_shape_dwn.push_back(static_cast<ncells_t>(math::ceil(g / d)));
      m_flds_l_corner_dwn.push_back(static_cast<ncells_t>(math::ceil(l / d)));
      m_flds_l_first.push_back(static_cast<ncells_t>(f));
      m_flds_l_shape_dwn.push_back(static_cast<ncells_t>(math::ceil((n - f) / d)));
    }

    m_io.DefineAttribute("NGhosts",
                         incl_ghosts ? N_GHOSTS : static_cast<ncells_t>(0));
    m_io.DefineAttribute("Dimension", m_flds_g_shape.size());
    m_io.DefineAttribute("Coordinates", std::string(coords.to_string()));

    for (auto i { 0u }; i < m_flds_g_shape.size(); ++i) {
      // cell-centers
      // ConstantDims is intentionally NOT set: per-rank slab shape can change
      // when dynamic load balancing shifts domain boundaries between writes.
      m_io.DefineVariable<real_t>("X" + std::to_string(i + 1),
                                  { m_flds_g_shape_dwn[i] },
                                  { m_flds_l_corner_dwn[i] },
                                  { m_flds_l_shape_dwn[i] });
      // cell-edges
      const auto is_last = (m_flds_l_corner[i] + m_flds_l_shape[i] ==
                            m_flds_g_shape[i]);
      m_io.DefineVariable<real_t>("X" + std::to_string(i + 1) + "e",
                                  { m_flds_g_shape_dwn[i] + 1 },
                                  { m_flds_l_corner_dwn[i] },
                                  { m_flds_l_shape_dwn[i] + (is_last ? 1 : 0) });
      m_io.DefineVariable<std::size_t>(
        "N" + std::to_string(i + 1) + "l",
        { static_cast<unsigned long>(2 * domain_idx.second) },
        { static_cast<unsigned long>(2 * domain_idx.first) },
        { static_cast<unsigned long>(2) });
    }

    if constexpr (std::is_same<typename ndfield_t<Dim::_3D, 6>::array_layout,
                               Kokkos::LayoutRight>::value) {
      m_io.DefineAttribute("LayoutRight", 1);
    } else {
      std::reverse(m_flds_g_shape_dwn.begin(), m_flds_g_shape_dwn.end());
      std::reverse(m_flds_l_corner_dwn.begin(), m_flds_l_corner_dwn.end());
      std::reverse(m_flds_l_shape_dwn.begin(), m_flds_l_shape_dwn.end());
      m_io.DefineAttribute("LayoutRight", 0);
    }
  }

  void Writer::defineFieldOutputs(const SimEngine&                S,
                                  const std::vector<std::string>& flds_out) {
    m_flds_writers.clear();
    raise::ErrorIf(m_flds_g_shape_dwn.empty() or m_flds_l_corner_dwn.empty() or
                     m_flds_l_shape_dwn.empty(),
                   "Mesh layout must be defined before field output",
                   HERE);
    for (const auto& fld : flds_out) {
      m_flds_writers.emplace_back(S, fld);
    }
    for (const auto& fld : m_flds_writers) {
      // ConstantDims is intentionally NOT set: per-rank slab shape can change
      // when dynamic load balancing shifts domain boundaries between writes.
      if (fld.comp.empty()) {
        // scalar
        m_io.DefineVariable<real_t>(fld.name(),
                                    m_flds_g_shape_dwn,
                                    m_flds_l_corner_dwn,
                                    m_flds_l_shape_dwn);
      } else {
        // vector or tensor
        for (auto i { 0u }; i < fld.comp.size(); ++i) {
          m_io.DefineVariable<real_t>(fld.name(i),
                                      m_flds_g_shape_dwn,
                                      m_flds_l_corner_dwn,
                                      m_flds_l_shape_dwn);
        }
      }
    }
  }

  void Writer::defineSpectraOutputs(const std::vector<spidx_t>& specs,
                                    const std::vector<size_t>& num_spatial_bins) {
    m_spectra_writers.clear();
    for (const auto& s : specs) {
      m_spectra_writers.emplace_back(s);
    }
    m_io.DefineVariable<real_t>("sEbn", {}, {}, { adios2::UnknownDim });
    const auto spatial_binning_enabled = std::any_of(num_spatial_bins.begin(),
                                                     num_spatial_bins.end(),
                                                     [](const auto& n) {
                                                       return n != 1u;
                                                     });
    const auto nspec_dims = spatial_binning_enabled ? num_spatial_bins.size() + 1u
                                                    : 1u;
    for (const auto& sp : m_spectra_writers) {
      m_io.DefineVariable<real_t>(sp.name(),
                                  {},
                                  {},
                                  adios2::Dims(nspec_dims, adios2::UnknownDim));
    }
    if (spatial_binning_enabled) {
      const auto dim = num_spatial_bins.size();
      for (auto d { 0u }; d < dim; ++d) {
        m_io.DefineVariable<real_t>("sX" + std::to_string(d + 1) + "bn",
                                    {},
                                    {},
                                    { adios2::UnknownDim });
      }
    }
  }

  void Writer::writeAttrs(const prm::Parameters& params) {
    params.write(m_io);
  }

  template <Dimension D, int N>
  void WriteField(adios2::IO&               io,
                  adios2::Engine&           writer,
                  std::vector<std::any>&    keepalive,
                  const std::string&        varname,
                  const ndfield_t<D, N>&    field,
                  std::size_t               comp,
                  std::vector<unsigned int> dwn,
                  std::vector<ncells_t>     first_cell,
                  bool                      ghosts,
                  const adios2::Dims&       loc_corner_dwn,
                  const adios2::Dims&       loc_shape_dwn) {
    // when dwn != 1 in any direction, it is assumed that ghosts == false
    auto var = io.InquireVariable<real_t>(varname);
    // Refresh the per-step (start, count) so the slab tracks any rebalance
    // that happened since the variable was declared.
    var.SetSelection(adios2::Box<adios2::Dims>(loc_corner_dwn, loc_shape_dwn));
    const auto   gh_zones = ghosts ? 0 : N_GHOSTS;
    ndarray_t<D> output_field {};

    if constexpr (D == Dim::_1D) {
      if (ghosts || dwn[0] == 1) {
        auto slice_i1 = cell_range_t(gh_zones, field.extent(0) - gh_zones);
        auto slice    = Kokkos::subview(field, slice_i1, comp);
        output_field  = array_t<real_t*> { "output_field", slice.extent(0) };
        Kokkos::deep_copy(output_field, slice);
      } else {

        const auto   dwn1          = dwn[0];
        const double first_cell1_d = static_cast<double>(first_cell[0]);
        const double nx1_full      = field.extent(0) - 2 * N_GHOSTS;
        const auto   first_cell1   = first_cell[0];

        const auto nx1_dwn = static_cast<ncells_t>(
          math::ceil((nx1_full - first_cell1_d) / dwn1));

        output_field = array_t<real_t*> { "output_field", nx1_dwn };
        Kokkos::parallel_for(
          "outputField",
          nx1_dwn,
          Lambda(cellidx_t i1) {
            output_field(i1) = field(first_cell1 + i1 * dwn1 + N_GHOSTS, comp);
          });
      }
    } else if constexpr (D == Dim::_2D) {
      if (ghosts || (dwn[0] == 1 && dwn[1] == 1)) {
        auto slice_i1 = cell_range_t(gh_zones, field.extent(0) - gh_zones);
        auto slice_i2 = cell_range_t(gh_zones, field.extent(1) - gh_zones);
        auto slice    = Kokkos::subview(field, slice_i1, slice_i2, comp);
        output_field  = array_t<real_t**> { "output_field",
                                            slice.extent(0),
                                            slice.extent(1) };
        Kokkos::deep_copy(output_field, slice);
      } else {
        const auto   dwn1          = dwn[0];
        const auto   dwn2          = dwn[1];
        const double first_cell1_d = static_cast<double>(first_cell[0]);
        const double first_cell2_d = static_cast<double>(first_cell[1]);
        const double nx1_full      = field.extent(0) - 2 * N_GHOSTS;
        const double nx2_full      = field.extent(1) - 2 * N_GHOSTS;
        const auto   first_cell1   = first_cell[0];
        const auto   first_cell2   = first_cell[1];

        const auto nx1_dwn = static_cast<ncells_t>(
          math::ceil((nx1_full - first_cell1_d) / dwn1));
        const auto nx2_dwn = static_cast<ncells_t>(
          math::ceil((nx2_full - first_cell2_d) / dwn2));
        output_field = array_t<real_t**> { "output_field", nx1_dwn, nx2_dwn };
        Kokkos::parallel_for(
          "outputField",
          CreateRangePolicy<Dim::_2D>({ 0, 0 }, { nx1_dwn, nx2_dwn }),
          Lambda(cellidx_t i1, cellidx_t i2) {
            output_field(i1, i2) = field(first_cell1 + i1 * dwn1 + N_GHOSTS,
                                         first_cell2 + i2 * dwn2 + N_GHOSTS,
                                         comp);
          });
      }
    } else if constexpr (D == Dim::_3D) {
      if (ghosts || (dwn[0] == 1 && dwn[1] == 1 && dwn[2] == 1)) {
        auto slice_i1 = cell_range_t(gh_zones, field.extent(0) - gh_zones);
        auto slice_i2 = cell_range_t(gh_zones, field.extent(1) - gh_zones);
        auto slice_i3 = cell_range_t(gh_zones, field.extent(2) - gh_zones);
        auto slice = Kokkos::subview(field, slice_i1, slice_i2, slice_i3, comp);
        output_field = array_t<real_t***> { "output_field",
                                            slice.extent(0),
                                            slice.extent(1),
                                            slice.extent(2) };
        Kokkos::deep_copy(output_field, slice);
      } else {
        const auto   dwn1          = dwn[0];
        const auto   dwn2          = dwn[1];
        const auto   dwn3          = dwn[2];
        const double first_cell1_d = static_cast<double>(first_cell[0]);
        const double first_cell2_d = static_cast<double>(first_cell[1]);
        const double first_cell3_d = static_cast<double>(first_cell[2]);
        const double nx1_full      = field.extent(0) - 2 * N_GHOSTS;
        const double nx2_full      = field.extent(1) - 2 * N_GHOSTS;
        const double nx3_full      = field.extent(2) - 2 * N_GHOSTS;
        const auto   first_cell1   = first_cell[0];
        const auto   first_cell2   = first_cell[1];
        const auto   first_cell3   = first_cell[2];

        const auto nx1_dwn = static_cast<ncells_t>(
          math::ceil((nx1_full - first_cell1_d) / dwn1));
        const auto nx2_dwn = static_cast<ncells_t>(
          math::ceil((nx2_full - first_cell2_d) / dwn2));
        const auto nx3_dwn = static_cast<ncells_t>(
          math::ceil((nx3_full - first_cell3_d) / dwn3));

        output_field = array_t<real_t***> { "output_field", nx1_dwn, nx2_dwn, nx3_dwn };
        Kokkos::parallel_for(
          "outputField",
          CreateRangePolicy<Dim::_3D>({ 0, 0, 0 }, { nx1_dwn, nx2_dwn, nx3_dwn }),
          Lambda(cellidx_t i1, cellidx_t i2, cellidx_t i3) {
            output_field(i1, i2, i3) = field(first_cell1 + i1 * dwn1 + N_GHOSTS,
                                             first_cell2 + i2 * dwn2 + N_GHOSTS,
                                             first_cell3 + i3 * dwn3 + N_GHOSTS,
                                             comp);
          });
      }
    }
    auto output_field_h = Kokkos::create_mirror_view(output_field);
    Kokkos::deep_copy(output_field_h, output_field);
    writer.Put(var, output_field_h, adios2::Mode::Deferred);
    // Keep the host mirror (and via Kokkos refcount, its allocation) alive
    // until EndStep runs the deferred PerformPuts.
    keepalive.emplace_back(output_field_h);
  }

  template <Dimension D, int N>
  void Writer::writeField(const std::vector<std::string>& names,
                          const ndfield_t<D, N>&          fld,
                          const std::vector<size_t>&      addresses) {
    raise::ErrorIf(addresses.size() > N,
                   "addresses vector size must be less than N",
                   HERE);
    raise::ErrorIf(names.size() != addresses.size(),
                   "# of names != # of addresses ",
                   HERE);
    for (auto i { 0u }; i < addresses.size(); ++i) {
      WriteField<D, N>(m_io,
                       m_writer,
                       m_keepalive,
                       names[i],
                       fld,
                       addresses[i],
                       m_dwn,
                       m_flds_l_first,
                       m_flds_ghosts,
                       m_flds_l_corner_dwn,
                       m_flds_l_shape_dwn);
    }
  }

  void Writer::writeParticleQuantity(const array_t<real_t*>& array,
                                     npart_t                 glob_total,
                                     npart_t                 loc_offset,
                                     const std::string&      varname) {
    auto var = m_io.InquireVariable<real_t>(varname);
    var.SetShape({ glob_total });
    var.SetSelection(
      adios2::Box<adios2::Dims>({ loc_offset }, { array.extent(0) }));
    auto array_h = Kokkos::create_mirror_view(array);
    Kokkos::deep_copy(array_h, array);
    m_writer.Put<real_t>(var, array_h, adios2::Mode::Deferred);
    m_keepalive.emplace_back(array_h);
  }

  template <uint8_t N, class HostView>
  void PutSpectrumSpatial(adios2::IO&            io,
                          adios2::Engine&        writer,
                          std::vector<std::any>& keepalive,
                          const std::string&     varname,
                          const HostView&        counts_h) {
    auto var = io.InquireVariable<real_t>(varname);

    adios2::Dims start(N, 0u), count(N, 0u), zeros(N, 0u);
    auto         ntot { 1ul };
    for (auto d { 0u }; d < N; ++d) {
      count[d]  = counts_h.extent(d);
      ntot     *= counts_h.extent(d);
    }

#if defined(MPI_ENABLED)
    HostView counts_h_all { "counts_h_all", counts_h.layout() };
    int      rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(counts_h.data(),
               counts_h_all.data(),
               static_cast<int>(ntot),
               mpi::get_type<real_t>(),
               MPI_SUM,
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank == MPI_ROOT_RANK) {
      var.SetSelection(adios2::Box<adios2::Dims>(start, count));
      writer.Put<real_t>(var, counts_h_all.data(), adios2::Mode::Deferred);
      keepalive.emplace_back(counts_h_all);
    } else {
      var.SetSelection(adios2::Box<adios2::Dims>(start, zeros));
      writer.Put<real_t>(var, nullptr, adios2::Mode::Sync);
    }
#else
    var.SetSelection(adios2::Box<adios2::Dims>(start, count));
    writer.Put<real_t>(var, counts_h.data(), adios2::Mode::Deferred);
    keepalive.emplace_back(counts_h);
#endif
  }

  void Writer::writeSpectrum(const array_t<real_t*>& counts,
                             const std::string&      varname) {
    auto var      = m_io.InquireVariable<real_t>(varname);
    auto counts_h = Kokkos::create_mirror_view(counts);
    Kokkos::deep_copy(counts_h, counts);
#if defined(MPI_ENABLED)
    array_t<real_t*> counts_all { "counts_all", counts.extent(0) };
    auto             counts_h_all = Kokkos::create_mirror_view(counts_all);
    int              rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(counts_h.data(),
               counts_h_all.data(),
               static_cast<int>(counts_h.extent(0)),
               mpi::get_type<real_t>(),
               MPI_SUM,
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank == MPI_ROOT_RANK) {
      var.SetSelection(
        adios2::Box<adios2::Dims>({ 0u }, { counts_h_all.extent(0) }));
      m_writer.Put<real_t>(var, counts_h_all, adios2::Mode::Deferred);
      m_keepalive.emplace_back(counts_h_all);
    } else {
      var.SetSelection(adios2::Box<adios2::Dims>({ 0u }, { 0u }));
      m_writer.Put<real_t>(var, nullptr);
    }
#else
    var.SetSelection(adios2::Box<adios2::Dims>({}, { counts.extent(0) }));
    m_writer.Put<real_t>(var, counts_h, adios2::Mode::Deferred);
    m_keepalive.emplace_back(counts_h);
#endif
  }

  template <uint8_t N>
  void Writer::writeSpectrumSpatial(const nddata_t<N, real_t>& counts,
                                    const std::string&         varname) {
    static_assert(N >= 2 and N <= 4, "writeSpectrumSpatial: N must be 2, 3 or 4");
    // host-resident copy, layout inherited from `counts`
    auto counts_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        counts);
    using counts_h_t = decltype(counts_h);
    using layout_t   = typename counts_h_t::array_layout;
    if constexpr (std::is_same<layout_t, Kokkos::LayoutRight>::value) {
      PutSpectrumSpatial<N>(m_io, m_writer, m_keepalive, varname, counts_h);
    } else {
      // ADIOS2 reads the raw buffer as row-major: remap on the host
      using counts_rm_t =
        Kokkos::View<typename counts_h_t::data_type, Kokkos::LayoutRight, Kokkos::HostSpace>;
      counts_rm_t counts_rm {};
      if constexpr (N == 2) {
        counts_rm = counts_rm_t { "counts_rm",
                                  counts_h.extent(0),
                                  counts_h.extent(1) };
      } else if constexpr (N == 3) {
        counts_rm = counts_rm_t { "counts_rm",
                                  counts_h.extent(0),
                                  counts_h.extent(1),
                                  counts_h.extent(2) };
      } else {
        counts_rm = counts_rm_t { "counts_rm",
                                  counts_h.extent(0),
                                  counts_h.extent(1),
                                  counts_h.extent(2),
                                  counts_h.extent(3) };
      }
      Kokkos::deep_copy(counts_rm, counts_h);
      PutSpectrumSpatial<N>(m_io, m_writer, m_keepalive, varname, counts_rm);
    }
  }

  void Writer::writeSpectrumBins(const array_t<real_t*>& e_bins,
                                 const std::string&      varname) {
    auto var      = m_io.InquireVariable<real_t>(varname);
    auto e_bins_h = Kokkos::create_mirror_view(e_bins);
    Kokkos::deep_copy(e_bins_h, e_bins);
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == MPI_ROOT_RANK) {
      var.SetSelection(adios2::Box<adios2::Dims>({ 0u }, { e_bins_h.extent(0) }));
      m_writer.Put<real_t>(var, e_bins_h.data(), adios2::Mode::Deferred);
      m_keepalive.emplace_back(e_bins_h);
    } else {
      var.SetSelection(adios2::Box<adios2::Dims>({ 0u }, { 0u }));
      m_writer.Put<real_t>(var, nullptr, adios2::Mode::Sync);
    }
#else
    var.SetSelection(adios2::Box<adios2::Dims>({}, { e_bins_h.extent(0) }));
    m_writer.Put<real_t>(var, e_bins_h, adios2::Mode::Deferred);
    m_keepalive.emplace_back(e_bins_h);
#endif
  }

  void Writer::writeMesh(unsigned short                  dim,
                         const array_t<real_t*>&         xc,
                         const array_t<real_t*>&         xe,
                         const std::vector<std::size_t>& loc_off_sz) {
    // Per-step (start, count) for the per-rank slab; tracks the (possibly
    // rebalanced) layout cached by setLocalLayout().
    auto varc = m_io.InquireVariable<real_t>("X" + std::to_string(dim + 1));
    auto vare = m_io.InquireVariable<real_t>("X" + std::to_string(dim + 1) + "e");
    // m_flds_l_corner_dwn / m_flds_l_shape_dwn are reversed for non-LayoutRight
    // (see defineMeshLayout / setLocalLayout); m_flds_l_corner / m_flds_l_shape
    // / m_flds_g_shape are not. Map the dim-order index to the dwn-array index.
    constexpr bool layout_right = std::is_same<typename ndfield_t<Dim::_3D, 6>::array_layout,
                                               Kokkos::LayoutRight>::value;
    const auto i_dwn   = layout_right ? static_cast<std::size_t>(dim)
                                      : (m_flds_g_shape.size() - 1u -
                                       static_cast<std::size_t>(dim));
    const auto is_last = (m_flds_l_corner[dim] + m_flds_l_shape[dim] ==
                          m_flds_g_shape[dim]);
    varc.SetSelection(adios2::Box<adios2::Dims>({ m_flds_l_corner_dwn[i_dwn] },
                                                { m_flds_l_shape_dwn[i_dwn] }));
    vare.SetSelection(adios2::Box<adios2::Dims>(
      { m_flds_l_corner_dwn[i_dwn] },
      { m_flds_l_shape_dwn[i_dwn] + (is_last ? 1ul : 0ul) }));
    auto xc_h = Kokkos::create_mirror_view(xc);
    auto xe_h = Kokkos::create_mirror_view(xe);
    Kokkos::deep_copy(xc_h, xc);
    Kokkos::deep_copy(xe_h, xe);
    m_writer.Put(varc, xc_h, adios2::Mode::Deferred);
    m_writer.Put(vare, xe_h, adios2::Mode::Deferred);
    m_keepalive.emplace_back(xc_h);
    m_keepalive.emplace_back(xe_h);
    auto vard = m_io.InquireVariable<std::size_t>(
      "N" + std::to_string(dim + 1) + "l");
    // loc_off_sz is a caller-side local; copy into keepalive so the pointer
    // we hand to ADIOS2 stays valid until EndStep.
    auto loc_off_sz_copy = std::make_shared<std::vector<std::size_t>>(loc_off_sz);
    m_writer.Put(vard, loc_off_sz_copy->data(), adios2::Mode::Deferred);
    m_keepalive.emplace_back(std::move(loc_off_sz_copy));
  }

  void Writer::beginWriting(WriteModeTags write_mode,
                            timestep_t    tstep,
                            simtime_t     time) {
    raise::ErrorIf(write_mode == WriteMode::None, "None is not a valid mode", HERE);
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (m_active_mode != WriteMode::None) {
      raise::Fatal("Already writing", HERE);
    }
    try {
      path_t filename;

      const std::string ext = (m_engine == "hdf5") ? "h5" : "bp";
      std::string       mode_str;
      if (write_mode == WriteMode::Fields) {
        mode_str = "fields";
      } else if (write_mode == WriteMode::Particles) {
        mode_str = "particles";
      } else if (write_mode == WriteMode::Spectra) {
        mode_str = "spectra";
      } else {
        raise::Fatal("Unknown write mode", HERE);
      }
      CallOnce(
        [](auto&& main_path, auto&& mode_path) {
          if (!std::filesystem::exists(main_path)) {
            std::filesystem::create_directory(main_path);
          }
          if (!std::filesystem::exists(main_path / mode_path)) {
            std::filesystem::create_directory(main_path / mode_path);
          }
        },
        m_root,
        mode_str);
#if defined(MPI_ENABLED)
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      filename = m_root / path_t(mode_str) /
                 fmt::format("%s.%08lu.%s", mode_str.c_str(), tstep, ext.c_str());
      m_mode   = adios2::Mode::Write;
      m_writer = m_io.Open(filename, m_mode);
      m_writer.BeginStep();
      m_writer.Put(m_io.InquireVariable<timestep_t>("Step"), &tstep);
      m_writer.Put(m_io.InquireVariable<simtime_t>("Time"), &time);
      m_active_mode = write_mode;
    } catch (std::exception& e) {
      raise::Fatal(e.what(), HERE);
    }
  }

  void Writer::endWriting(WriteModeTags write_mode) {
    raise::ErrorIf(write_mode == WriteMode::None, "None is not a valid mode", HERE);
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (m_active_mode == WriteMode::None) {
      raise::Fatal("Not writing", HERE);
    }
    if (m_active_mode != write_mode) {
      raise::Fatal("Writing mode mismatch", HERE);
    }
    m_active_mode = WriteMode::None;
    m_writer.EndStep();
    // EndStep flushes all Deferred Put buffers; safe to release keepalive.
    m_keepalive.clear();
    m_writer.Close();
  }

#define WRITE_FIELD(D, N)                                                      \
  template void Writer::writeField<D, N>(const std::vector<std::string>&,      \
                                         const ndfield_t<D, N>&,               \
                                         const std::vector<std::size_t>&);     \
  template void WriteField<D, N>(adios2::IO&,                                  \
                                 adios2::Engine&,                              \
                                 std::vector<std::any>&,                       \
                                 const std::string&,                           \
                                 const ndfield_t<D, N>&,                       \
                                 std::size_t,                                  \
                                 std::vector<unsigned int>,                    \
                                 std::vector<ncells_t>,                        \
                                 bool,                                         \
                                 const adios2::Dims&,                          \
                                 const adios2::Dims&);
  WRITE_FIELD(Dim::_1D, 3)
  WRITE_FIELD(Dim::_1D, 6)
  WRITE_FIELD(Dim::_2D, 3)
  WRITE_FIELD(Dim::_2D, 6)
  WRITE_FIELD(Dim::_3D, 3)
  WRITE_FIELD(Dim::_3D, 6)
#undef WRITE_FIELD

#define WRITE_SPECTRUM_SPATIAL(N)                                              \
  template void Writer::writeSpectrumSpatial<N>(const nddata_t<N, real_t>&,    \
                                                const std::string&);
  WRITE_SPECTRUM_SPATIAL(2)
  WRITE_SPECTRUM_SPATIAL(3)
  WRITE_SPECTRUM_SPATIAL(4)
#undef WRITE_SPECTRUM_SPATIAL

} // namespace out
