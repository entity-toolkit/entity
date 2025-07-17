#include "checkpoint/writer.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "framework/parameters.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace checkpoint {

  void Writer::init(adios2::ADIOS*     ptr_adios,
                    const path_t&      checkpoint_root,
                    timestep_t         interval,
                    simtime_t          interval_time,
                    int                keep,
                    const std::string& walltime) {
    m_keep            = keep;
    m_checkpoint_root = checkpoint_root;
    m_enabled         = keep != 0;
    if (not m_enabled) {
      return;
    }
    m_tracker.init("checkpoint", interval, interval_time, walltime);
    p_adios = ptr_adios;
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);

    m_io = p_adios->DeclareIO("Entity::Checkpoint");
    m_io.SetEngine("BPFile");

    m_io.DefineVariable<timestep_t>("Step");
    m_io.DefineVariable<simtime_t>("Time");
    m_io.DefineAttribute("NGhosts", ntt::N_GHOSTS);

    CallOnce(
      [](auto&& checkpoint_root) {
        if (!std::filesystem::exists(checkpoint_root)) {
          std::filesystem::create_directory(checkpoint_root);
        }
      },
      m_checkpoint_root);
  }

  void Writer::defineFieldVariables(const ntt::SimEngine&        S,
                                    const std::vector<ncells_t>& glob_shape,
                                    const std::vector<ncells_t>& loc_corner,
                                    const std::vector<ncells_t>& loc_shape) {
    auto gs6 = std::vector<ncells_t>(glob_shape.begin(), glob_shape.end());
    auto lc6 = std::vector<ncells_t>(loc_corner.begin(), loc_corner.end());
    auto ls6 = std::vector<ncells_t>(loc_shape.begin(), loc_shape.end());
    gs6.push_back(6);
    lc6.push_back(0);
    ls6.push_back(6);

    m_io.DefineVariable<real_t>("em", gs6, lc6, ls6);
    if (S == ntt::SimEngine::GRPIC) {
      m_io.DefineVariable<real_t>("em0", gs6, lc6, ls6);
      auto gs3 = std::vector<ncells_t>(glob_shape.begin(), glob_shape.end());
      auto lc3 = std::vector<ncells_t>(loc_corner.begin(), loc_corner.end());
      auto ls3 = std::vector<ncells_t>(loc_shape.begin(), loc_shape.end());
      gs3.push_back(3);
      lc3.push_back(0);
      ls3.push_back(3);
      m_io.DefineVariable<real_t>("cur0", gs3, lc3, ls3);
    }
  }

  void Writer::defineParticleVariables(const ntt::Coord& C,
                                       Dimension         dim,
                                       std::size_t       nspec,
                                       const std::vector<unsigned short>& nplds) {
    raise::ErrorIf(nplds.size() != nspec,
                   "Number of payloads does not match the number of species",
                   HERE);
    for (auto s { 0u }; s < nspec; ++s) {
      m_io.DefineVariable<npart_t>(fmt::format("s%d_npart", s + 1),
                                   { adios2::UnknownDim },
                                   { adios2::UnknownDim },
                                   { adios2::UnknownDim });

      for (auto d { 0u }; d < dim; ++d) {
        m_io.DefineVariable<int>(fmt::format("s%d_i%d", s + 1, d + 1),
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim });
        m_io.DefineVariable<prtldx_t>(fmt::format("s%d_dx%d", s + 1, d + 1),
                                      { adios2::UnknownDim },
                                      { adios2::UnknownDim },
                                      { adios2::UnknownDim });
        m_io.DefineVariable<int>(fmt::format("s%d_i%d_prev", s + 1, d + 1),
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim });
        m_io.DefineVariable<prtldx_t>(fmt::format("s%d_dx%d_prev", s + 1, d + 1),
                                      { adios2::UnknownDim },
                                      { adios2::UnknownDim },
                                      { adios2::UnknownDim });
      }

      if (dim == Dim::_2D and C != ntt::Coord::Cart) {
        m_io.DefineVariable<real_t>(fmt::format("s%d_phi", s + 1),
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim });
      }

      for (auto d { 0u }; d < 3; ++d) {
        m_io.DefineVariable<real_t>(fmt::format("s%d_ux%d", s + 1, d + 1),
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim },
                                    { adios2::UnknownDim });
      }

      m_io.DefineVariable<short>(fmt::format("s%d_tag", s + 1),
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim },
                                 { adios2::UnknownDim });
      m_io.DefineVariable<real_t>(fmt::format("s%d_weight", s + 1),
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim },
                                  { adios2::UnknownDim });
      if (nplds[s] > 0) {
        m_io.DefineVariable<real_t>(fmt::format("s%d_plds", s + 1),
                                    { adios2::UnknownDim, nplds[s] },
                                    { adios2::UnknownDim, 0 },
                                    { adios2::UnknownDim, nplds[s] });
      }
    }
  }

  auto Writer::shouldSave(timestep_t step, simtime_t time) -> bool {
    return m_enabled and m_tracker.shouldWrite(step, time);
  }

  void Writer::beginSaving(timestep_t step, simtime_t time) {
    raise::ErrorIf(!m_enabled, "Checkpoint is not enabled", HERE);
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (m_writing_mode) {
      raise::Fatal("Already writing", HERE);
    }
    m_writing_mode = true;
    try {
      const auto filename = m_checkpoint_root / fmt::format("step-%08lu.bp", step);
      const auto metafilename = m_checkpoint_root /
                                fmt::format("meta-%08lu.toml", step);
      m_writer = m_io.Open(filename, adios2::Mode::Write);
      m_written.push_back({ filename, metafilename });
      logger::Checkpoint(fmt::format("Writing checkpoint to %s and %s",
                                     filename.c_str(),
                                     metafilename.c_str()),
                         HERE);
    } catch (std::exception& e) {
      raise::Fatal(e.what(), HERE);
    }

    m_writer.BeginStep();
    m_writer.Put(m_io.InquireVariable<timestep_t>("Step"), &step);
    m_writer.Put(m_io.InquireVariable<simtime_t>("Time"), &time);
  }

  void Writer::endSaving() {
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (!m_writing_mode) {
      raise::Fatal("Not writing", HERE);
    }
    m_writing_mode = false;
    m_writer.EndStep();
    m_writer.Close();

    // optionally remove the oldest checkpoint
    CallOnce([&]() {
      if (m_keep > 0 and m_written.size() > (std::size_t)m_keep) {
        const auto oldest = m_written.front();
        if (std::filesystem::exists(oldest.first) and
            std::filesystem::exists(oldest.second)) {
          std::filesystem::remove_all(oldest.first);
          std::filesystem::remove(oldest.second);
          m_written.erase(m_written.begin());
        } else {
          raise::Warning("Checkpoint file does not exist for some reason", HERE);
        }
      }
    });
  }

  template <typename T>
  void Writer::savePerDomainVariable(const std::string& varname,
                                     std::size_t        total,
                                     std::size_t        offset,
                                     T                  data) {
    auto var = m_io.InquireVariable<T>(varname);
    var.SetShape({ total });
    var.SetSelection(adios2::Box<adios2::Dims>({ offset }, { 1 }));
    m_writer.Put(var, &data);
  }

  void Writer::saveAttrs(const ntt::SimulationParams& params, simtime_t time) {
    CallOnce([&]() {
      std::ofstream metadata;
      if (m_written.empty()) {
        raise::Fatal("No checkpoint file to save metadata", HERE);
      }
      metadata.open(m_written.back().second.c_str());
      metadata << "[metadata]\n"
               << "  time = " << time << "\n\n"
               << params.data() << std::endl;
      metadata.close();
    });
  }

  template <Dimension D, int N>
  void Writer::saveField(const std::string&     fieldname,
                         const ndfield_t<D, N>& field) {
    auto field_h = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(field_h, field);
    m_writer.Put(m_io.InquireVariable<real_t>(fieldname),
                 field_h.data(),
                 adios2::Mode::Sync);
  }

  template <typename T>
  void Writer::saveParticleQuantity(const std::string& quantity,
                                    npart_t            glob_total,
                                    npart_t            loc_offset,
                                    npart_t            loc_size,
                                    const array_t<T*>& data) {
    const auto slice = range_tuple_t(0, loc_size);
    auto       var   = m_io.InquireVariable<T>(quantity);

    var.SetShape({ glob_total });
    var.SetSelection(adios2::Box<adios2::Dims>({ loc_offset }, { loc_size }));

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    auto data_sub = Kokkos::subview(data_h, slice);
    m_writer.Put(var, data_sub.data(), adios2::Mode::Sync);
  }

  void Writer::saveParticlePayloads(const std::string&       quantity,
                                    std::size_t              nplds,
                                    npart_t                  glob_total,
                                    npart_t                  loc_offset,
                                    npart_t                  loc_size,
                                    const array_t<real_t**>& data) {
    const auto slice = range_tuple_t(0, loc_size);
    auto       var   = m_io.InquireVariable<real_t>(quantity);

    var.SetShape({ glob_total, nplds });
    var.SetSelection(
      adios2::Box<adios2::Dims>({ loc_offset, 0 }, { loc_size, nplds }));

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    auto data_sub = Kokkos::subview(data_h, slice, range_tuple_t(0, nplds));
    m_writer.Put(var, data_sub.data(), adios2::Mode::Sync);
  }

#define CHECKPOINT_PERDOMAIN_VARIABLE(T)                                       \
  template void Writer::savePerDomainVariable<T>(const std::string&,           \
                                                 std::size_t,                  \
                                                 std::size_t,                  \
                                                 T);
  CHECKPOINT_PERDOMAIN_VARIABLE(int)
  CHECKPOINT_PERDOMAIN_VARIABLE(float)
  CHECKPOINT_PERDOMAIN_VARIABLE(double)
  CHECKPOINT_PERDOMAIN_VARIABLE(npart_t)
#undef CHECKPOINT_PERDOMAIN_VARIABLE

#define CHECKPOINT_FIELD(D, N)                                                 \
  template void Writer::saveField<D, N>(const std::string&,                    \
                                        const ndfield_t<D, N>&);
  CHECKPOINT_FIELD(Dim::_1D, 3)
  CHECKPOINT_FIELD(Dim::_1D, 6)
  CHECKPOINT_FIELD(Dim::_2D, 3)
  CHECKPOINT_FIELD(Dim::_2D, 6)
  CHECKPOINT_FIELD(Dim::_3D, 3)
  CHECKPOINT_FIELD(Dim::_3D, 6)
#undef CHECKPOINT_FIELD

#define CHECKPOINT_PARTICLE_QUANTITY(T)                                        \
  template void Writer::saveParticleQuantity<T>(const std::string&,            \
                                                npart_t,                       \
                                                npart_t,                       \
                                                npart_t,                       \
                                                const array_t<T*>&);
  CHECKPOINT_PARTICLE_QUANTITY(int)
  CHECKPOINT_PARTICLE_QUANTITY(float)
  CHECKPOINT_PARTICLE_QUANTITY(double)
  CHECKPOINT_PARTICLE_QUANTITY(short)
#undef CHECKPOINT_PARTICLE_QUANTITY

} // namespace checkpoint
