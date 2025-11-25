#include "output/checkpoint.h"

#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>

#include <filesystem>
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

} // namespace checkpoint
