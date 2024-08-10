#include "checkpoint/writer.h"

#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "framework/parameters.h"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace checkpoint {

  void Writer::init(adios2::ADIOS* ptr_adios,
                    std::size_t    interval,
                    long double    interval_time,
                    int            keep) {
    m_keep    = keep;
    m_enabled = keep != 0;
    if (!m_enabled) {
      return;
    }
    m_tracker.init("checkpoint", interval, interval_time);
    p_adios = ptr_adios;
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);

    m_io = p_adios->DeclareIO("Entity::Checkpoint");
    m_io.SetEngine("BPFile");

    m_io.DefineVariable<std::size_t>("Step");
    m_io.DefineVariable<long double>("Time");
    m_io.DefineAttribute("NGhosts", ntt::N_GHOSTS);

    const std::filesystem::path save_path { "checkpoints" };
    if (!std::filesystem::exists(save_path)) {
      std::filesystem::create_directory(save_path);
    }
    p_adios->EnterComputationBlock();
  }

  auto Writer::shouldSave(std::size_t step, long double time) -> bool {
    return m_enabled and m_tracker.shouldWrite(step, time);
  }

  void Writer::beginSaving(const ntt::SimulationParams& params,
                           std::size_t                  step,
                           long double                  time) {
    raise::ErrorIf(!m_enabled, "Checkpoint is not enabled", HERE);
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    p_adios->ExitComputationBlock();
    if (m_writing_mode) {
      raise::Fatal("Already writing", HERE);
    }
    m_writing_mode = true;
    try {
      auto fname = fmt::format("checkpoints/step-%08lu.bp", step);
      m_writer   = m_io.Open(fname, adios2::Mode::Write);
      m_written.push_back(fname);
    } catch (std::exception& e) {
      raise::Fatal(e.what(), HERE);
    }

    // write the metadata
    std::ofstream metadata;
    metadata.open(fmt::format("checkpoints/meta-%08lu.toml", step).c_str());
    metadata << params.data() << std::endl;
    metadata.close();

    m_writer.BeginStep();
    m_writer.Put(m_io.InquireVariable<std::size_t>("Step"), &step);
    m_writer.Put(m_io.InquireVariable<long double>("Time"), &time);
  }

  void Writer::endSaving() {
    raise::ErrorIf(p_adios == nullptr, "ADIOS pointer is null", HERE);
    if (!m_writing_mode) {
      raise::Fatal("Not writing", HERE);
    }
    m_writing_mode = false;
    m_writer.EndStep();
    m_writer.Close();
    p_adios->EnterComputationBlock();

    // optionally remove the oldest checkpoint
    if (m_keep > 0 and m_written.size() > (std::size_t)m_keep) {
      const auto oldest = m_written.front();
      if (std::filesystem::exists(oldest)) {
        std::filesystem::remove_all(oldest);
        m_written.erase(m_written.begin());
      } else {
        raise::Warning("Checkpoint file does not exist for some reason", HERE);
      }
    }
  }
} // namespace checkpoint
