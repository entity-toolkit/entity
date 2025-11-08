/**
 * @file output/checkpoint.h
 * @brief Class that handles checkpoint writing
 * @implements
 *   - checkpoint::Writer
 * @cpp:
 *   - checkpoint.cpp
 * @namespaces:
 *   - checkpoint::
 */

#ifndef OUTPUT_CHECKPOINT_H
#define OUTPUT_CHECKPOINT_H

#include "global.h"

#include "utils/tools.h"

#include <adios2.h>

#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

  class Writer {
    adios2::ADIOS* p_adios { nullptr };

    adios2::IO     m_io;
    adios2::Engine m_writer;

    tools::Tracker m_tracker {};

    bool m_writing_mode { false };

    std::vector<std::pair<std::string, std::string>> m_written;

    int    m_keep;
    bool   m_enabled;
    path_t m_checkpoint_root;

  public:
    Writer() {}

    ~Writer() = default;

    void init(adios2::ADIOS*,
              const path_t&,
              timestep_t,
              simtime_t,
              int,
              const std::string& = "");

    auto shouldSave(timestep_t, simtime_t) -> bool;

    void beginSaving(timestep_t, simtime_t);
    void endSaving();

    [[nodiscard]]
    auto io() -> adios2::IO& {
      return m_io;
    }

    [[nodiscard]]
    auto writer() -> adios2::Engine& {
      return m_writer;
    }

    [[nodiscard]]
    auto written() const -> const std::vector<std::pair<std::string, std::string>>& {
      return m_written;
    }

    [[nodiscard]]
    auto enabled() const -> bool {
      return m_enabled;
    }
  };

} // namespace checkpoint

#endif // OUTPUT_CHECKPOINT_H
