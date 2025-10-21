/**
 * @file checkpoint/writer.h
 * @brief Class that dumps checkpoints
 * @implements
 *   - checkpoint::Writer
 * @cpp:
 *   - writer.cpp
 * @namespaces:
 *   - checkpoint::
 */

#ifndef CHECKPOINT_WRITER_H
#define CHECKPOINT_WRITER_H

#include "enums.h"
#include "global.h"

#include "utils/tools.h"

#include "framework/parameters.h"

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

    void saveAttrs(const ntt::SimulationParams&, simtime_t);

    template <Dimension D, int N>
    void saveField(const std::string&, const ndfield_t<D, N>&);

    void defineFieldVariables(const ntt::SimEngine&,
                              const std::vector<ncells_t>&,
                              const std::vector<ncells_t>&,
                              const std::vector<ncells_t>&);

    [[nodiscard]]
    auto io() -> adios2::IO& {
      return m_io;
    }

    [[nodiscard]]
    auto writer() -> adios2::Engine& {
      return m_writer;
    }

    [[nodiscard]]
    auto enabled() const -> bool {
      return m_enabled;
    }
  };

} // namespace checkpoint

#endif // CHECKPOINT_WRITER_H
