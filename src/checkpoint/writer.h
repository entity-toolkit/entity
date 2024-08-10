/**
 * @file checkpoint/writer.h
 * @brief Class that dumps checkpoints
 * @implements
 *   - checkpoint::Writer
 * @cpp:
 *   - writer.cpp
 * @namespaces:
 *   - save::
 */

#ifndef CHECKPOINT_WRITER_H
#define CHECKPOINT_WRITER_H

#include "global.h"

#include "utils/tools.h"

#include "framework/parameters.h"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <vector>

namespace checkpoint {

  class Writer {
    adios2::ADIOS* p_adios { nullptr };

    adios2::IO     m_io;
    adios2::Engine m_writer;

    tools::Tracker m_tracker {};

    bool m_writing_mode { false };

    std::vector<std::string> m_written;

    int  m_keep;
    bool m_enabled;

  public:
    Writer() {}

    ~Writer() = default;

    void init(adios2::ADIOS*, std::size_t, long double, int);

    auto shouldSave(std::size_t, long double) -> bool;

    void beginSaving(const ntt::SimulationParams&, std::size_t, long double);
    void endSaving();

    auto enabled() const -> bool {
      return m_enabled;
    }
  };

} // namespace checkpoint

#endif // CHECKPOINT_WRITER_H
