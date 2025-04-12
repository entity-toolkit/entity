/**
 * @file output/spectra.h
 * @brief Defines the metadata for particle spectra
 * @implements
 *   - out::OutputSpectra
 */

#ifndef OUTPUT_SPECTRA_H
#define OUTPUT_SPECTRA_H

#include "global.h"

#include <string>

namespace out {

  class OutputSpectra {
    const spidx_t m_sp;

  public:
    OutputSpectra(spidx_t sp) : m_sp { sp } {}

    ~OutputSpectra() = default;

    [[nodiscard]]
    auto species() const -> spidx_t {
      return m_sp;
    }

    [[nodiscard]]
    auto name() const -> std::string {
      return "sN_" + std::to_string(m_sp);
    }
  };

} // namespace out

#endif // OUTPUT_SPECTRA_H
