/**
 * @file output/particles.h
 * @brief Defines the metadata for particle output
 * @implements
 *   - out::OutputSpecies
 */

#ifndef OUTPUT_PARTICLES_H
#define OUTPUT_PARTICLES_H

#include <string>

namespace out {

  class OutputSpecies {
    const unsigned short m_sp;

  public:
    OutputSpecies(unsigned short sp) : m_sp { sp } {}

    ~OutputSpecies() = default;

    [[nodiscard]]
    auto species() const -> unsigned short {
      return m_sp;
    }

    [[nodiscard]]
    auto name(const std::string& q, unsigned short c) const -> std::string {
      return "p" + q + (c == 0 ? "" : std::to_string(c)) + "_" +
             std::to_string(m_sp);
    }
  };

} // namespace out

#endif // OUTPUT_PARTICLES_H
