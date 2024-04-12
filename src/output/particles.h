/**
 * @file output/particles.h
 * @brief Defines the metadata for particle output
 * @implements
 *   - out::OutputParticle
 * @cpp:
 *   - particles.cpp
 * @depends:
 *   - enums.h
 *   - utils/error.h
 *   - output/utils/getspecies.h
 */

#ifndef OUTPUT_PARTICLES_H
#define OUTPUT_PARTICLES_H

#include "enums.h"

#include <string>
#include <vector>

using namespace ntt;

namespace out {

  class OutputParticle {
    const unsigned short m_sp;

  public:
    OutputParticle(unsigned short sp) : m_sp { sp } {}

    ~OutputParticle() = default;

    [[nodiscard]]
    auto species() const -> unsigned short {
      return m_sp;
    }
  };

} // namespace out

#endif // OUTPUT_PARTICLES_H