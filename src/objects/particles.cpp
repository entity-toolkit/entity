#include "global.h"
#include "particles.h"
#include "arrays.h"

#include <map>
#include <variant>
#include <vector>
#include <cstddef>
#include <string_view>

namespace ntt::particles {

ParticleSpecies::ParticleSpecies() {
  m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("i1", &m_i1));
  m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("i2", &m_i2));
  m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("i3", &m_i3));
}

auto ParticleSpecies::getSizeInBytes() -> std::size_t {
  std::size_t size_in_bytes {0};
  for (auto const& [name, array] : m_particle_arrays) {
    size_in_bytes += std::visit([](auto&& arr) -> std::size_t {
      return arr->getSizeInBytes();
    }, array);
  }
  return size_in_bytes;
}

}
