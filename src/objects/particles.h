#ifndef OBJECTS_PARTICLES_H
#define OBJECTS_PARTICLES_H

#include "global.h"
#include "arrays.h"

#include <map>
#include <variant>
#include <vector>
#include <cstddef>
#include <string_view>

namespace ntt::particles {

using ArrayPntr_t = std::variant<arrays::OneDArray<int>*, arrays::OneDArray<float>*, arrays::OneDArray<double>*, arrays::OneDArray<bool>*>;
using ListOfArrays_t = std::map<std::string_view, ArrayPntr_t>;

class ParticleSpecies {
  // species properties
  float m_mass, m_charge;

  // particle arrays
  arrays::OneDArray<int> m_i1, m_i2, m_i3;
  arrays::OneDArray<float> m_x1, m_x2, m_x3;
  arrays::OneDArray<real_t> m_ux1, m_ux2, m_ux3;
  arrays::OneDArray<float> m_weight;

  ListOfArrays_t m_particle_arrays;

  // information
  std::size_t m_npart{0}, m_maxnpart;
public:
  // TODO: finish this
  ParticleSpecies();
  // TODO: update destructor
  ~ParticleSpecies() = default;

  auto getSizeInBytes() -> std::size_t;
};

}

#endif
