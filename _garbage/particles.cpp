#include "global.h"
#include "particles.h"
#include "arrays.h"

#include <map>
#include <variant>
#include <vector>
#include <cstddef>
#include <string_view>
#include <stdexcept>
#include <iostream>

namespace ntt::particles {

void ParticleSpecies::printDetails(std::ostream &os, int s) {
  os << ". . [species #" << s << "]\n";
  os << "     mass: " << m_mass << "\n";
  os << "     charge: " << m_charge << "\n";
  os << "     pusher: " << stringifyParticlePusher(m_pusher) << "\n";
  os << "     used: " << m_npart << "/" << m_maxnpart << "\n";
}
void ParticleSpecies::printDetails(std::ostream &os) {
  printDetails(os, 0);
}
void ParticleSpecies::printDetails(int s) {
  printDetails(std::cout, s);
}
void ParticleSpecies::printDetails() {
  printDetails(std::cout);
}

void ParticleSpecies::allocate() {
  allocate(m_maxnpart);
}

void ParticleSpecies::allocate(std::size_t maxnpart) {
# ifdef DEBUG
  if (m_dimension == UNDEFINED_D) {
    throw std::runtime_error("# Error: undefined dimensionality in `ParticleSpecies`.");
  }
  if (m_allocated) {
    throw std::runtime_error("# Error: `ParticleSpecies` already allocated.");
  }
# endif
  // pointer to each parameter is stored in the `m_particle_arrays`
  m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("x1", &m_x1));

  if ((m_dimension == TWO_D) || (m_dimension == THREE_D)) {
    m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("x2", &m_x2));
  }
  if (m_dimension == THREE_D) {
    m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("x3", &m_x3));
  }

  m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("ux1", &m_ux1));
  m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("ux2", &m_ux2));
  m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("ux3", &m_ux3));
  m_particle_arrays.insert(std::pair<std::string_view, ArrayPntr_t>("weight", &m_weight));

  m_maxnpart = maxnpart;
  for (auto const& [name, array] : m_particle_arrays) {
    std::visit([&maxnpart](auto&& arr) {
      arr->allocate(maxnpart);
    }, array);
  }
  m_allocated = true;
}

void ParticleSpecies::addParticle(std::vector<real_t> x, std::vector<real_t> ux) {
  addParticle(x, ux, 1.0);
}

void ParticleSpecies::addParticle(std::vector<real_t> x, std::vector<real_t> ux, float weight) {
  // check that enough is allocated already
# ifdef DEBUG
  if (!m_allocated || (m_npart >= m_maxnpart)) {
    throw std::runtime_error("# Error: not enough particles allocated or particles not allocated at all.");
  }
  if ((x.size() < 1) || (x.size() > 3)) {
    throw std::runtime_error("# Error: wrong addParticle in `ParticleSpecies`.");
  }
  if (ux.size() != 3) {
    throw std::runtime_error("# Error: wrong addParticle in `ParticleSpecies`.");
  }
# endif
  // coordinate
  m_x1.set(m_npart, x[0]);
  if (x.size() > 1) {
    m_x2.set(m_npart, x[1]);
  }
  if (x.size() > 2) {
    m_x3.set(m_npart, x[2]);
  }
  // 4-vel
  m_ux1.set(m_npart, ux[0]);
  m_ux2.set(m_npart, ux[1]);
  m_ux3.set(m_npart, ux[2]);
  // weight
  m_weight.set(m_npart, weight);
  ++m_npart;
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
