#ifndef OBJECTS_PARTICLES_H
#define OBJECTS_PARTICLES_H

#include "global.h"

namespace ntt {

template<template<typename T = std::nullptr_t> class D>
class ParticleSpecies {
  D<> m_dim;
  float m_mass, m_charge;

  std::size_t m_npart{0}, m_maxnpart;
public:
  ParticleSpecies(const float& m, const float& ch, const std::size_t& maxnpart);
  ~ParticleSpecies() = default;
};

}

#endif
