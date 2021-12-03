#ifndef OBJECTS_MESHBLOCK_H
#define OBJECTS_MESHBLOCK_H

#include "global.h"
#include "constants.h"
#include "fields.h"
#include "sim_params.h"
#include "particles.h"

#include <vector>
#include <type_traits>
#include <typeinfo>
#include <cmath>
#include <utility>

namespace ntt {

template <Dimension D>
struct Meshblock : public Fields<D>, public Grid<D> {
  std::vector<Particles<D>> particles;

  Meshblock(std::vector<std::size_t>, std::vector<ParticleSpecies>&);
  ~Meshblock() = default;

  void verify(const SimulationParams&);
};

} // namespace ntt

#endif
