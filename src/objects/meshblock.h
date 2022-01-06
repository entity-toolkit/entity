#ifndef OBJECTS_MESHBLOCK_H
#define OBJECTS_MESHBLOCK_H

#include "global.h"
#include "grid.h"
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
  struct Meshblock : Fields<D> {
    std::vector<Particles<D>> particles;
    std::shared_ptr<Grid<D>> grid;

    Meshblock(std::vector<real_t>, std::vector<std::size_t>, std::vector<ParticleSpecies>&);

    ~Meshblock() = default;

    void verify(const SimulationParams&);
  };

} // namespace ntt

#endif
