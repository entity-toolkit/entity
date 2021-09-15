/* *
   *
   *  Functionality for particle species
   *
   *  @namespace ntt::particles
   *
 * */

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

/* *
   *
   *  Class that stores particle species as 1d arrays of properties (velocities, positions, etc).
   *
   *  @parameters:
   *     m_dimension          : dimension of space in which particles live (for 1d -- x2, x3 are not allocated)
   *     m_x[1/2/3]           : ....
   *
   *  @methods:
   *
   *  @comment:
   *
   *  @example:
   *
 * */

using ArrayPntr_t = std::variant<arrays::OneDArray<int>*, arrays::OneDArray<float>*, arrays::OneDArray<double>*, arrays::OneDArray<bool>*>;
using ListOfArrays_t = std::map<std::string_view, ArrayPntr_t>;

class ParticleSpecies {
public:

  Dimension m_dimension{UNDEFINED_D};
  ParticlePusher m_pusher{UNDEFINED_PUSHER};

  // species properties
  float m_mass, m_charge;

  // particle arrays
  arrays::OneDArray<real_t> m_x1, m_x2, m_x3;
  arrays::OneDArray<real_t> m_ux1, m_ux2, m_ux3;
  arrays::OneDArray<float> m_weight;

  ListOfArrays_t m_particle_arrays;

  // information
  std::size_t m_npart{0}, m_maxnpart;
  bool m_allocated{false};

  ParticleSpecies() {}
  ParticleSpecies(const Dimension &dim) : m_dimension(dim) {}
  ParticleSpecies(const ParticlePusher &pusher) : m_pusher(pusher) {}
  ParticleSpecies(const Dimension &dim, const ParticlePusher &pusher) : m_dimension(dim), m_pusher(pusher) {}
  ~ParticleSpecies() = default;

  void setDimension(const Dimension &dim) { m_dimension = dim; }
  void setPusher(const ParticlePusher &pusher) { m_pusher = pusher; }
  void setMass(const float &m) { m_mass = m; }
  void setCharge(const float &ch) { m_charge = ch; }
  void setMaxnpart(const std::size_t &maxnpart) { m_maxnpart = maxnpart; }

  void printDetails(std::ostream &);
  void printDetails(std::ostream &, int s);
  void printDetails(int s);
  void printDetails();

  void addParticle(std::vector<real_t> x, std::vector<real_t> ux);
  void addParticle(std::vector<real_t> x, std::vector<real_t> ux, float weight);
  void allocate();
  void allocate(std::size_t maxnpart);
  [[nodiscard]] auto getSizeInBytes() -> std::size_t;
};

}

#endif
