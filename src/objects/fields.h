#ifndef OBJECTS_FIELDS_H
#define OBJECTS_FIELDS_H

#include "global.h"
#include "grid.h"

#include <vector>

namespace ntt {

namespace fld {

  enum FieldSelector { ex1 = 0, ex2, ex3, bx1, bx2, bx3 };

  enum CurrSelector { jx1 = 0, jx2, jx3 };

} // namespace fld

template <Dimension D>
struct Fields {
  // sizes of these arrays are ...
  //   resolution + 2 * N_GHOSTS in every direction
  RealFieldND<D, 6> em_fields;
  RealFieldND<D, 3> j_fields;

  Fields(std::vector<std::size_t> res);
  ~Fields() = default;

  [[nodiscard]] auto get_imin() const -> long int { return N_GHOSTS; }
  [[nodiscard]] auto get_imax() const -> long int { return N_GHOSTS + this->m_resolution[0]; }
  [[nodiscard]] auto get_jmin() const -> long int { return N_GHOSTS; }
  [[nodiscard]] auto get_jmax() const -> long int { return N_GHOSTS + this->m_resolution[1]; }
  [[nodiscard]] auto get_kmin() const -> long int { return N_GHOSTS; }
  [[nodiscard]] auto get_kmax() const -> long int { return N_GHOSTS + this->m_resolution[2]; }

  auto loopActiveCells() -> RangeND<D>;
};

} // namespace ntt

#endif
