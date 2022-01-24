#ifndef FRAMEWORK_FIELDS_H
#define FRAMEWORK_FIELDS_H

#include "global.h"

namespace ntt {
  enum em { ex1 = 0, ex2 = 1, ex3 = 2, bx1 = 3, bx2 = 4, bx3 = 5 };
  enum cur { jx1 = 0, jx2 = 1, jx3 = 2 };

  /**
   * Container for the fields. Used a parent class for the Meshblock.
   *
   * @tparam D Dimension.
   * @tparam S Simulation type.
   */
  template <Dimension D, SimulationType S>
  class Fields {
  public:
    /**
     * EM fields stored as Kokkos Views of dimension D * 6.
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field component.
     * @note Address : em(i, j, k, em::***).
     */
    RealFieldND<D, 6> em;
    /**
     * Current fields stored as Kokkos Views of dimension D * 6.
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field component.
     * @note Address : cur(i, j, k, cur::***).
     */
    RealFieldND<D, 3> cur;

    /**
     * Constructor for the fields container. Also sets the active cell sizes and ranges.
     *
     * @param res resolution vector of size D (dimension).
     */
    Fields(std::vector<unsigned int> res);
    ~Fields() = default;
  };

} // namespace ntt

#endif
