#ifndef FRAMEWORK_FIELDS_H
#define FRAMEWORK_FIELDS_H

#include "wrapper.h"

namespace ntt {
  enum em { ex1 = 0, ex2 = 1, ex3 = 2, bx1 = 3, bx2 = 4, bx3 = 5 };
  enum cur { jx1 = 0, jx2 = 1, jx3 = 2 };

  /**
   * @brief Container for the fields. Used a parent class for the Meshblock.
   * @tparam D Dimension.
   * @tparam S Simulation type.
   */
  template <Dimension D, SimulationType S>
  struct Fields {
    // * * * * * * * * * * * * * * * * * * * *
    // PIC-specific
    // * * * * * * * * * * * * * * * * * * * *

    /**
     * @brief EM fields at current time step stored as Kokkos Views of dimension D * 6.
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component.
     * @note Address : em(i, j, k, em::***).
     *
     * @note Bx1 is stored at (    i, j+1/2, k+1/2, n-1/2)
     * @note Bx2 is stored at (i+1/2,     j, k+1/2, n-1/2)
     * @note Bx3 is stored at (i+1/2, j+1/2,     k, n-1/2)
     *
     * @note Ex1 is stored at (i+1/2,     j,     k,     n)
     * @note Ex2 is stored at (    i, j+1/2,     k,     n)
     * @note Ex3 is stored at (    i,     j, k+1/2,     n)
     */
    ndfield_t<D, 6>        em;
    ndfield_mirror_t<D, 6> em_h;
    /**
     * Current fields at current time step stored as Kokkos Views of dimension D * 3.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x3 for each field
     * component.
     * @note Address : cur(i, j, k, cur::***).
     *
     * @note Jx1 is deposited at (i+1/2,     j,     k, n+1/2)
     * @note Jx2 is deposited at (    i, j+1/2,     k, n+1/2)
     * @note Jx3 is deposited at (    i,     j, k+1/2, n+1/2)
     */
    ndfield_t<D, 3>        cur;
    ndfield_mirror_t<D, 3> cur_h;
    /**
     * Current fields at previous time step stored as Kokkos Views of dimension D * 3.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x3 for each field
     * component.
     * @note Address : cur0(i, j, k, cur::***).
     */
    ndfield_t<D, 3> cur0;
#ifdef GRPIC_SIMTYPE
    // * * * * * * * * * * * * * * * * * * * *
    // GRPIC-specific
    // * * * * * * * * * * * * * * * * * * * *

    /**
     * Auxiliary E and H fields stored as Kokkos Views of dimension D * 6.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component.
     * @note Address : aux(i, j, k, em::***).
     */
    ndfield_t<D, 6> aux;
    /**
     * EM fields at previous time step stored as Kokkos Views of dimension D * 6.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component.
     * @note Address : em0(i, j, k, em::***).
     */
    ndfield_t<D, 6> em0;
    /**
     * Vector potential
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component.
     * @note Address : aphi(i, j, k, 0).
     */
    ndfield_t<D, 1>        aphi;
    ndfield_mirror_t<D, 1> aphi_h;
#endif

    /**
     * @brief Constructor for the fields container. Also sets the active cell sizes and ranges.
     * @param res resolution vector of size D (dimension).
     */
    Fields(std::vector<unsigned int> res);
    ~Fields() = default;

    /**
     * @brief Synchronize data from device to host.
     */
    void SynchronizeHostDevice();
  };

} // namespace ntt

#endif
