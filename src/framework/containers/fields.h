/**
 * @file framework/containers/fields.h
 * @brief Definition of the Fields container
 * @implements
 *   - ntt::Fields
 * @cpp:
 *   - fields.cpp
 * @namespaces:
 *   - ntt::
 * @note SRPIC engine allocates em(6), bckp(6), cur(3), buff(3)
 * @note GRPIC engine allocates em(6), bckp(6), cur(3), buff(3), aux(6), em0(6), cur0(3)
 * @note Each field has resolution + 2 * N_GHOSTS components in each direction
 * @note Vector field components are stored as the last index in corresponding field
 */

#ifndef FRAMEWORK_CONTAINERS_FIELDS_H
#define FRAMEWORK_CONTAINERS_FIELDS_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include <vector>

namespace ntt {

  /**
   * @brief Container for the fields. Used a parent class for the Meshblock
   * @tparam D Dimension
   * @tparam S Simulation engine
   */
  template <Dimension D, SimEngine::type S>
  struct Fields {
    /* SRPIC & GRPIC -------------------------------------------------------- */
    /**
     * @brief EM fields at current time step stored as Kokkos Views of dimension D * 6
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component
     * @note Address : em(i, j, k, em::***)
     *
     * @note Bx1 is stored at (    i, j+1/2, k+1/2, n-1/2)
     * @note Bx2 is stored at (i+1/2,     j, k+1/2, n-1/2)
     * @note Bx3 is stored at (i+1/2, j+1/2,     k, n-1/2)
     *
     * @note Ex1 is stored at (i+1/2,     j,     k,     n)
     * @note Ex2 is stored at (    i, j+1/2,     k,     n)
     * @note Ex3 is stored at (    i,     j, k+1/2,     n)
     */
    ndfield_t<D, 6> em;
    /**
     * Backup fields used for intermediate operations
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each
     * field component
     * @note Address : bckp(i, j, k, ***)
     */
    ndfield_t<D, 6> bckp;
    /**
     * Current fields at current time step stored as Kokkos Views of dimension D * 3
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x3 for each field
     * component
     * @note Address : cur(i, j, k, cur::***)
     *
     * @note Jx1 is deposited at (i+1/2,     j,     k, n+1/2)
     * @note Jx2 is deposited at (    i, j+1/2,     k, n+1/2)
     * @note Jx3 is deposited at (    i,     j, k+1/2, n+1/2)
     */
    ndfield_t<D, 3> cur;
    /**
     * Buffers for fields/currents/moments
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x3 for each
     * field component
     * @note Address : buff(i, j, k, ***)
     */
    ndfield_t<D, 3> buff;

    /* GRPIC specific ------------------------------------------------------- */
    /**
     * Auxiliary E and H fields stored as Kokkos Views of dimension D * 6
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each
     * field component
     * @note Address : aux(i, j, k, em::***)
     */
    ndfield_t<D, 6> aux;
    /**
     * EM fields at previous time step stored as Kokkos Views of dimension D * 6
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each
     * field component
     * @note Address : em0(i, j, k, em::***)
     */
    ndfield_t<D, 6> em0;
    /**
     * Current fields for previous timestep (GR)
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x3
     * for each field  component
     * @note Address : cur0(i, j, k, ***)
     */
    ndfield_t<D, 3> cur0;

    /**
     * @brief Constructor for the fields container. Also sets the active cell sizes and ranges
     * @param res resolution vector of size D (dimension)
     */
    Fields() {}

    Fields(const std::vector<std::size_t>& res);

    Fields(Fields&& other) noexcept
      : em { std::move(other.em) }
      , bckp { std::move(other.bckp) }
      , cur { std::move(other.cur) }
      , buff { std::move(other.buff) }
      , aux { std::move(other.aux) }
      , em0 { std::move(other.em0) }
      , cur0 { std::move(other.cur0) } {}

    Fields& operator=(Fields&& other) noexcept {
      if (this != &other) {
        em   = std::move(other.em);
        bckp = std::move(other.bckp);
        cur  = std::move(other.cur);
        buff = std::move(other.buff);
        aux  = std::move(other.aux);
        em0  = std::move(other.em0);
        cur0 = std::move(other.cur0);
      }
      return *this;
    }

    Fields(const Fields&)            = delete;
    Fields& operator=(const Fields&) = delete;

    ~Fields() = default;

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto memory_footprint() const -> std::size_t {
      std::size_t em_footprint   = 6;
      std::size_t bckp_footprint = 6;
      std::size_t cur_footprint  = 3;
      std::size_t buff_footprint = 3;
      std::size_t aux_footprint  = 6;
      std::size_t em0_footprint  = 6;
      std::size_t cur0_footprint = 3;
      for (auto d = 0; d < D; ++d) {
        em_footprint   *= em.extent(d);
        bckp_footprint *= bckp.extent(d);
        cur_footprint  *= cur.extent(d);
        buff_footprint *= buff.extent(d);
        aux_footprint  *= aux.extent(d);
        em0_footprint  *= em0.extent(d);
        cur0_footprint *= cur0.extent(d);
      }
      return (std::size_t)(sizeof(real_t)) *
             (em_footprint + bckp_footprint + cur_footprint + buff_footprint +
              aux_footprint + em0_footprint + cur0_footprint);
    }
  };

} // namespace ntt

#endif // FRAMEWORK_CONTAINERS_FIELDS_H
