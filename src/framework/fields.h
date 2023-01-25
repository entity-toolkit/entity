#ifndef FRAMEWORK_FIELDS_H
#define FRAMEWORK_FIELDS_H

#include "wrapper.h"

#include <vector>

namespace ntt {
  using resolution_t = std::vector<unsigned int>;
  enum em { ex1 = 0, ex2 = 1, ex3 = 2, bx1 = 3, bx2 = 4, bx3 = 5 };
  enum cur { jx1 = 0, jx2 = 1, jx3 = 2 };
  enum fld { dens = 0 };

  /**
   * @brief To keep track what fields are stored in buffer arrays we use an enum indicator.
   * Nothing really depends on the actual values, they are purely to help ...
   * ... guide the development, and make catching bugs easier.
   */
  enum class Content : int {
    empty = 0,

    ex1_hat,
    ex2_hat,
    ex3_hat,
    ex1_hat_int,
    ex2_hat_int,
    ex3_hat_int,

    bx1_hat,
    bx2_hat,
    bx3_hat,
    bx1_hat_int,
    bx2_hat_int,
    bx3_hat_int,

    ex1_cntrv,
    ex2_cntrv,
    ex3_cntrv,
    bx1_cntrv,
    bx2_cntrv,
    bx3_cntrv,

    ex1_cov,
    ex2_cov,
    ex3_cov,
    bx1_cov,
    bx2_cov,
    bx3_cov,

    jx1_hat,
    jx2_hat,
    jx3_hat,
    jx1_hat_int,
    jx2_hat_int,
    jx3_hat_int,

    jx1_cntrv,
    jx2_cntrv,
    jx3_cntrv,

    jx1_cov,
    jx2_cov,
    jx3_cov,

    jx1_curly,
    jx2_curly,
    jx3_curly,

    mass_density,
    ch_density,
    num_density
  };

  /**
   * @brief Assert that the content of a vector of Content is empty.
   */
  void AssertEmptyContent(const std::vector<Content>&);

  /**
   * @brief Assert particular content of a vector of Content.
   */
  void AssertContent(const std::vector<Content>&, const std::vector<Content>&);

  /**
   * @brief Impose particular Content.
   */
  void ImposeContent(std::vector<Content>&, const std::vector<Content>&);

  /**
   * @brief Impose empty Content.
   */
  void ImposeEmptyContent(std::vector<Content>&);

  /**
   * @brief Container for the fields. Used a parent class for the Meshblock.
   * @tparam D Dimension.
   * @tparam S Simulation engine.
   */
  template <Dimension D, SimulationEngine S>
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
    std::vector<Content>   em_content   = std::vector<Content>(6, Content::empty);
    std::vector<Content>   em_h_content = std::vector<Content>(6, Content::empty);
    /**
     * Backup fields used for intermediate operations.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component.
     * @note Address : bckp(i, j, k, ***).
     */
    ndfield_t<D, 6>        bckp;
    ndfield_mirror_t<D, 6> bckp_h;
    std::vector<Content>   bckp_content   = std::vector<Content>(6, Content::empty);
    std::vector<Content>   bckp_h_content = std::vector<Content>(6, Content::empty);
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
    std::vector<Content>   cur_content   = std::vector<Content>(3, Content::empty);
    std::vector<Content>   cur_h_content = std::vector<Content>(3, Content::empty);
    /**
     * Buffers fields used primarily to store currents at previous time step.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x3 for each field
     * component.
     * @note Address : buff(i, j, k, ***).
     */
    ndfield_t<D, 3>        buff;
    ndfield_mirror_t<D, 3> buff_h;
    std::vector<Content>   buff_content   = std::vector<Content>(3, Content::empty);
    std::vector<Content>   buff_h_content = std::vector<Content>(3, Content::empty);
#ifdef GRPIC_ENGINE
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
    ndfield_t<D, 6>        aux;
    /**
     * EM fields at previous time step stored as Kokkos Views of dimension D * 6.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component.
     * @note Address : em0(i, j, k, em::***).
     */
    ndfield_t<D, 6>        em0;
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
    Fields(resolution_t res);
    ~Fields() = default;

    /**
     * @brief Synchronize data from device to host.
     */
    void SynchronizeHostDevice();
  };

}    // namespace ntt

#endif
