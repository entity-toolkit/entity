#ifndef FRAMEWORK_FIELDS_H
#define FRAMEWORK_FIELDS_H

#include "wrapper.h"

#include <vector>

#define AssertContent(a, b)                                                                   \
  {                                                                                           \
    NTTLog();                                                                                 \
    ntt::AssertContent_((a), (b));                                                            \
  }

#define AssertEmptyContent(a)                                                                 \
  {                                                                                           \
    NTTLog();                                                                                 \
    ntt::AssertEmptyContent_((a));                                                            \
  }

namespace ntt {
  using resolution_t = std::vector<unsigned int>;
  enum em {
    ex1 = 0,
    ex2 = 1,
    ex3 = 2,
    dx1 = 0,
    dx2 = 1,
    dx3 = 2,
    bx1 = 3,
    bx2 = 4,
    bx3 = 5,
    hx1 = 3,
    hx2 = 4,
    hx3 = 5
  };
  enum cur { jx1 = 0, jx2 = 1, jx3 = 2 };

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
    jx3_curly
  };

  /**
   * @brief Assert that the content of a vector of Content is empty.
   * Should be accessed through the macro AssertEmptyContent, ...
   * ... which also prints the line number/file name.
   */
  void AssertEmptyContent_(const std::vector<Content>&);

  /**
   * @brief Assert particular content of a vector of Content.
   * Should be accessed through the macro AssertContent, ...
   * ... which also prints the line number/file name.
   */
  void AssertContent_(const std::vector<Content>&, const std::vector<Content>&);

  /**
   * @brief Impose particular Content.
   */
  void ImposeContent(std::vector<Content>&, const std::vector<Content>&);
  /**
   * @brief Impose particular Content.
   * @overload
   */
  void ImposeContent(Content&, const Content&);

  /**
   * @brief Impose empty Content.
   */
  void ImposeEmptyContent(std::vector<Content>&);

  /**
   * @brief Impose empty Content.
   * @overload
   */
  void ImposeEmptyContent(Content&);

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
    ndfield_t<D, 6>      em;
    std::vector<Content> em_content = std::vector<Content>(6, Content::empty);
    /**
     * Backup fields used for intermediate operations.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component.
     * @note Address : bckp(i, j, k, ***).
     */
    ndfield_t<D, 6>      bckp;
    std::vector<Content> bckp_content = std::vector<Content>(6, Content::empty);
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
    ndfield_t<D, 3>      cur;
    std::vector<Content> cur_content = std::vector<Content>(3, Content::empty);
    /**
     * Buffers fields used primarily to store currents at previous time step.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x3 for each field
     * component.
     * @note Address : buff(i, j, k, ***).
     */
    ndfield_t<D, 3>      buff;
    std::vector<Content> buff_content = std::vector<Content>(3, Content::empty);

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
    ndfield_t<D, 6>      aux;
    /**
     * EM fields at previous time step stored as Kokkos Views of dimension D * 6.
     *
     * @note Sizes are : resolution + 2 * N_GHOSTS in each direction x6 for each field
     * component.
     * @note Address : em0(i, j, k, em::***).
     */
    ndfield_t<D, 6>      em0;

    /**
     * @brief Constructor for the fields container. Also sets the active cell sizes and ranges.
     * @param res resolution vector of size D (dimension).
     */
    Fields(resolution_t res);
    ~Fields() = default;
  };

}    // namespace ntt

#endif
