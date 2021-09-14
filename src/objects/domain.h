/* *
   *
   *  Container for the information about the domain
   *
   *  @namespace: ntt
   *
   *  @comment: simulation object stores an object of this domain class that carries
   *            all the domain information (extent, coordinate system, boundary conditions etc)
   *
 * */

#ifndef OBJECTS_DOMAIN_H
#define OBJECTS_DOMAIN_H

#include "global.h"

#include <vector>

namespace ntt {

/* *
   *
   *  `Domain` is a class that stores all the domain information and handles
   *  coordinate/index transformations internally.
   *
   *  @parameters:
   *     m_dimension                : dimensionality of the domain (1d, 2d, 3d)
   *     m_coord_system             : coordinate system (see `global.h` for options)
   *     m_boundary                 : vector of boundary conditions (for each direction; see `global.h` for options)
   *     m_extent                   : vector of extents in physical units in each dimension (x1min, x1max, x2min, x2max, ...)
   *     m_resolution               : vector of the number of cells in each dimension
   *
   *  @methods:
   *     set_extent()               : set the extent (accepts a vector)
   *     set_resolution()           : set the resolution (accepts a vector)
   *     set_boundaries()           : set the boundary conditions (accepts a vector)
   *     set_boundary_x[1/2/3]()    : set the boundary condition in direction #1, #2, #3
   *     x[1/2/3][min/max]()        : get the corner position in physical units in each of the dimensions
   *     dxi()                      : get the vector of the cell size (in physical units) in each dimension
   *     dx[1/2/3]()                : get the cell size (in physical units) in particular dimension
   *     sizexi()                   : get the vector of the sizes (in physical units) in each dimension
   *     sizex[1/2/3]()             : get the size (in physical units) in particular dimension
   *     nx[1/2/3]()                : get the resolution in particular dimension
   *     x1x2x3_to_ijk()            : convert physical coordinates to cell index
   *     ijk_to_x1x2x3()            : convert cell index to physical coordinates (cell corner)
   *
   *  @example:
   *            ```c++
   *              Domain my_domain(TWO_D, POLAR_COORD);
   *              my_domain.set_extent({0.1, 2.0, 0.0, 2.0 * PI});
   *              my_domain.set_resolution({100, 25});
   *              my_domain.set_boundaries({OPEN_BC, PERIODIC_BC});
   *
   *              my_domain.dxi(); // this returns a vector of `{0.019, 0.2513}`
   *              my_domain.sizexi(); // this returns a vector of `{1.9, 2.0 * PI}`
   *
   *              my_domain.x1x2x3_to_ijk({1.5, 1.2}); // this returns a vector of `{73, 4}`
   *              auto r_phi = my_domain.ijk_to_x1x2x3({66, 15}); // this returns a vector of `{1.354, 3.76991}`
   *              my_domain.x1x2x3_to_ijk(r_phi); // this returns ({66, 15});
   *            ```
   *
 * */
class Domain {
protected:
  const Dimension m_dimension;
  const CoordinateSystem m_coord_system;
  std::vector<BoundaryCondition> m_boundary;

  std::vector<real_t> m_extent;
  std::vector<int> m_resolution;

public:
  Domain(Dimension dim, CoordinateSystem coord) : m_dimension(dim), m_coord_system(coord) {}
  ~Domain() = default;
  void set_extent(const std::vector<real_t> &extent);
  void set_resolution(const std::vector<int> &resolution);
  void set_boundaries(const std::vector<BoundaryCondition> &bc);
  void set_boundary_x1(const BoundaryCondition &bc);
  void set_boundary_x2(const BoundaryCondition &bc);
  void set_boundary_x3(const BoundaryCondition &bc);

  // a bunch of getters to simplify the workflow
  //    extent of each dimension
  [[nodiscard]] auto x1min() const -> real_t;
  [[nodiscard]] auto x1max() const -> real_t;
  [[nodiscard]] auto x2min() const -> real_t;
  [[nodiscard]] auto x2max() const -> real_t;
  [[nodiscard]] auto x3min() const -> real_t;
  [[nodiscard]] auto x3max() const -> real_t;

  //    step in each dimension
  [[nodiscard]] auto dxi() const -> std::vector<real_t>;
  [[nodiscard]] auto dx1() const -> real_t;
  [[nodiscard]] auto dx2() const -> real_t;
  [[nodiscard]] auto dx3() const -> real_t;

  //    size & resolution in each dimension
  [[nodiscard]] auto sizexi() const -> std::vector<real_t>;
  [[nodiscard]] auto sizex1() const -> real_t;
  [[nodiscard]] auto sizex2() const -> real_t;
  [[nodiscard]] auto sizex3() const -> real_t;
  [[nodiscard]] auto nx1() const -> int;
  [[nodiscard]] auto nx2() const -> int;
  [[nodiscard]] auto nx3() const -> int;

  // converters
  [[nodiscard]] auto x1x2x3_to_ijk(std::vector<real_t> x1x2x3) -> std::vector<int>;
  [[nodiscard]] auto ijk_to_x1x2x3(std::vector<int> ijk) -> std::vector<real_t>;

  friend class Simulation;
};

} // namespace ntt

#endif
