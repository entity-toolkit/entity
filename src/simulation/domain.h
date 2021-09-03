#ifndef SIMULATION_DOMAIN_H
#define SIMULATION_DOMAIN_H

#include "global.h"

#include <vector>

namespace ntt {

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
  void set_extent(std::vector<real_t> extent);
  void set_resolution(std::vector<int> resolution);
  void set_boundaries(const std::vector<BoundaryCondition> &bc);
  void set_boundary_x1(BoundaryCondition bc);
  void set_boundary_x2(BoundaryCondition bc);
  void set_boundary_x3(BoundaryCondition bc);

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
