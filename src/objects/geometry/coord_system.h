#ifndef OBJECTS_GEOMETRY_COORD_SYSTEM_H
#define OBJECTS_GEOMETRY_COORD_SYSTEM_H

#include "global.h"

#include <string>

namespace ntt {

template <Dimension D>
struct CoordinateSystem {
  CoordinateSystem() = default;
  CoordinateSystem(const std::string& label) : m_label{label} {}
  [[nodiscard]] auto get_label() const -> std::string { return m_label; }
  virtual ~CoordinateSystem() = default;

  virtual Inline auto convert_x1TOx(const real_t&) const -> real_t { return -1.0; }
  virtual Inline auto convert_x1x2TOxy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
  virtual Inline auto convert_x1x2x3TOxyz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

  virtual Inline auto convert_xTOx1(const real_t&) const -> real_t { return -1.0; }
  virtual Inline auto convert_xyTOx1x2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
  virtual Inline auto convert_xyzTOx1x2x3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

  // velocity conversion
  virtual Inline auto convert_ux1TOux(const real_t&) const -> real_t { return -1.0; }
  virtual Inline auto convert_ux1ux2TOuxuy(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; }
  virtual Inline auto convert_ux1ux2ux3TOuxuyuz(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }

  virtual Inline auto convert_uxTOux1(const real_t&) const -> real_t { return -1.0; }
  virtual Inline auto convert_uxuyTOux1ux2(const real_t&, const real_t&) const -> std::tuple<real_t, real_t> { return {-1.0, -1.0}; };
  virtual Inline auto convert_uxuyuzTOux1ux2ux3(const real_t&, const real_t&, const real_t&) const -> std::tuple<real_t, real_t, real_t> { return {-1.0, -1.0, -1.0}; }
protected:
  std::string m_label;
};

}


#endif
