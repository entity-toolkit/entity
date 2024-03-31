/**
 * @file arch/directions.h
 * @brief Defines the functionality to work with logically cartesian directions
 * @implements
 *   - dir::Directions
 *   - dir::direction_t
 * @depends:
 *   - global.h
 *   - utils/error.h
 *   - utils/log.h
 * @namespaces:
 *   - dir::Directions
 * @note 
 * dir::Directions<D>::all contains all possible directions in D
 * dimensions with increments in each direction
 * @note 
 * dir::Directions<D>::unique is similar to ::all,
 * but contains only unique directions (i.e. no +1/-1)
 */

#ifndef GLOBAL_ARCH_DIRECTIONS_H
#define GLOBAL_ARCH_DIRECTIONS_H

#include "global.h"

#include "utils/error.h"
#include "utils/log.h"

#include <ostream>
#include <vector>

#include <initializer_list>

namespace dir {

  template <Dimension D>
  struct Directions {};

  template <Dimension D>
  struct direction_t : public std::vector<short> {
    direction_t() : std::vector<short>(D, 0) {}

    direction_t(std::initializer_list<short> list) : std::vector<short>(list) {
      raise::ErrorIf(list.size() != D,
                     "Wrong number of elements in direction_t initializer list",
                     HERE);
    }

    auto operator-() const -> direction_t<D> {
      auto result = direction_t<D> {};
      for (std::size_t i = 0; i < (short)D; ++i) {
        result[i] = -(*this)[i];
      }
      return result;
    }

    auto operator==(const direction_t<D>& other) const -> bool {
      for (std::size_t i = 0; i < (short)D; ++i) {
        if ((*this)[i] != other[i]) {
          return false;
        }
      }
      return true;
    }
  };

  template <Dimension D>
  inline auto operator<<(std::ostream& os, const direction_t<D>& dir)
    -> std::ostream& {
    for (auto& d : dir) {
      os << d << " ";
    }
    return os;
  }

  template <>
  struct Directions<Dim::_1D> {
    inline static const std::vector<direction_t<Dim::_1D>> all = { { -1 }, { 1 } };
    inline static const std::vector<direction_t<Dim::_1D>> unique = { { 1 } };
  };

  template <>
  struct Directions<Dim::_2D> {
    inline static const std::vector<direction_t<Dim::_2D>> all = {
      {-1, -1},
      {-1,  0},
      {-1,  1},
      { 0, -1},
      { 0,  1},
      { 1, -1},
      { 1,  0},
      { 1,  1}
    };
    inline static const std::vector<direction_t<Dim::_2D>> unique = {
      { 0, 1},
      { 1, 1},
      { 1, 0},
      {-1, 1}
    };
  };

  template <>
  struct Directions<Dim::_3D> {
    inline static const std::vector<direction_t<Dim::_3D>> all = {
      {-1, -1, -1},
      {-1, -1,  0},
      {-1, -1,  1},
      {-1,  0, -1},
      {-1,  0,  0},
      {-1,  0,  1},
      {-1,  1, -1},
      {-1,  1,  0},
      {-1,  1,  1},
      { 0, -1, -1},
      { 0, -1,  0},
      { 0, -1,  1},
      { 0,  0, -1},
      { 0,  0,  1},
      { 0,  1, -1},
      { 0,  1,  0},
      { 0,  1,  1},
      { 1, -1, -1},
      { 1, -1,  0},
      { 1, -1,  1},
      { 1,  0, -1},
      { 1,  0,  0},
      { 1,  0,  1},
      { 1,  1, -1},
      { 1,  1,  0},
      { 1,  1,  1}
    };
    inline static const std::vector<direction_t<Dim::_3D>> unique = {
      { 0,  0,  1},
      { 0,  1,  0},
      { 1,  0,  0},
      { 1,  1,  0},
      {-1,  1,  0},
      { 0,  1,  1},
      { 0, -1,  1},
      { 1,  0,  1},
      {-1,  0,  1},
      { 1,  1,  1},
      {-1,  1,  1},
      { 1, -1,  1},
      { 1,  1, -1}
    };
  };

} // namespace dir

#endif // GLOBAL_ARCH_DIRECTIONS_H