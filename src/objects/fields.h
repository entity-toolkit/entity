/* *
   *
   *  Basic 1d/2d/3d field functionality
   *
   *  @namespace ntt::fields
   *
   *  @comment: the main difference between the "fields" and "arrays" is
   *            that the fields also know about the "ghost" zones,
   *            and can traverse both real and ghost zones separately
   *
 * */

#ifndef OBJECTS_FIELDS_H
#define OBJECTS_FIELDS_H

#include "global.h"
#include "arrays.h"

#include <cstddef>

namespace ntt::fields {

  /* *
     *
     *  `OneDField`, `TwoDField`, and `ThreeDField` are each inherited from
     *  the corresponding 1d/2d/3d array classes.
     *
     *  @methods:
     *     allocate()           : allocate array of a given extent (arguments do NOT include the ghost zones)
     *     get_extent()         : get the number of actual (non-ghost) elements in each direction
     *     setAt()              : set the value at a particular index (see below)
     *     getAt()              : set the value at a particular index (see below)
     *
     *  @comment: `TwoDField.allocate(32, 32)` will allocate an array of effective size `(32 + 2 * NGHOSTS) x (32 + 2 * NGHOSTS)`
     *
     *  @comment: `setAt()` and `getAt()` are different from `set()` and `get()` in that their indexing starts from `-NGHOSTS`
     *            to `SIZE + NGHOSTS - 1`; i.e., for the field allocated in the comment above `TwoDField.getAt(0, 0)` will get
     *            the value in the first actual (non-ghost) cell (which is effectively at position (`NGHOSTS`, `NGHOSTS`) array-wise)
     *
     *  @example:
     *            ```c++
     *              // assume NGHOSTS == 2
     *              TwoDField<float> my_field(32, 32);
     *              // the array underlying array `m_data` has a size of `36 x 36`
     *
     *              my_field.getAt(0, 0);
     *              // ^-- this points at the same element as this -->
     *              my_field.get(NGHOSTS, NGHOSTS);
     *
     *              my_field.get_size(1); // this returns "36"
     *              my_field.get_extent(1); // this returns "32"
     *              my_field.getSizeInBytes(); // this returns "5184" (36 x 36 x 4)
     *            ```
     *
   * */
template <class T> class OneDField : public ntt::arrays::OneDArray<T> {
public:
  OneDField() {};
  OneDField(std::size_t n1_);

  void allocate(std::size_t n1_);
  void setAt(int i1, T value);
  auto getAt(int i1) -> T;

  auto get_extent(short int n) -> int;
};

template <class T> class TwoDField : public ntt::arrays::TwoDArray<T> {
public:
  TwoDField() {};
  TwoDField(std::size_t n1_, std::size_t n2_);

  void allocate(std::size_t n1_, std::size_t n2_);
  void setAt(int i1, int i2, T value);
  auto getAt(int i1, int i2) -> T;

  auto get_extent(short int n) -> int;
};

template <class T> class ThreeDField : public ntt::arrays::ThreeDArray<T> {
public:
  ThreeDField() {};
  ThreeDField(std::size_t n1_, std::size_t n2_, std::size_t n3_);

  void allocate(std::size_t n1_, std::size_t n2_, std::size_t n3_);
  void setAt(int i1, int i2, int i3, T value);
  auto getAt(int i1, int i2, int i3) -> T;

  auto get_extent(short int n) -> int;
};
} // namespace ntt::fields

#endif
