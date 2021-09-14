/* *
   *
   *  Basic 1d/2d/3d array functionality
   *
   *  @namespace: ntt::arrays
   *
   *  @comment: for memory intensive arrays prefer this type to `vector`-s
   *
 * */

#ifndef OBJECTS_ARRAYS_H
#define OBJECTS_ARRAYS_H

#include <cstddef>

namespace ntt::arrays {

/* *
   *
   *  `Array` is a template abstract class to work with multidimensional arrays
   *  of any specific type (`int`, `float`, `double`, `bool`) in a `Fortran`-like
   *  fashion.
   *
   *  @parameters:
   *     m_data               : actual data of type T stored as a pointer
   *     m_allocated          : flag whether the data is already allocated or not
   *
   *  @methods:
   *     getSizeInBytes()     : get size of the data in B [virtual]
   *     get_size()           : get the number of elements in the array [virtual]
   *
 * */
template <class T> class Array {
protected:
  T *m_data;
  bool m_allocated{false};

public:
  Array() {};
  ~Array();
  virtual auto getSizeInBytes() -> std::size_t = 0;
  virtual auto get_size(short int n) -> std::size_t = 0;
};

/* *
   *
   *  `OneDArray`, `TwoDArray`, and `ThreeDArray` are each inherited classes
   *  from the base `Array` that specify functionality (such as allocators, getters/setters etc)
   *  for 1d/2d/3d arrays correspondingly.
   *
   *  @parameters:
   *     n1 [n2, n3]          : number of elements of the array in each dimension
   *
   *  @methods:
   *     allocate()           : allocate array of given size
   *     fillWith()           : fill the whole array with particular values
   *     get()                : get the value at a particular position
   *     set()                : set the value at a particular position
   *
   *  @example:
   *            ```c++
   *              ThreeDArray<double> my_array(32, 32, 32);
   *              // the array underlying array `m_data` has a size of `32 x 32 x 32`
   *              my_array.fillWith(2.0); // filling data with "2.0"
   *              my_array.get(10, 10, 10); // returns "2.0"
   *              my_array.get_size(1); // returns "32"
   *              my_array.getSizeInBytes(1); // returns "262144" (32 x 32 x 32 x 8)
   *            ```
   *
 * */
template <class T> class OneDArray : public Array<T> {
protected:
  std::size_t n1;

public:
  OneDArray() {};
  OneDArray(std::size_t n1_);

  void allocate(std::size_t n1_);
  void fillWith(T value);
  void set(std::size_t i1, T value);
  auto get(std::size_t i1) -> T;

  auto get_size(short int n) -> std::size_t override;
  auto getSizeInBytes() -> std::size_t override;
};

template <class T> class TwoDArray : public Array<T> {
protected:
  std::size_t n1;
  std::size_t n2;

public:
  TwoDArray() {};
  TwoDArray(std::size_t n1_, std::size_t n2_);

  void allocate(std::size_t n1_, std::size_t n2_);
  void fillWith(T value);
  void set(std::size_t i1, std::size_t i2, T value);
  auto get(std::size_t i1, std::size_t i2) -> T;

  auto get_size(short int n) -> std::size_t override;
  auto getSizeInBytes() -> std::size_t override;
};

template <class T> class ThreeDArray : public Array<T> {
protected:
  std::size_t n1;
  std::size_t n2;
  std::size_t n3;

public:
  ThreeDArray() {};
  ThreeDArray(std::size_t n1_, std::size_t n2_, std::size_t n3_);

  void allocate(std::size_t n1_, std::size_t n2_, std::size_t n3_);
  void fillWith(T value);
  void set(std::size_t i1, std::size_t i2, std::size_t i3, T value);
  auto get(std::size_t i1, std::size_t i2, std::size_t i3) -> T;

  auto get_size(short int n) -> std::size_t override;
  auto getSizeInBytes() -> std::size_t override;
};
} // namespace ntt::arrays

#endif
