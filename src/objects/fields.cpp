#include "global.h"
#include "fields.h"
#include "arrays.h"

#include <cassert>
#include <stdexcept>

namespace ntt::fields {

// constructor
template <class T> OneDField<T>::OneDField(std::size_t n1_) { this->allocate(n1_); }
template <class T> TwoDField<T>::TwoDField(std::size_t n1_, std::size_t n2_) { this->allocate(n1_, n2_); }
template <class T> ThreeDField<T>::ThreeDField(std::size_t n1_, std::size_t n2_, std::size_t n3_) {
  this->allocate(n1_, n2_, n3_);
}

// allocators
template <class T> void OneDField<T>::allocate(std::size_t n1_) {
  ntt::arrays::OneDArray<T>::allocate(n1_ + 2 * N_GHOSTS);
}
template <class T> void TwoDField<T>::allocate(std::size_t n1_, std::size_t n2_) {
  ntt::arrays::TwoDArray<T>::allocate(n1_ + 2 * N_GHOSTS, n2_ + 2 * N_GHOSTS);
}
template <class T> void ThreeDField<T>::allocate(std::size_t n1_, std::size_t n2_, std::size_t n3_) {
  ntt::arrays::ThreeDArray<T>::allocate(n1_ + 2 * N_GHOSTS, n2_ + 2 * N_GHOSTS, n3_ + 2 * N_GHOSTS);
}

// element setters
template <class T> void OneDField<T>::setAt(int i1, T value) {
# ifdef DEBUG
  if ((i1 < -static_cast<int>(N_GHOSTS)) || (i1 > static_cast<int>(this->n1 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i1 out of range for 1d field in `setAt`.");
  }
# endif
  std::size_t index1 = static_cast<std::size_t>(i1 + static_cast<int>(N_GHOSTS));
  ntt::arrays::OneDArray<T>::set(index1, value);
}
template <class T> void TwoDField<T>::setAt(int i1, int i2, T value) {
# ifdef DEBUG
  if ((i1 < -static_cast<int>(N_GHOSTS)) || (i1 > static_cast<int>(this->n1 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i1 out of range for 2d field in `setAt`.");
  }
  if ((i2 < -static_cast<int>(N_GHOSTS)) || (i2 > static_cast<int>(this->n2 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i2 out of range for 2d field in `setAt`.");
  }
# endif
  std::size_t index1 = static_cast<std::size_t>(i1 + static_cast<int>(N_GHOSTS));
  std::size_t index2 = static_cast<std::size_t>(i2 + static_cast<int>(N_GHOSTS));
  ntt::arrays::TwoDArray<T>::set(index1, index2, value);
}
template <class T> void ThreeDField<T>::setAt(int i1, int i2, int i3, T value) {
# ifdef DEBUG
  if ((i1 < -static_cast<int>(N_GHOSTS)) || (i1 > static_cast<int>(this->n1 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i1 out of range for 3d field in `setAt`.");
  }
  if ((i2 < -static_cast<int>(N_GHOSTS)) || (i2 > static_cast<int>(this->n2 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i2 out of range for 3d field in `setAt`.");
  }
  if ((i3 < -static_cast<int>(N_GHOSTS)) || (i3 > static_cast<int>(this->n3 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i3 out of range for 3d field in `setAt`.");
  }
# endif
  std::size_t index1 = static_cast<std::size_t>(i1 + static_cast<int>(N_GHOSTS));
  std::size_t index2 = static_cast<std::size_t>(i2 + static_cast<int>(N_GHOSTS));
  std::size_t index3 = static_cast<std::size_t>(i3 + static_cast<int>(N_GHOSTS));
  ntt::arrays::ThreeDArray<T>::set(index1, index2, index3, value);
}

// element getters
template <class T> auto OneDField<T>::getAt(int i1) -> T {
# ifdef DEBUG
  if ((i1 < -static_cast<int>(N_GHOSTS)) || (i1 > static_cast<int>(this->n1 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i1 out of range for 1d field in `getAt`.");
  }
# endif
  std::size_t index1 = static_cast<std::size_t>(i1 + static_cast<int>(N_GHOSTS));
  return ntt::arrays::OneDArray<T>::get(index1);
}
template <class T> auto TwoDField<T>::getAt(int i1, int i2) -> T {
# ifdef DEBUG
  if ((i1 < -static_cast<int>(N_GHOSTS)) || (i1 > static_cast<int>(this->n1 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i1 out of range for 2d field in `getAt`.");
  }
  if ((i2 < -static_cast<int>(N_GHOSTS)) || (i2 > static_cast<int>(this->n2 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i2 out of range for 2d field in `getAt`.");
  }
# endif
  std::size_t index1 = static_cast<std::size_t>(i1 + static_cast<int>(N_GHOSTS));
  std::size_t index2 = static_cast<std::size_t>(i2 + static_cast<int>(N_GHOSTS));
  return ntt::arrays::TwoDArray<T>::get(index1, index2);
}
template <class T> auto ThreeDField<T>::getAt(int i1, int i2, int i3) -> T {
# ifdef DEBUG
  if ((i1 < -static_cast<int>(N_GHOSTS)) || (i1 > static_cast<int>(this->n1 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i1 out of range for 3d field in `getAt`.");
  }
  if ((i2 < -static_cast<int>(N_GHOSTS)) || (i2 > static_cast<int>(this->n2 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i2 out of range for 3d field in `getAt`.");
  }
  if ((i3 < -static_cast<int>(N_GHOSTS)) || (i3 > static_cast<int>(this->n3 - N_GHOSTS) - 1)) {
    throw std::out_of_range("# Error: i3 out of range for 3d field in `getAt`.");
  }
# endif
  std::size_t index1 = static_cast<std::size_t>(i1 + static_cast<int>(N_GHOSTS));
  std::size_t index2 = static_cast<std::size_t>(i2 + static_cast<int>(N_GHOSTS));
  std::size_t index3 = static_cast<std::size_t>(i3 + static_cast<int>(N_GHOSTS));
  return ntt::arrays::ThreeDArray<T>::get(index1, index2, index3);
}

// size getters
template <class T> auto OneDField<T>::get_extent(short int n) -> int {
  UNUSED(n);
  assert(n == 1);
  return static_cast<int>(this->n1 - 2 * N_GHOSTS);
}
template <class T> auto TwoDField<T>::get_extent(short int n) -> int {
  assert((n == 1) || (n == 2));
  if (n == 1) {
    return static_cast<int>(this->n1 - 2 * N_GHOSTS);
  } else {
    return static_cast<int>(this->n2 - 2 * N_GHOSTS);
  }
}
template <class T> auto ThreeDField<T>::get_extent(short int n) -> int {
  assert((n == 1) || (n == 2) || (n == 3));
  if (n == 1) {
    return static_cast<int>(this->n1 - 2 * N_GHOSTS);
  } else if (n == 2) {
    return static_cast<int>(this->n2 - 2 * N_GHOSTS);
  } else {
    return static_cast<int>(this->n3 - 2 * N_GHOSTS);
  }
}

} // namespace ntt::fields
