#include "global.h"
#include "arrays.h"

#include <cassert>
#include <stdexcept>

namespace ntt::arrays {
template <class T> Array<T>::~Array() {
  if (this->m_allocated) {
    delete[] this->m_data;
  }
  this->m_allocated = false;
}

// constructor
template <class T> OneDArray<T>::OneDArray(std::size_t n1_) { this->allocate(n1_); }
template <class T> TwoDArray<T>::TwoDArray(std::size_t n1_, std::size_t n2_) { this->allocate(n1_, n2_); }
template <class T> ThreeDArray<T>::ThreeDArray(std::size_t n1_, std::size_t n2_, std::size_t n3_) {
  this->allocate(n1_, n2_, n3_);
}

// allocator
template <class T> void OneDArray<T>::allocate(std::size_t n1_) {
  if (this->m_allocated) {
    throw std::runtime_error("# Error: 1d array already allocated.");
  }
  this->n1 = n1_;
  this->n1_full = n1_ + 2 * N_GHOSTS;
  this->m_data = new T[this->n1_full];
  this->m_allocated = true;
}
template <class T> void TwoDArray<T>::allocate(std::size_t n1_, std::size_t n2_) {
  if (this->m_allocated) {
    throw std::runtime_error("# Error: 2d array already allocated.");
  }
  this->n1 = n1_;
  this->n1_full = n1_ + 2 * N_GHOSTS;
  this->n2 = n2_;
  this->n2_full = n2_ + 2 * N_GHOSTS;
  this->m_data = new T[this->n1_full * this->n2_full];
  this->m_allocated = true;
}
template <class T> void ThreeDArray<T>::allocate(std::size_t n1_, std::size_t n2_, std::size_t n3_) {
  if (this->m_allocated) {
    throw std::runtime_error("# Error: 3d array already allocated.");
  }
  this->n1 = n1_;
  this->n1_full = n1_ + 2 * N_GHOSTS;
  this->n2 = n2_;
  this->n2_full = n2_ + 2 * N_GHOSTS;
  this->n3 = n3_;
  this->n3_full = n3_ + 2 * N_GHOSTS;
  this->m_data = new T[this->n1_full * this->n2_full * this->n3_full];
  this->m_allocated = true;
}

// filler
template <class T> void OneDArray<T>::fillWith(T value, bool fill_ghosts) {
  if (!(this->m_allocated)) {
    throw std::runtime_error("# Error: 1d array is not allocated.");
  }
  std::size_t i1min{fill_ghosts ? (0) : (N_GHOSTS)};
  std::size_t i1max{fill_ghosts ? (this->n1_full) : (this->n1)};
  for (std::size_t i1{i1min}; i1 < i1max; ++i1) {
    this->m_data[i1] = value;
  }
}
template <class T> void TwoDArray<T>::fillWith(T value, bool fill_ghosts) {
  if (!(this->m_allocated)) {
    throw std::runtime_error("# Error: 2d array is not allocated.");
  }
  std::size_t i1min{fill_ghosts ? (0) : (N_GHOSTS)};
  std::size_t i1max{fill_ghosts ? (this->n1_full) : (this->n1)};
  std::size_t i2min{fill_ghosts ? (0) : (N_GHOSTS)};
  std::size_t i2max{fill_ghosts ? (this->n2_full) : (this->n2)};
  for (std::size_t i2{i2min}; i2 < i2max; ++i2) {
    std::size_t j2{(this->n1_full) * i2};
    for (std::size_t i1{i1min}; i1 < i1max; ++i1) {
      this->m_data[i1 + j2] = value;
    }
  }
}
template <class T> void ThreeDArray<T>::fillWith(T value, bool fill_ghosts) {
  if (!(this->m_allocated)) {
    throw std::runtime_error("# Error: 3d array is not allocated.");
  }
  std::size_t i1min{fill_ghosts ? (0) : (N_GHOSTS)};
  std::size_t i1max{fill_ghosts ? (this->n1_full) : (this->n1)};
  std::size_t i2min{fill_ghosts ? (0) : (N_GHOSTS)};
  std::size_t i2max{fill_ghosts ? (this->n2_full) : (this->n2)};
  std::size_t i3min{fill_ghosts ? (0) : (N_GHOSTS)};
  std::size_t i3max{fill_ghosts ? (this->n3_full) : (this->n3)};
  for (std::size_t i3{i3min}; i3 < i3max; ++i3) {
    std::size_t j3{(this->n2_full) * i3};
    for (std::size_t i2{i2min}; i2 < i2max; ++i2) {
      std::size_t j23{(this->n1_full) * (i2 + j3)};
      for (std::size_t i1{i1min}; i1 < i1max; ++i1) {
        this->m_data[i1 + j23] = value;
      }
    }
  }
}

// element setter
template <class T> void OneDArray<T>::set(std::size_t i1, T value) {
# ifdef DEBUG
  if (!(i1 < this->n1_full)) {
    std::out_of_range("# Error: i1 out of range for 1d array.");
  }
# endif
  this->m_data[i1] = value;
}
template <class T> void TwoDArray<T>::set(std::size_t i1, std::size_t i2, T value) {
# ifdef DEBUG
  if (!(i1 < this->n1_full)) {
    std::out_of_range("# Error: i1 out of range for 2d array.");
  }
  if (!(i2 < this->n2_full)) {
    std::out_of_range("# Error: i2 out of range for 2d array.");
  }
# endif
  this->m_data[i1 + (this->n1_full) * i2] = value;
}
template <class T> void ThreeDArray<T>::set(std::size_t i1, std::size_t i2, std::size_t i3, T value) {
# ifdef DEBUG
  if (!(i1 < this->n1_full)) {
    std::out_of_range("# Error: i1 out of range for 3d array.");
  }
  if (!(i2 < this->n2_full)) {
    std::out_of_range("# Error: i2 out of range for 3d array.");
  }
  if (!(i3 < this->n3_full)) {
    std::out_of_range("# Error: i3 out of range for 3d array.");
  }
# endif
  this->m_data[i1 + (this->n1_full) * (i2 + (this->n2_full) * i3)] = value;
}

// element getter
template <class T> auto OneDArray<T>::get(std::size_t i1) -> T {
# ifdef DEBUG
  if (!(i1 < this->n1_full)) {
    std::out_of_range("# Error: i1 out of range for 1d array.");
  }
# endif
  return this->m_data[i1];
}
template <class T> auto TwoDArray<T>::get(std::size_t i1, std::size_t i2) -> T {
# ifdef DEBUG
  if (!(i1 < this->n1_full)) {
    std::out_of_range("# Error: i1 out of range for 2d array.");
  }
  if (!(i2 < this->n2_full)) {
    std::out_of_range("# Error: i2 out of range for 2d array.");
  }
# endif
  return this->m_data[i1 + (this->n1_full) * i2];
}
template <class T> auto ThreeDArray<T>::get(std::size_t i1, std::size_t i2, std::size_t i3) -> T {
# ifdef DEBUG
  if (!(i1 < this->n1_full)) {
    std::out_of_range("# Error: i1 out of range for 3d array.");
  }
  if (!(i2 < this->n2_full)) {
    std::out_of_range("# Error: i2 out of range for 3d array.");
  }
  if (!(i3 < this->n3_full)) {
    std::out_of_range("# Error: i3 out of range for 3d array.");
  }
# endif
  return this->m_data[i1 + (this->n1_full) * (i2 + (this->n2_full) * i3)];
}

// size getter
template <class T> auto OneDArray<T>::get_size(short int n) -> std::size_t {
  UNUSED(n);
  assert(n == 1);
  return this->n1;
}
template <class T> auto TwoDArray<T>::get_size(short int n) -> std::size_t {
  assert((n == 1) || (n == 2));
  if (n == 1) {
    return this->n1;
  } else {
    return this->n2;
  }
}
template <class T> auto ThreeDArray<T>::get_size(short int n) -> std::size_t {
  assert((n == 1) || (n == 2) || (n == 3));
  if (n == 1) {
    return this->n1;
  } else if (n == 2) {
    return this->n2;
  } else {
    return this->n3;
  }
}

// memory usage
template <class T> auto OneDArray<T>::getSizeInBytes() -> std::size_t {
  if (this->m_allocated) {
    return sizeof(T) * this->n1_full;
  } else {
    return 0;
  }
}
template <class T> auto TwoDArray<T>::getSizeInBytes() -> std::size_t {
  if (this->m_allocated) {
    return sizeof(T) * this->n1_full * this->n2_full;
  } else {
    return 0;
  }
}
template <class T> auto ThreeDArray<T>::getSizeInBytes() -> std::size_t {
  if (this->m_allocated) {
    return sizeof(T) * this->n1_full * this->n2_full * this->n3_full;
  } else {
    return 0;
  }
}
} // namespace ntt::arrays
