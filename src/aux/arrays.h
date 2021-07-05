#ifndef AUX_ARRAYS_H
#define AUX_ARRAYS_H

#include <cstddef>

namespace ntt::arrays {
template <class T> class Array {
protected:
  T *m_data;
  bool m_allocated{false};

public:
  Array() = default;
  ~Array();
  virtual auto getSizeInBytes() -> std::size_t = 0;
  virtual auto get_size(short int n) -> std::size_t = 0;
};

template <class T> class OneDArray : public Array<T> {
private:
  std::size_t n1, n1_full;

public:
  OneDArray() = default;
  OneDArray(std::size_t n1_);

  void allocate(std::size_t n1_);
  void fillWith(T value, bool fill_ghosts = false);
  void set(std::size_t i1, T value);
  auto get(std::size_t i1) -> T;

  auto get_size(short int n) -> std::size_t override;
  auto getSizeInBytes() -> std::size_t override;
};

template <class T> class TwoDArray : public Array<T> {
private:
  std::size_t n1, n1_full;
  std::size_t n2, n2_full;

public:
  TwoDArray() = default;
  TwoDArray(std::size_t n1_, std::size_t n2_);

  void allocate(std::size_t n1_, std::size_t n2_);
  void fillWith(T value, bool fill_ghosts = false);
  void set(std::size_t i1, std::size_t i2, T value);
  auto get(std::size_t i1, std::size_t i2) -> T;

  auto get_size(short int n) -> std::size_t override;
  auto getSizeInBytes() -> std::size_t override;
};

template <class T> class ThreeDArray : public Array<T> {
private:
  std::size_t n1, n1_full;
  std::size_t n2, n2_full;
  std::size_t n3, n3_full;

public:
  ThreeDArray() = default;
  ThreeDArray(std::size_t n1_, std::size_t n2_, std::size_t n3_);

  void allocate(std::size_t n1_, std::size_t n2_, std::size_t n3_);
  void fillWith(T value, bool fill_ghosts = false);
  void set(std::size_t i1, std::size_t i2, std::size_t i3, T value);
  auto get(std::size_t i1, std::size_t i2, std::size_t i3) -> T;

  auto get_size(short int n) -> std::size_t override;
  auto getSizeInBytes() -> std::size_t override;
};
} // namespace ntt::arrays

#endif
