#ifndef UTILS_QMATH_H
#define UTILS_QMATH_H

#include "wrapper.h"

#include <cfloat>

namespace ntt {

  /**
   * @brief Function to compare two float values.
   * @param a First value.
   * @param b Second value.
   * @param epsilon Accuracy [optional].
   * @returns true/false.
   */
  Inline bool AlmostEqual(float a, float b, float epsilon = 0.00001f) {
    if (a == b) {
      return true;
    } else {
      float diff { math::abs(a - b) };
      if (diff <= 1e-6) {
        return true;
      } else {
        return diff <= math::min(math::abs(a), math::abs(b)) * epsilon;
      }
    }
  }

  /**
   * @brief Function to compare two double values.
   * @param a First value.
   * @param b Second value.
   * @param epsilon Accuracy [optional].
   * @returns true/false.
   */
  Inline bool AlmostEqual(double a, double b, double epsilon = 1e-8) {
    if (a == b) {
      return true;
    } else {
      double diff { math::abs(a - b) };
      if (diff <= 1e-12) {
        return true;
      } else {
        return diff <= math::min(math::abs(a), math::abs(b)) * epsilon;
      }
    }
  }

  /**
   * @brief Function to compare two vectors or coordinate vectors.
   * @param a First vector.
   * @param b Second vector.
   * @param epsilon Accuracy [optional].
   * @returns true/false.
   */
  template <Dimension D>
  Inline bool AlmostEqual(const vec_t<D>& a,
                          const vec_t<D>& b,
                          real_t          epsilon
                          = std::is_same<real_t, float>::value ? 0.00001f : 1e-8) {
    for (auto i { 0 }; i < static_cast<short>(D); ++i) {
      if (!AlmostEqual(a[i], b[i], epsilon)) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Function to compare a number with zero.
   * @tparam T Type of the number.
   * @param a Number
   * @param epsilon Accuracy [optional].
   * @returns true/false.
   */
  template <typename T>
  Inline bool CloseToZero(T a, T epsilon = std::is_same<T, float>::value ? 0.00001f : 1e-8) {
    return math::abs(a) < epsilon;
  }
}    // namespace ntt

template bool ntt::AlmostEqual<ntt::Dim1>(const ntt::vec_t<ntt::Dim1>&,
                                          const ntt::vec_t<ntt::Dim1>&,
                                          real_t);
template bool ntt::AlmostEqual<ntt::Dim2>(const ntt::vec_t<ntt::Dim2>&,
                                          const ntt::vec_t<ntt::Dim2>&,
                                          real_t);
template bool ntt::AlmostEqual<ntt::Dim3>(const ntt::vec_t<ntt::Dim3>&,
                                          const ntt::vec_t<ntt::Dim3>&,
                                          real_t);

template bool ntt::CloseToZero<float>(float, float);
template bool ntt::CloseToZero<double>(double, double);
#endif
