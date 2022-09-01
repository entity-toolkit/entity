#ifndef UTILS_QMATH_H
#define UTILS_QMATH_H

namespace ntt {

  /**
   * @brief Function to compare two float values.
   * @param a First value.
   * @param b Second value.
   * @param epsilon Accuracy.
   * @returns true/false.
   */
  bool AlmostEqual(float a, float b, float epsilon = 0.00001f);

  /**
   * @brief Function to compare two double values.
   * @param a First value.
   * @param b Second value.
   * @param epsilon Accuracy.
   * @returns true/false.
   */
  bool AlmostEqual(double a, double b, double epsilon = 1e-8);
} // namespace ntt

#endif
