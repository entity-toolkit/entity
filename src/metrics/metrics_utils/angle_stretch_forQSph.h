/**
 * @brief Angle stretching for stretched spherical.
 * @implements dtheta_deta
 * @implements eta2theta
 * @implements theta2eta
 */

/**
 * @brief Compute d(th) / d(eta) for a given eta.
 *
 */
Inline auto dtheta_deta(const real_t& eta) const -> real_t {
  if (cmp::AlmostZero(h)) {
    return ONE;
  } else {
    return (ONE + TWO * h +
            static_cast<real_t>(12.0) * h * (eta * constant::INV_PI) *
              ((eta * constant::INV_PI) - ONE));
  }
}

/**
 * @brief Convert quasi-spherical eta to spherical theta.
 *
 */
Inline auto eta2theta(const real_t& eta) const -> real_t {
  if (cmp::AlmostZero(h)) {
    return eta;
  } else {
    return eta + TWO * h * eta * (constant::PI - TWO * eta) *
                   (constant::PI - eta) * constant::INV_PI_SQR;
  }
}

/**
 * @brief Convert spherical theta to quasi-spherical eta.
 *
 */
Inline auto theta2eta(const real_t& theta) const -> real_t {
  if (cmp::AlmostZero(h)) {
    return theta;
  } else {
    using namespace constant;
    // R = (-9 h^2 (Pi - 2 y) + Sqrt[3] Sqrt[-(h^3 ((-4 + h) (Pi + 2 h Pi)^2 +
    // 108 h Pi y - 108 h y^2))])^(1/3)
    double           R { math::pow(
      -9.0 * SQR(h) * (PI - 2.0 * theta) +
        SQRT3 * math::sqrt(
                  (CUBE(h) * ((4.0 - h) * SQR(PI + h * TWO_PI) -
                              108.0 * h * PI * theta + 108.0 * h * SQR(theta)))),
      1.0 / 3.0) };
    // eta = Pi^(2/3)(6 Pi^(1/3) + 2 2^(1/3)(h-1)(3Pi)^(2/3)/R + 2^(2/3) 3^(1/3) R / h)/12
    constexpr double PI_TO_TWO_THIRD { 2.14502939711102560008 };
    constexpr double PI_TO_ONE_THIRD { 1.46459188756152326302 };
    constexpr double TWO_TO_TWO_THIRD { 1.58740105196819947475 };
    constexpr double THREE_TO_ONE_THIRD { 1.442249570307408382321 };
    constexpr double TWO_TO_ONE_THIRD { 1.2599210498948731647672 };
    constexpr double THREE_PI_TO_TWO_THIRD { 4.46184094890142313715794 };
    return static_cast<real_t>(
      PI_TO_TWO_THIRD *
      (6.0 * PI_TO_ONE_THIRD +
       2.0 * TWO_TO_ONE_THIRD * (h - ONE) * THREE_PI_TO_TWO_THIRD / R +
       TWO_TO_TWO_THIRD * THREE_TO_ONE_THIRD * R / h) /
      12.0);
  }
}