#include "global.h"

#include "cartesian.h"

namespace ntt {

template<Dimension D>
Inline auto CartesianSystem<D>::transform_x1TOx(const real_t& x1) const -> real_t { return x1; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_x1x2TOxy(const real_t& x1, const real_t& x2) const -> std::tuple<real_t, real_t> { return {x1, x2}; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_x1x2x3TOxyz(const real_t& x1, const real_t& x2, const real_t& x3) const -> std::tuple<real_t, real_t, real_t> { return {x1, x2, x3}; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_xTOx1(const real_t& x) const -> real_t {return x; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_xyTOx1x2(const real_t& x, const real_t& y) const -> std::tuple<real_t, real_t> { return {x, y}; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_xyzTOx1x2x3(const real_t& x, const real_t& y, const real_t& z) const -> std::tuple<real_t, real_t, real_t> { return {x, y, z}; }

// velocity conversion
template<Dimension D>
Inline auto CartesianSystem<D>::transform_ux1TOux(const real_t& ux1) const -> real_t { return ux1; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_ux1ux2TOuxuy(const real_t& ux1, const real_t& ux2) const -> std::tuple<real_t, real_t> { return {ux1, ux2}; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_ux1ux2ux3TOuxuyuz(const real_t& ux1, const real_t& ux2, const real_t& ux3) const -> std::tuple<real_t, real_t, real_t> { return {ux1, ux2, ux3}; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_uxTOux1(const real_t& ux) const -> real_t { return ux; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_uxuyTOux1ux2(const real_t& ux, const real_t& uy) const -> std::tuple<real_t, real_t> { return {ux, uy}; }

template<Dimension D>
Inline auto CartesianSystem<D>::transform_uxuyuzTOux1ux2ux3(const real_t& ux, const real_t& uy, const real_t& uz) const -> std::tuple<real_t, real_t, real_t> { return {ux, uy, uz}; }

}

template struct ntt::CartesianSystem<ntt::ONE_D>;
template struct ntt::CartesianSystem<ntt::TWO_D>;
template struct ntt::CartesianSystem<ntt::THREE_D>;

//
// // curvilinear-specific conversions
// #ifndef HARDCODE_FLAT_COORDS
//
// // curvilinear-to-cartesian
// template <>
// Inline auto Grid<ONE_D>::transform_x1TOx(const real_t& x1) const -> real_t {
// #  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  else
//   UNUSED(x1);
//   throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
// #  endif
// }
//
// template <>
// Inline auto Grid<TWO_D>::transform_x1x2TOxy(const real_t& x1, const real_t& x2) const
//     -> std::tuple<real_t, real_t> {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   return {x1 * std::cos(x2), x1 * std::sin(x2)};
// #  else
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// template <>
// Inline auto
// Grid<THREE_D>::transform_x1x2x3TOxyz(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> std::tuple<real_t, real_t, real_t> {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// // cartesian-to-curvilinear
// template <>
// Inline auto Grid<ONE_D>::transform_xTOx1(const real_t& x) const -> real_t {
// #  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  else
//   UNUSED(x);
//   throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
// #  endif
// }
//
// template <>
// Inline auto Grid<TWO_D>::transform_xyTOx1x2(const real_t& x, const real_t& y) const
//     -> std::tuple<real_t, real_t> {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   real_t r {std::sqrt(x * x + y * y)};
//   return {r, std::acos(y / r)};
// #  else
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// template <>
// Inline auto
// Grid<THREE_D>::transform_xyzTOx1x2x3(const real_t& x, const real_t& y, const real_t& z) const
//     -> std::tuple<real_t, real_t, real_t> {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// // Jacobian coefficients
// // h-values
// // 1d
// template <>
// Inline auto Grid<ONE_D>::Jacobian_h1(const real_t& x1) const -> real_t {
// #  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  else
//   UNUSED(x1);
//   throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
// #  endif
// }
//
// template <>
// Inline auto Grid<ONE_D>::Jacobian_h2(const real_t& x1) const -> real_t {
// #  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  else
//   UNUSED(x1);
//   throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
// #  endif
// }
//
// template <>
// Inline auto Grid<ONE_D>::Jacobian_h3(const real_t& x1) const -> real_t {
// #  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  else
//   UNUSED(x1);
//   throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
// #  endif
// }
//
//
// // 2d
// template <>
// Inline auto Grid<TWO_D>::Jacobian_h1(const real_t& x1, const real_t& x2) const -> real_t {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   UNUSED(x1);
//   UNUSED(x2);
//   return ONE;
// #  else
//   UNUSED(x1);
//   UNUSED(x2);
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// template <>
// Inline auto Grid<TWO_D>::Jacobian_h2(const real_t& x1, const real_t& x2) const -> real_t {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   UNUSED(x2);
//   return x1;
// #  else
//   UNUSED(x1);
//   UNUSED(x2);
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// template <>
// Inline auto Grid<TWO_D>::Jacobian_h3(const real_t& x1, const real_t& x2) const
//     -> real_t {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   return x1 * std::sin(x2);
// #  else
//   UNUSED(x1);
//   UNUSED(x2);
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// // 3d
// template <>
// Inline auto Grid<THREE_D>::Jacobian_h1(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_h2(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_h3(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// // Matrix
// // 1d
// template <>
// Inline auto Grid<ONE_D>::Jacobian_11(const real_t& x1) const -> real_t {
// #  ifdef HARDCODE_CARTESIAN_LIKE_COORDS
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  else
//   UNUSED(x1);
//   throw std::runtime_error("# Error: dimensionality and coord system incompatible.");
// #  endif
// }
//
// // 2d
// template <>
// Inline auto Grid<TWO_D>::Jacobian_11(const real_t& x1, const real_t& x2) const -> real_t {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   UNUSED(x1);
//   return std::sin(x2);
// #  else
//   UNUSED(x1);
//   UNUSED(x2);
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// template <>
// Inline auto Grid<TWO_D>::Jacobian_12(const real_t& x1, const real_t& x2) const -> real_t {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   return x1 * std::cos(x2);
// #  else
//   UNUSED(x1);
//   UNUSED(x2);
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// template <>
// Inline auto Grid<TWO_D>::Jacobian_21(const real_t& x1, const real_t& x2) const -> real_t {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   UNUSED(x1);
//   return std::sin(x2);
// #  else
//   UNUSED(x1);
//   UNUSED(x2);
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// template <>
// Inline auto Grid<TWO_D>::Jacobian_22(const real_t& x1, const real_t& x2) const -> real_t {
// #  ifdef HARDCODE_SPHERICAL_COORDS
//   return x1 * std::cos(x2);
// #  else
//   UNUSED(x1);
//   UNUSED(x2);
//   throw std::logic_error("# NOT IMPLEMENTED.");
// #  endif
// }
//
// // 3d
// template <>
// Inline auto Grid<THREE_D>::Jacobian_11(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_12(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_13(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_21(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_22(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_23(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_31(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_32(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// template <>
// Inline auto Grid<THREE_D>::Jacobian_33(const real_t& x1, const real_t& x2, const real_t& x3) const
//     -> real_t {
//   throw std::logic_error("# NOT IMPLEMENTED.");
// }
//
// #endif
