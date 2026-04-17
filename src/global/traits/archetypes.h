/**
 * @file traits/archetypes.h
 * @brief Defines a set of traits for commonly used class archetypes
 * @implements
 *   - EnrgDistClass<> - checks if a class can be used as an energy distribution
 *   - SpatialDistClass<> - checks if a class can be used as a spatial distribution
 * @namespaces:
 */
#ifndef GLOBAL_TRAITS_ARCHETYPES_H
#define GLOBAL_TRAITS_ARCHETYPES_H

#include "global.h"

template <class ED>
concept EnrgDistClass = requires(const ED&             edist,
                                 const coord_t<ED::D>& x_Ph,
                                 vec_t<Dim::_3D>&      v) {
  { edist(x_Ph, v) } -> std::same_as<void>;
};

template <class SD>
concept SpatialDistClass = requires(const SD& sdist, const coord_t<SD::D>& x_Ph) {
  { sdist(x_Ph) } -> std::convertible_to<real_t>;
};

#endif // GLOBAL_TRAITS_ARCHETYPES_H