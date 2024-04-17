/**
 * @file archetypes/problem_generator.hpp
 * @brief Base class for all problem generators
 * @implements
 *   - ntt::ProblemGenerator<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - framework/parameters.h
 * @namespace
 *   - ntt::
 * @note
 * To have easier access to variables inside ProblemGenerator in children
 * classes, one should simply add:
 * ```c++
 * using ProblemGenerator<S, M>::D;
 * using ProblemGenerator<S, M>::C;
 * using ProblemGenerator<S, M>::params;
 * ```
 * in the child class. This will allow to access these variables without
 * the need to use `this->` or `ProblemGenerator<S, M>::` prefix.
 */

#ifndef ARCHETYPES_PROBLEM_GENERATOR_HPP
#define ARCHETYPES_PROBLEM_GENERATOR_HPP

#include "enums.h"
#include "global.h"

#include "framework/parameters.h"

namespace ntt {

  template <SimEngine::type S, class M>
  struct ProblemGenerator {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr bool      is_pgen { true };
    static constexpr Dimension D { M::Dim };
    static constexpr Coord     C { M::CoordType };

    SimulationParams& params;

    ProblemGenerator(SimulationParams& p) : params { p } {}

    ~ProblemGenerator() = default;
  };

} // namespace ntt

#endif // ARCHETYPES_PROBLEM_GENERATOR_HPP
