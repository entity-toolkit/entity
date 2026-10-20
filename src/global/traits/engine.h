/**
 * @file traits/engine.h
 * @brief Defines a set of traits to check if an engine class satisfies certain conditions
 * @implements
 *   - IsSR<> - checks if engine is a special relativistic engine
 *   - isSR() - checks if a SimEngine enum instance corresponds to a special relativistic engine
 *   - IsGR<> - checks if engine is a general relativistic engine
 *   - isGR() - checks if a SimEngine enum instance corresponds to a general relativistic engine
 *   - VelocitiesInCartesianBasis<> - checks if engine has velocities in Cartesian basis
 *   - VelocitiesInCovariantBasis<> - checks if engine has velocities in covariant basis
 *   - StressEnergyInTetradBasis<> - checks if engine needs stress-energy tensor in tetrad basis
 *   - StressEnergyInContravariantBasis<> - checks if engine needs stress-energy tensor in contravariant basis
 *   - UserFieldsInTetradBasis<> - checks if engine expects user-defined fields in tetrad basis
 *   - userFieldsInTetradBasis() - checks if a SimEngine enum instance corresponds
 * to an engine that expects user-defined fields in tetrad basis
 *   - HasImplicitPhiCoordinate<> - checks if engine has particles with an implicit phi coordinate
 * @namespaces:
 *   - ::traits::engine::
 */
#ifndef TRAITS_ENGINE_H
#define TRAITS_ENGINE_H

#include "enums.h"

#include "traits/metric.h"

namespace traits::engine {

  template <ntt::SimEngine::type S>
  concept IsSR = (S == ntt::SimEngine::SRPIC) or (S == ntt::SimEngine::HYBRID);

  [[nodiscard]]
  constexpr auto isSR(ntt::SimEngine s) noexcept -> bool {
    return (s == ntt::SimEngine::SRPIC) or (s == ntt::SimEngine::HYBRID);
  }

  template <ntt::SimEngine::type S>
  concept IsGR = (S == ntt::SimEngine::GRPIC);

  [[nodiscard]]
  constexpr auto isGR(ntt::SimEngine s) noexcept -> bool {
    return s == ntt::SimEngine::GRPIC;
  }

  template <ntt::SimEngine::type S>
  concept VelocitiesInCartesianBasis = (S == ntt::SimEngine::SRPIC) or
                                       (S == ntt::SimEngine::HYBRID);

  template <ntt::SimEngine::type S>
  concept VelocitiesInCovariantBasis = (S == ntt::SimEngine::GRPIC);

  template <ntt::SimEngine::type S>
  concept StressEnergyInTetradBasis = (S == ntt::SimEngine::SRPIC) or
                                      (S == ntt::SimEngine::HYBRID);

  template <ntt::SimEngine::type S>
  concept StressEnergyInContravariantBasis = (S == ntt::SimEngine::GRPIC);

  template <ntt::SimEngine::type S>
  concept DefinesEM0Fields = (S == ntt::SimEngine::GRPIC) or
                             (S == ntt::SimEngine::HYBRID);

  template <ntt::SimEngine::type S>
  concept DefinesAuxFields = (S == ntt::SimEngine::GRPIC) or
                             (S == ntt::SimEngine::HYBRID);

  template <ntt::SimEngine::type S>
  concept DefinesCur0Fields = (S == ntt::SimEngine::GRPIC);

  template <ntt::SimEngine::type S>
  concept WriteEM0FieldToCheckpoint = (S == ntt::SimEngine::GRPIC);

  template <ntt::SimEngine::type S>
  concept WriteCurFieldToCheckpoint = (S == ntt::SimEngine::GRPIC);

  template <ntt::SimEngine::type S>
  concept UserFieldsInTetradBasis = (S == ntt::SimEngine::SRPIC) or
                                    (S == ntt::SimEngine::HYBRID);

  template <ntt::SimEngine::type S>
  concept UserFieldsInContravariantBasis = (S == ntt::SimEngine::GRPIC);

  [[nodiscard]]
  constexpr auto userFieldsInTetradBasis(ntt::SimEngine s) noexcept -> bool {
    return (s == ntt::SimEngine::SRPIC) or (s == ntt::SimEngine::HYBRID);
  }

  template <ntt::SimEngine::type S, class M>
  concept HasImplicitPhiCoordinate = IsSR<S> and ::traits::metric::HasPrtlDim<M> and
                                     ::traits::metric::HasD<M> and
                                     M::Dim == Dim::_2D and M::PrtlDim == Dim::_3D;

} // namespace traits::engine

#endif // TRAITS_ENGINE_H
