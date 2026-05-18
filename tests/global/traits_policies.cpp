#include "global.h"

#include "traits/policies.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/pushers/context.h"

#include <Kokkos_Pair.hpp>

#include <vector>

using namespace ntt;

// Minimal mock metric providing PrtlDim and Dim
struct MockMetric {
  static constexpr Dimension Dim { Dimension::_2D };
  static constexpr Dimension PrtlDim { Dimension::_2D };
};

// --- NoPolicy variants ---

static_assert(traits::emission::IsNoPolicy<traits::emission::NoPolicy_t>);
static_assert(not traits::emission::IsNoPolicy<int>);
static_assert(not traits::emission::IsNoPolicy<MockMetric>);

static_assert(traits::extfields::IsNoPolicy<traits::extfields::NoPolicy_t>);
static_assert(not traits::extfields::IsNoPolicy<int>);

static_assert(
  traits::custom_prtl_update::IsNoPolicy<traits::custom_prtl_update::NoPolicy_t>);
static_assert(not traits::custom_prtl_update::IsNoPolicy<int>);

// NoPolicy satisfies the composite concepts
static_assert(EmissionPolicyClass<traits::emission::NoPolicy_t, MockMetric>);

struct MockFieldSetter {
  real_t ex1(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }
};

static_assert(ExtFieldsPolicyClass<traits::extfields::NoPolicy_t, Dimension::_2D>);
static_assert(ExtFieldsPolicyClass<MockFieldSetter, Dimension::_2D>);

static_assert(
  CustomParticleUpdatePolicyClass<traits::custom_prtl_update::NoPolicy_t, MockMetric>);

// --- EmissionPolicyClass with a real emission policy ---

struct MockPayload {};

struct ValidEmissionPolicy {
  using Payload = MockPayload;

  std::vector<npart_t> numbers_injected() {
    return {};
  }

  std::vector<spidx_t> emitted_species_indices() const {
    return {};
  }

  Kokkos::pair<bool, bool> shouldEmit(const coord_t<Dimension::_2D>&,
                                      const coord_t<Dimension::_2D>&,
                                      const vec_t<Dim::_3D>&,
                                      const vec_t<Dim::_3D>&,
                                      const vec_t<Dim::_3D>&,
                                      vec_t<Dim::_3D>&,
                                      Payload&) const {
    return { false, false };
  }

  void emit(const tuple_t<int, Dimension::_2D>&,
            const tuple_t<prtldx_t, Dimension::_2D>&,
            const vec_t<Dim::_3D>&,
            real_t,
            real_t,
            const Payload&) const {}
};

// Individual emission traits
static_assert(traits::emission::HasPayload<ValidEmissionPolicy>);
static_assert(traits::emission::HasNumbersInjected<ValidEmissionPolicy>);
static_assert(traits::emission::HasEmittedSpeciesIndices<ValidEmissionPolicy>);
static_assert(traits::emission::HasShouldEmit<ValidEmissionPolicy, MockMetric>);
static_assert(traits::emission::HasEmit<ValidEmissionPolicy, MockMetric>);

static_assert(EmissionPolicyClass<ValidEmissionPolicy, MockMetric>);

// Missing Payload fails the concept
struct NoPayload_EmissionPolicy {
  std::vector<npart_t> numbers_injected() {
    return {};
  }

  std::vector<spidx_t> emitted_species_indices() const {
    return {};
  }
};

static_assert(not traits::emission::HasPayload<NoPayload_EmissionPolicy>);
static_assert(not EmissionPolicyClass<NoPayload_EmissionPolicy, MockMetric>);

// --- ExtFieldsPolicyClass ---

struct WithFx1 {
  real_t fx1(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }
};

struct Empty {};

static_assert(ExtFieldsPolicyClass<WithFx1, Dimension::_2D>);
static_assert(not ExtFieldsPolicyClass<Empty, Dimension::_2D>);

// --- CustomParticleUpdatePolicyClass with a real updater ---

struct ValidCustomPrtlUpdate {
  void operator()(prtlidx_t,
                  const kernel::sr::PusherContext&,
                  const kernel::sr::PusherBoundaries<Dimension::_2D>&,
                  const ntt::ParticleArrays&,
                  const MockMetric&) const {}
};

static_assert(CustomParticleUpdatePolicyClass<ValidCustomPrtlUpdate, MockMetric>);

// Wrong signature: missing the metric argument
struct BadCustomPrtlUpdate {
  void operator()(prtlidx_t,
                  const kernel::sr::PusherContext&,
                  const kernel::sr::PusherBoundaries<Dimension::_2D>&,
                  const ntt::ParticleArrays&) const {}
};

static_assert(not CustomParticleUpdatePolicyClass<BadCustomPrtlUpdate, MockMetric>);

auto main() -> int {
  return 0;
}
