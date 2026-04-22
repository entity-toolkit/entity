#include "traits/pgen.h"

#include "global.h"

#include "utils/numeric.h"

using namespace ntt;

// Dummy domain type used where concept requires a domain argument
struct DummyDomain {};

// Stub field setter (satisfies SRFieldSetterClass via ex1)
struct StubFieldSetter {
  real_t ex1(const coord_t<Dimension::_1D>&) const { return ZERO; }
};

// --- HasD ---

struct WithD {
  static constexpr Dimension D { Dimension::_1D };
};

struct NoD {};

static_assert(traits::pgen::HasD<WithD>);
static_assert(not traits::pgen::HasD<NoD>);

// --- HasInitFlds / HasExtCurrent ---

struct WithInitFlds {
  StubFieldSetter init_flds {};
};

struct WithExtCurrent {
  StubFieldSetter ext_current {};
};

static_assert(traits::pgen::HasInitFlds<WithInitFlds>);
static_assert(not traits::pgen::HasInitFlds<NoD>);
static_assert(traits::pgen::HasExtCurrent<WithExtCurrent>);
static_assert(not traits::pgen::HasExtCurrent<NoD>);

// --- HasEmissionPolicy / HasCustomPrtlUpdate ---

struct WithEmissionPolicy {
  void EmissionPolicy(simtime_t, spidx_t, DummyDomain&) const {}
};

struct WithCustomPrtlUpdate {
  void CustomParticleUpdate(simtime_t, spidx_t, DummyDomain&) const {}
};

static_assert(traits::pgen::HasEmissionPolicy<WithEmissionPolicy, DummyDomain>);
static_assert(not traits::pgen::HasEmissionPolicy<NoD, DummyDomain>);
static_assert(traits::pgen::HasCustomPrtlUpdate<WithCustomPrtlUpdate, DummyDomain>);
static_assert(not traits::pgen::HasCustomPrtlUpdate<NoD, DummyDomain>);

// --- HasExternalFields ---
// return type must expose .first as bool and .second as anything

struct WithExternalFields {
  auto ExternalFields(simtime_t, spidx_t, DummyDomain&) const
    -> std::pair<bool, StubFieldSetter> {
    return { true, {} };
  }
};

// .first is int, not bool
struct BadExternalFields {
  auto ExternalFields(simtime_t, spidx_t, DummyDomain&) const
    -> std::pair<int, StubFieldSetter> {
    return { 1, {} };
  }
};

static_assert(traits::pgen::HasExternalFields<WithExternalFields, DummyDomain>);
static_assert(not traits::pgen::HasExternalFields<BadExternalFields, DummyDomain>);
static_assert(not traits::pgen::HasExternalFields<NoD, DummyDomain>);

// --- HasInitPrtls ---

struct WithInitPrtls {
  void InitPrtls(DummyDomain&) {}
};

static_assert(traits::pgen::HasInitPrtls<WithInitPrtls, DummyDomain>);
static_assert(not traits::pgen::HasInitPrtls<NoD, DummyDomain>);

// --- HasAtmFields / HasMatchFields / HasMatchFieldsInX1/X2/X3 ---

struct WithAllMatchFields {
  StubFieldSetter AtmFields(simtime_t) const { return {}; }
  StubFieldSetter MatchFields(simtime_t) const { return {}; }
  StubFieldSetter MatchFieldsInX1(simtime_t) const { return {}; }
  StubFieldSetter MatchFieldsInX2(simtime_t) const { return {}; }
  StubFieldSetter MatchFieldsInX3(simtime_t) const { return {}; }
};

static_assert(traits::pgen::HasAtmFields<WithAllMatchFields>);
static_assert(traits::pgen::HasMatchFields<WithAllMatchFields>);
static_assert(traits::pgen::HasMatchFieldsInX1<WithAllMatchFields>);
static_assert(traits::pgen::HasMatchFieldsInX2<WithAllMatchFields>);
static_assert(traits::pgen::HasMatchFieldsInX3<WithAllMatchFields>);
static_assert(not traits::pgen::HasAtmFields<NoD>);
static_assert(not traits::pgen::HasMatchFields<NoD>);

// --- HasFixFieldsConst ---

struct WithFixFieldsConst {
  auto FixFieldsConst(simtime_t, const bc_in&, ntt::em) const
    -> std::pair<real_t, bool> {
    return { ZERO, false };
  }
};

// wrong return type: plain int is not convertible to std::pair<real_t, bool>
struct BadFixFieldsConst {
  int FixFieldsConst(simtime_t, const bc_in&, ntt::em) const { return 0; }
};

static_assert(traits::pgen::HasFixFieldsConst<WithFixFieldsConst>);
static_assert(not traits::pgen::HasFixFieldsConst<NoD>);
static_assert(not traits::pgen::HasFixFieldsConst<BadFixFieldsConst>);

// --- HasCustomPostStep ---

struct WithCustomPostStep {
  void CustomPostStep(timestep_t, simtime_t, DummyDomain&) {}
};

static_assert(traits::pgen::HasCustomPostStep<WithCustomPostStep, DummyDomain>);
static_assert(not traits::pgen::HasCustomPostStep<NoD, DummyDomain>);

// --- HasCustomFieldOutput ---

struct WithCustomFieldOutput {
  static constexpr Dimension D { Dimension::_2D };
  void CustomFieldOutput(const std::string&,
                         ndfield_t<Dimension::_2D, 6>&,
                         index_t,
                         timestep_t,
                         simtime_t,
                         const DummyDomain&) {}
};

static_assert(traits::pgen::HasCustomFieldOutput<WithCustomFieldOutput, DummyDomain>);
static_assert(not traits::pgen::HasCustomFieldOutput<NoD, DummyDomain>);

// --- HasCustomStatOutput ---

struct WithCustomStatOutput {
  real_t CustomStat(const std::string&, timestep_t, simtime_t, const DummyDomain&) {
    return ZERO;
  }
};

static_assert(traits::pgen::HasCustomStatOutput<WithCustomStatOutput, DummyDomain>);
static_assert(not traits::pgen::HasCustomStatOutput<NoD, DummyDomain>);

auto main() -> int {
  return 0;
}
