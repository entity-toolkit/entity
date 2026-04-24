#include "enums.h"
#include "global.h"

#include "traits/archetypes.h"
#include "utils/numeric.h"

#include <Kokkos_Pair.hpp>

using namespace ntt;

// --- EnrgDistClass ---

struct ValidEnrgDist {
  static constexpr Dimension D { Dimension::_2D };

  void operator()(const coord_t<Dimension::_2D>&, vec_t<Dim::_3D>&) const {}
};

struct WrongD_EnrgDist {
  void operator()(const coord_t<Dimension::_3D>&, vec_t<Dim::_3D>&) const {}
};

struct WrongReturn_EnrgDist {
  static constexpr Dimension D { Dimension::_2D };

  int operator()(const coord_t<Dimension::_2D>&, vec_t<Dim::_3D>&) const {
    return 0;
  }
};

static_assert(EnrgDistClass<ValidEnrgDist, Dimension::_2D>);
static_assert(not EnrgDistClass<WrongD_EnrgDist, Dimension::_2D>);
static_assert(not EnrgDistClass<WrongReturn_EnrgDist, Dimension::_2D>);

// --- SpatialDistClass ---

struct ValidSpatialDist {
  static constexpr Dimension D { Dimension::_1D };

  real_t operator()(const coord_t<Dimension::_1D>&) const {
    return ZERO;
  }
};

struct NoD_SpatialDist {
  real_t operator()(const coord_t<Dimension::_1D>&) const {
    return ZERO;
  }
};

struct WrongArg_SpatialDist {
  static constexpr Dimension D { Dimension::_1D };

  real_t operator()(real_t) const {
    return ZERO;
  }
};

static_assert(SpatialDistClass<ValidSpatialDist>);
static_assert(not SpatialDistClass<NoD_SpatialDist>);
static_assert(not SpatialDistClass<WrongArg_SpatialDist>);

// --- Individual field component traits ---

struct WithEx1 {
  real_t ex1(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }
};

struct WithBx1Bx2Bx3 {
  real_t bx1(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }

  real_t bx2(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }

  real_t bx3(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }
};

struct WithDx1Dx2Dx3 {
  real_t dx1(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }

  real_t dx2(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }

  real_t dx3(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }
};

struct WithFx1Fx2Fx3 {
  real_t fx1(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }

  real_t fx2(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }

  real_t fx3(const coord_t<Dimension::_2D>&) const {
    return ZERO;
  }
};

struct WithConditionalEx1Bx2 {
  Kokkos::pair<bool, real_t> ex1(const coord_t<Dimension::_2D>&,
                                 const vec_t<Dim::_3D>&,
                                 const vec_t<Dim::_3D>&) const {
    return { false, ZERO };
  }

  Kokkos::pair<bool, real_t> bx2(const coord_t<Dimension::_2D>&,
                                 const vec_t<Dim::_3D>&,
                                 const vec_t<Dim::_3D>&) const {
    return { false, ZERO };
  }
};

struct Empty {};

static_assert(traits::fieldsetter::HasEx1<WithEx1, Dimension::_2D>);
static_assert(not traits::fieldsetter::HasEx1<Empty, Dimension::_2D>);
static_assert(not traits::fieldsetter::HasEx1<WithBx1Bx2Bx3, Dimension::_2D>);

static_assert(traits::fieldsetter::HasBx1<WithBx1Bx2Bx3, Dimension::_2D>);
static_assert(traits::fieldsetter::HasBx2<WithBx1Bx2Bx3, Dimension::_2D>);
static_assert(traits::fieldsetter::HasBx3<WithBx1Bx2Bx3, Dimension::_2D>);
static_assert(not traits::fieldsetter::HasBx1<Empty, Dimension::_2D>);

static_assert(traits::fieldsetter::HasDx1<WithDx1Dx2Dx3, Dimension::_2D>);
static_assert(traits::fieldsetter::HasDx2<WithDx1Dx2Dx3, Dimension::_2D>);
static_assert(traits::fieldsetter::HasDx3<WithDx1Dx2Dx3, Dimension::_2D>);

static_assert(traits::fieldsetter::HasFx1<WithFx1Fx2Fx3, Dimension::_2D>);
static_assert(traits::fieldsetter::HasFx2<WithFx1Fx2Fx3, Dimension::_2D>);
static_assert(traits::fieldsetter::HasFx3<WithFx1Fx2Fx3, Dimension::_2D>);

// conditional variants require 3-arg signature returning Kokkos::pair<bool, real_t>
static_assert(
  traits::fieldsetter::HasConditionalEx1<WithConditionalEx1Bx2, Dimension::_2D>);
static_assert(
  traits::fieldsetter::HasConditionalBx2<WithConditionalEx1Bx2, Dimension::_2D>);
// plain ex1/bx2 (wrong signature) do not satisfy conditional traits
static_assert(not traits::fieldsetter::HasConditionalEx1<WithEx1, Dimension::_2D>);
static_assert(
  not traits::fieldsetter::HasConditionalBx1<WithBx1Bx2Bx3, Dimension::_2D>);

// --- SRFieldSetterClass: at least one E or B component ---

static_assert(SRFieldSetterClass<WithEx1, Dimension::_2D>);
static_assert(SRFieldSetterClass<WithBx1Bx2Bx3, Dimension::_2D>);
static_assert(not SRFieldSetterClass<Empty, Dimension::_2D>);
static_assert(not SRFieldSetterClass<WithDx1Dx2Dx3, Dimension::_2D>);
static_assert(not SRFieldSetterClass<WithFx1Fx2Fx3, Dimension::_2D>);

// --- FieldSetterClass SRPIC: at least one E or B ---

static_assert(FieldSetterClass<WithEx1, SimEngine::SRPIC, Dimension::_2D>);
static_assert(FieldSetterClass<WithBx1Bx2Bx3, SimEngine::SRPIC, Dimension::_2D>);
static_assert(not FieldSetterClass<Empty, SimEngine::SRPIC, Dimension::_2D>);
static_assert(
  not FieldSetterClass<WithDx1Dx2Dx3, SimEngine::SRPIC, Dimension::_2D>);

// --- FieldSetterClass GRPIC: all three B or all three D ---

static_assert(FieldSetterClass<WithBx1Bx2Bx3, SimEngine::GRPIC, Dimension::_2D>);
static_assert(FieldSetterClass<WithDx1Dx2Dx3, SimEngine::GRPIC, Dimension::_2D>);
static_assert(not FieldSetterClass<WithEx1, SimEngine::GRPIC, Dimension::_2D>);
static_assert(not FieldSetterClass<Empty, SimEngine::GRPIC, Dimension::_2D>);

// --- ConditionalSRFieldSetterClass: at least one conditional E or B ---

static_assert(ConditionalSRFieldSetterClass<WithConditionalEx1Bx2, Dimension::_2D>);
static_assert(not ConditionalSRFieldSetterClass<WithEx1, Dimension::_2D>);
static_assert(not ConditionalSRFieldSetterClass<Empty, Dimension::_2D>);

auto main() -> int {
  return 0;
}
