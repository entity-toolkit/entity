#include "traits/metric.h"

#include "enums.h"
#include "global.h"

#include "utils/numeric.h"

using namespace ntt;

// Minimal mock satisfying all SRMetricClass requirements.
// Methods take/return the correct types; bodies are trivial stubs.
struct MockSRMetric {
  static constexpr Dimension        Dim     { Dimension::_2D };
  static constexpr Dimension        PrtlDim { Dimension::_2D };
  static constexpr ntt::Coord::type CoordType { ntt::Coord::type::Cartesian };

  template <int I, int J>
  real_t h_(const coord_t<Dimension::_2D>&) const { return ZERO; }

  template <int I, int J>
  real_t sqrt_h_(const coord_t<Dimension::_2D>&) const { return ZERO; }

  real_t sqrt_det_h(const coord_t<Dimension::_2D>&) const { return ZERO; }

  // single-component transform: transform<I, In, Out>(xi, scalar) -> real_t
  template <int I, Idx In, Idx Out>
  real_t transform(const coord_t<Dimension::_2D>&, real_t) const { return ZERO; }

  // vector transform: transform<In, Out>(xi, v_in, v_out) -> void
  template <Idx In, Idx Out>
  void transform(const coord_t<Dimension::_2D>&,
                 const vec_t<Dim::_3D>&,
                 vec_t<Dim::_3D>&) const {}

  // xyz transform
  template <Idx In, Idx Out>
  void transform_xyz(const coord_t<Dimension::_2D>&,
                     const vec_t<Dim::_3D>&,
                     vec_t<Dim::_3D>&) const {}

  // single-component convert: convert<I, In, Out>(scalar) -> real_t
  template <int I, Crd In, Crd Out>
  real_t convert(real_t) const { return ZERO; }

  // vector convert: convert<In, Out>(x_in, x_out) -> void
  template <Crd In, Crd Out>
  void convert(const coord_t<Dimension::_2D>&, coord_t<Dimension::_2D>&) const {}

  // xyz convert
  template <Crd In, Crd Out>
  void convert_xyz(const coord_t<Dimension::_2D>&, coord_t<Dimension::_2D>&) const {}
};

// Individual SR traits
static_assert(traits::metric::HasD<MockSRMetric>);
static_assert(traits::metric::HasPrtlDim<MockSRMetric>);
static_assert(traits::metric::HasCoordType<MockSRMetric>);
static_assert(traits::metric::HasH_ij<MockSRMetric>);
static_assert(traits::metric::HasSqrtH_ij<MockSRMetric>);
static_assert(traits::metric::HasSqrtDetH<MockSRMetric>);
static_assert(traits::metric::HasTransform_i<MockSRMetric>);
static_assert(traits::metric::HasTransform<MockSRMetric>);
static_assert(traits::metric::HasTransformXYZ<MockSRMetric>);
static_assert(traits::metric::HasConvert_i<MockSRMetric>);
static_assert(traits::metric::HasConvert<MockSRMetric>);
static_assert(traits::metric::HasConvertXYZ<MockSRMetric>);

// Composite SR concept
static_assert(SRMetricClass<MockSRMetric>);

// Negative: missing Dim
struct NoD_Metric {
  static constexpr Dimension        PrtlDim { Dimension::_2D };
  static constexpr ntt::Coord::type CoordType { ntt::Coord::type::Cartesian };
};

static_assert(not traits::metric::HasD<NoD_Metric>);
static_assert(not SRMetricClass<NoD_Metric>);

// Negative: missing CoordType
struct NoCoordType_Metric {
  static constexpr Dimension Dim     { Dimension::_2D };
  static constexpr Dimension PrtlDim { Dimension::_2D };
};

static_assert(not traits::metric::HasCoordType<NoCoordType_Metric>);

// Negative: missing sqrt_det_h
struct NoSqrtDetH_Metric : MockSRMetric {
  real_t sqrt_det_h(const coord_t<Dimension::_2D>&) const = delete;
};

static_assert(not traits::metric::HasSqrtDetH<NoSqrtDetH_Metric>);
static_assert(not SRMetricClass<NoSqrtDetH_Metric>);

// --- GRMetricClass ---

struct MockGRMetric : MockSRMetric {
  // additionally needed for GR
  template <int I, int J>
  real_t h(const coord_t<Dimension::_2D>&) const { return ZERO; }

  real_t sqrt_det_h_tilde(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t polar_area(real_t) const { return ZERO; }

  real_t alpha(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t beta1(const coord_t<Dimension::_2D>&) const { return ZERO; }

  real_t dr_alpha(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dr_beta1(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dr_h11(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dr_h22(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dr_h33(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dr_h13(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dt_alpha(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dt_beta1(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dt_h11(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dt_h22(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dt_h33(const coord_t<Dimension::_2D>&) const { return ZERO; }
  real_t dt_h13(const coord_t<Dimension::_2D>&) const { return ZERO; }
};

// Individual GR-only traits
static_assert(traits::metric::HasHij<MockGRMetric>);
static_assert(traits::metric::HasAlpha<MockGRMetric>);
static_assert(traits::metric::HasBeta1<MockGRMetric>);
static_assert(traits::metric::HasMetricDerivatives<MockGRMetric>);
static_assert(traits::metric::HasSqrtDetHTilde<MockGRMetric>);
static_assert(traits::metric::HasPolarArea<MockGRMetric>);
static_assert(traits::metric::CurvilinearMetric<MockGRMetric>);

// SR metric does not have GR-specific traits
static_assert(not traits::metric::HasHij<MockSRMetric>);
static_assert(not traits::metric::HasAlpha<MockSRMetric>);
static_assert(not traits::metric::CurvilinearMetric<MockSRMetric>);

// Composite GR concept
static_assert(GRMetricClass<MockGRMetric>);
static_assert(not GRMetricClass<MockSRMetric>);

// MetricClass accepts either SR or GR
static_assert(MetricClass<MockSRMetric>);
static_assert(MetricClass<MockGRMetric>);
static_assert(not MetricClass<NoD_Metric>);

auto main() -> int {
  return 0;
}
