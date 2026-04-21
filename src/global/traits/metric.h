/**
 * @file traits/metric.h
 * @brief Defines a set of traits to check if a metric class satisfies certain conditions
 * @implements
 *   - HasD<> - checks if metric has Dim static member
 *   - HasPrtlDim<> - checks if metric has PrtlDim static member
 *   - HasCoordType<> - checks if metric has CoordType static member
 *   - HasH_ij<> - checks if metric has h_ij method
 *   - HasHij<> - checks if metric has hij method
 *   - HasSqrtH_ij<> - checks if metric has sqrt_h_ij method
 *   - HasSqrtDetH<> - checks if metric has sqrt_det_h method
 *   - HasSqrtDetHTilde<> - checks if metric has sqrt_det_h_tilde method
 *   - HasPolarArea<> - checks if metric has polar_area method
 *   - HasTotVolume<> - checks if metric has totVolume method
 *   - HasTransform_i<> - checks if metric has transform method for single component
 *   - HasTransform<> - checks if metric has transform method for vector
 *   - HasTransformXYZ<> - checks if metric has transform_xyz method for vector
 *   - HasConvert_i<> - checks if metric has convert method for single component
 *   - HasConvert<> - checks if metric has convert method for vector
 *   - HasConvertXYZ<> - checks if metric has convert_xyz method for vector
 *   - HasAlpha<> - checks if metric has alpha method
 *   - HasBeta1<> - checks if metric has beta1 method
 *   - HasMetricDerivatives<> - checks if metric has metric derivatives methods
 *   - CurvilinearMetric<> - checks if metric is in curvilinear coordinates
 *   - ::SRMetricClass<> - checks if metric is a special-relativistic metric
 *   - ::GRMetricClass<> - checks if metric is a general-relativistic metric
 *   - ::MetricClass<> - checks if metric is a valid metric
 * @namespaces:
 *   - ::traits::metric::
 */
#ifndef TRAITS_METRIC_H
#define TRAITS_METRIC_H

#include "enums.h"
#include "global.h"

namespace traits::metric {

  template <class M>
  concept HasD = requires {
    { M::Dim } -> std::convertible_to<Dimension>;
  };

  template <class M>
  concept HasPrtlDim = requires {
    { M::PrtlDim } -> std::convertible_to<Dimension>;
  };

  template <class M>
  concept HasCoordType = requires {
    { M::CoordType } -> std::convertible_to<ntt::Coord::type>;
  };

  template <class M>
  concept HasH_ij = requires(const M& m, const coord_t<M::Dim>& xi) {
    { m.template h_<1, 1>(xi) } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasHij = requires(const M& m, const coord_t<M::Dim>& xi) {
    { m.template h<1, 1>(xi) } -> std::convertible_to<real_t>;
  };
  template <class M>
  concept HasSqrtH_ij = requires(const M& m, const coord_t<M::Dim>& xi) {
    { m.template sqrt_h_<1, 1>(xi) } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasSqrtDetH = requires(const M& m, const coord_t<M::Dim>& xi) {
    { m.sqrt_det_h(xi) } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasSqrtDetHTilde = requires(const M& m, const coord_t<M::Dim>& xi) {
    { m.sqrt_det_h_tilde(xi) } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasPolarArea = requires(const M& m, real_t xi_2) {
    { m.polar_area(xi_2) } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasTotVolume = requires(const M& m) {
    { m.totVolume() } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasTransform_i = requires(const M&               m,
                                    const coord_t<M::Dim>& xi,
                                    real_t                 v_in) {
    {
      m.template transform<1, Idx::U, Idx::D>(xi, v_in)
    } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasTransform = requires(const M&               m,
                                  const coord_t<M::Dim>& xi,
                                  const vec_t<Dim::_3D>& v_in,
                                  vec_t<Dim::_3D>&       v_out) {
    {
      m.template transform<Idx::U, Idx::D>(xi, v_in, v_out)
    } -> std::same_as<void>;
  };

  template <class M>
  concept HasTransformXYZ = requires(const M&                   m,
                                     const coord_t<M::PrtlDim>& xi,
                                     const vec_t<Dim::_3D>&     v_in,
                                     vec_t<Dim::_3D>&           v_out) {
    {
      m.template transform_xyz<Idx::XYZ, Idx::D>(xi, v_in, v_out)
    } -> std::same_as<void>;
  };

  template <class M>
  concept HasConvert_i = requires(const M& m, real_t x) {
    {
      m.template convert<1, Crd::Cd, Crd::Ph>(x)
    } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasConvert = requires(const M&               m,
                                const coord_t<M::Dim>& x_in,
                                coord_t<M::Dim>&       x_out) {
    { m.template convert<Crd::Cd, Crd::Ph>(x_in, x_out) } -> std::same_as<void>;
  };

  template <class M>
  concept HasConvertXYZ = requires(const M&                   m,
                                   const coord_t<M::PrtlDim>& x_in,
                                   coord_t<M::PrtlDim>&       x_out) {
    {
      m.template convert_xyz<Crd::Cd, Crd::XYZ>(x_in, x_out)
    } -> std::same_as<void>;
  };

  template <class M>
  concept HasAlpha = requires(const M& m, const coord_t<M::Dim>& xi) {
    { m.alpha(xi) } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasBeta1 = requires(const M& m, const coord_t<M::Dim>& xi) {
    { m.beta1(xi) } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept HasMetricDerivatives = requires(const M& m, const coord_t<M::Dim>& xi) {
    { m.dr_alpha(xi) } -> std::convertible_to<real_t>;
    { m.dr_beta1(xi) } -> std::convertible_to<real_t>;
    { m.dr_h11(xi) } -> std::convertible_to<real_t>;
    { m.dr_h22(xi) } -> std::convertible_to<real_t>;
    { m.dr_h33(xi) } -> std::convertible_to<real_t>;
    { m.dr_h13(xi) } -> std::convertible_to<real_t>;
    { m.dt_alpha(xi) } -> std::convertible_to<real_t>;
    { m.dt_beta1(xi) } -> std::convertible_to<real_t>;
    { m.dt_h11(xi) } -> std::convertible_to<real_t>;
    { m.dt_h22(xi) } -> std::convertible_to<real_t>;
    { m.dt_h33(xi) } -> std::convertible_to<real_t>;
    { m.dt_h13(xi) } -> std::convertible_to<real_t>;
  };

  template <class M>
  concept CurvilinearMetric = HasPolarArea<M> and HasSqrtDetHTilde<M>;

} // namespace traits::metric

template <class M>
concept SRMetricClass = ::traits::metric::HasD<M> and
                        ::traits::metric::HasPrtlDim<M> and
                        ::traits::metric::HasCoordType<M> and
                        ::traits::metric::HasH_ij<M> and
                        ::traits::metric::HasSqrtH_ij<M> and
                        ::traits::metric::HasSqrtDetH<M> and
                        ::traits::metric::HasTransform_i<M> and
                        ::traits::metric::HasTransform<M> and
                        ::traits::metric::HasTransformXYZ<M> and
                        ::traits::metric::HasConvert_i<M> and
                        ::traits::metric::HasConvert<M> and
                        ::traits::metric::HasConvertXYZ<M>;

template <class M>
concept GRMetricClass =
  ::traits::metric::CurvilinearMetric<M> and ::traits::metric::HasD<M> and
  ::traits::metric::HasPrtlDim<M> and ::traits::metric::HasCoordType<M> and
  ::traits::metric::HasHij<M> and ::traits::metric::HasH_ij<M> and
  ::traits::metric::HasSqrtDetH<M> and ::traits::metric::HasTransform_i<M> and
  ::traits::metric::HasTransform<M> and ::traits::metric::HasConvert_i<M> and
  ::traits::metric::HasConvert<M> and ::traits::metric::HasAlpha<M> and
  ::traits::metric::HasBeta1<M> and ::traits::metric::HasMetricDerivatives<M>;

template <class M>
concept MetricClass = SRMetricClass<M> or GRMetricClass<M>;

#endif // TRAITS_METRIC_H