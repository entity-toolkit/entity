/**
 * @file metrics/boyer_lindq_tp.h
 * @brief metric along constant flux surfaces in force-free fields, based on totoised Boyer-Lindquist-psi coordinates
 * @implements
 *   - metric::BoyerLindqTP<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 * !TODO
 * None radial surfaces needs to be implemented (dpsi_dr != 0).
 */

#ifndef METRICS_BOYER_LINDQ_TP_H
#define METRICS_BOYER_LINDQ_TP_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include "metrics/metric_base.h"

#include <map>
#include <string>
#include <vector>


namespace metric {

  template <Dimension D>
  class BoyerLindqTP : public MetricBase<D> {
    static_assert(D == Dim::_1D, "Only 1D boyer_lindq_tp is available");

  private:
    // Spin parameter, in [0,1[
    // and horizon size in units of rg
    // all physical extents are in units of rg
    const real_t a, rg_, psi0, th0;
    const real_t rh, rh_m, psi, bt, Omega, dpsi_dth, dbt_dth;
    const real_t eta_max, eta_min;
    const real_t d_eta, d_eta_inv;

    Inline auto Delta(const real_t& r) const -> real_t {
      return SQR(r) - TWO * r + SQR(a);
    }

    Inline auto Sigma(const real_t& r) const -> real_t {
      return SQR(r) + SQR(a) * SQR(math::cos(th0));
    }

    Inline auto A(const real_t& r) const -> real_t {
      return SQR(SQR(r) + SQR(a)) - SQR(a) * Delta(r) * SQR(math::sin(th0));
    }

    Inline auto omega(const real_t& r) const -> real_t {
      return TWO * a * r / A(r);
    }




  public:
    static constexpr const char*       Label { "boyer_lindq_tp" };
    static constexpr Dimension         PrtlDim { D };
    static constexpr ntt::Coord::type  CoordType { ntt::Coord::Bltp };
    static constexpr ntt::Metric::type MetricType { ntt::Metric::BoyerLindqTP };
    using MetricBase<D>::x1_min;
    using MetricBase<D>::x1_max;
    using MetricBase<D>::nx1;
    using MetricBase<D>::set_dxMin;

    BoyerLindqTP(std::vector<std::size_t>             res,
               boundaries_t<real_t>                 ext,
               const std::map<std::string, real_t>& params)
      : MetricBase<D> { res, ext }
      , a { params.at("a") }
      , psi0 { params.at("psi0") }
      , th0 { params.at("theta0") }
      , rg_ { ONE }
      , rh { ONE + math::sqrt(ONE - SQR(a)) }
      , rh_m { ONE - math::sqrt(ONE - SQR(a)) }
      , psi { psi0 * (1 - math::cos(th0)) }
      , bt { -HALF * psi0 * a * math::sin(th0) * math::cos(th0) / Sigma(rh) }
      , Omega { params.at("Omega") * a / (SQR(a) + SQR(rh)) }
      , dpsi_dth { psi0 * math::sin(th0) }
      , dbt_dth { -HALF * psi0 * a * (SQR(a * math::cos(th0)) + SQR(rh) * math::cos(TWO * th0)) / SQR(Sigma(rh)) }
      , eta_min { r2eta(x1_min) }
      , eta_max { r2eta(x1_max) }
      , d_eta { (eta_max - eta_min) / nx1 }
      , d_eta_inv { ONE / d_eta }{
      set_dxMin(find_dxMin());
    }

    ~BoyerLindqTP() = default;

    [[nodiscard]]
    Inline auto spin() const -> real_t {
      return a;
    }

    [[nodiscard]]   
    Inline auto rhorizon() const -> real_t {
      return rh;
    }

    [[nodiscard]]   
    Inline auto rhorizon_minus() const -> real_t {
      return rh_m;
    }

    [[nodiscard]]
    Inline auto rg() const -> real_t {
      return rg_;
    }
    
    [[nodiscard]]
    Inline auto OmegaF() const -> real_t {
      return Omega;
    }


    /**
     * lapse function
     * @param x coordinate array in code units
     */
    Inline auto alpha(const coord_t<D>& xi) const -> real_t {
      const real_t r_  { eta2r(xi[0] * d_eta + eta_min) };
      return math::sqrt(Sigma(r_) * Delta(r_) / A(r_));
    }

    /**
     * shift vector, only third covariant component is non-zero
     * @param x coordinate array in code units
     */

    Inline auto beta(const coord_t<D>& xi) const -> real_t {
      return math::sqrt(h<3, 3>(xi)) * omega(eta2r(xi[0] * d_eta + eta_min));
    }

    Inline auto beta3(const coord_t<D>& xi) const -> real_t {
      return -omega(eta2r(xi[0] * d_eta + eta_min));
    }

    /**
     * @brief Compute helper functions f0,f1,f2 in 1D-GRPIC
     */
    Inline auto f2(const coord_t<D>& xi) const -> real_t {
      const real_t r_  { eta2r(xi[0] * d_eta + eta_min) };
      return SQR(d_eta) * Sigma(r_) * (Delta(r_) + 
             A(r_) * SQR(bt / dpsi_dth ) 
	     );
    }

    Inline auto f1(const coord_t<D>& xi) const -> real_t {
      const real_t r_  { eta2r(xi[0] * d_eta + eta_min) };
      return d_eta * A(r_) * bt * (Omega + beta3(xi)) / psi0;
    }

    Inline auto f0(const coord_t<D>& xi) const -> real_t {
      return h_<3, 3>(xi) * SQR(Omega + beta3(xi));
    }

    /**
     * @brief force-free charge densities and currents
     */

    Inline auto rho_ff(const coord_t<D>& xi) const -> real_t {
      const real_t r_  { eta2r(xi[0] * d_eta + eta_min) };
      return psi0 / sqrt_det_h(xi) * math::sin(TWO * th0) *
             ((omega(r_) - Omega) * (ONE + SQR(a * math::sin(th0)) / Sigma(r_)) / SQR(alpha(xi)) +
               SQR(a * math::sin(th0)) * Omega / Sigma(r_));
    }

    Inline auto J_ff() const -> real_t {
      return -d_eta_inv * HALF * psi0 * a * 
            (ONE - TWO * SQR(rh * math::sin(th0)) / Sigma(rh))
            / Sigma(rh);
    }


     
    /**
     * minimum effective cell size for a given metric (in physical units)
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
        real_t min_dx { -ONE };
        for (int i { 0 }; i < nx1; ++i){
            real_t i_ { static_cast<real_t>(i) + HALF };
            coord_t<Dim::_1D> xi { i_ };
            real_t dx = ONE / math::sqrt(h<1, 1>(xi));
            if ((min_dx > dx) || (min_dx < 0.0)) {
            min_dx = dx;
            }
        }
        return min_dx;
    }

    /**
     * metric component with lower indices: h_ij
     * @param x coordinate array in code units
     */
    template <idx_t i, idx_t j>
    Inline auto h_(const coord_t<D>& x) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(j > 0 && j <= 3, "Invalid index j");

      const real_t r_  { eta2r(x[0] * d_eta + eta_min) };
      if constexpr (i == 1 && j == 1) {
        // h_11
        return SQR(d_eta) * Sigma(r_) * Delta(r_);
      } else if constexpr (i == 2 && j == 2) {
        // h_22
        return Sigma(r_) / SQR(dpsi_dth) ;
      } else if constexpr (i == 3 && j == 3) {
        // h_33
        return A(r_) * SQR(math::sin(th0)) / Sigma(r_);
      }else {
        return ZERO;
      }
    }

    /**
     * metric component with upper indices: h^ij
     * @param x coordinate array in code units
     */
    template <idx_t i, idx_t j>
    Inline auto h(const coord_t<D>& x) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(j > 0 && j <= 3, "Invalid index j");

      if constexpr (i == j) {
        return ONE / h_<i, j>(x);
      }else {
        return ZERO;
      }
    }

     /**
     * sqrt(det(h_ij))
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      return math::sqrt(h_<1, 1>(x) * h_<2, 2>(x) * h_<3, 3>(x));
    }

    /**
     * coordinate conversions
     */
    template <Crd in, Crd out>
    Inline void convert(const coord_t<D>& x_in, coord_t<D>& x_out) const {
      static_assert(in != out, "Invalid coordinate conversion");
      static_assert((in != Crd::XYZ) &&
                      (out != Crd::XYZ),
                    "Invalid coordinate conversion: XYZ not allowed in GR");
      if constexpr (in == Crd::Cd && (out == Crd::Sph || out == Crd::Ph)){
        // code -> sph/phys
        x_out[0] = eta2r(x_in[0] * d_eta + eta_min);
      }else if constexpr ((in == Crd::Sph || in == Crd::Ph) && out == Crd::Cd){
        //sph/phys -> code
        x_out[0] = (r2eta(x_in[0]) - eta_min) / d_eta;
      }
    }


    /**
     * component wise vector transformations
     */
    template <idx_t i, Idx in, Idx out>
    Inline auto transform(const coord_t<D>& x,
                          const real_t& v_in) const -> real_t {
      static_assert(in != out, "Invalid vector transformation");
      static_assert(in != Idx::XYZ && out != Idx::XYZ,
                    "Invalid vector transformation: XYZ not allowed in GR");
      static_assert(i > 0 && i <= 3, "Invalid index i");
      if constexpr ((in == Idx::T && out == Idx::Sph) ||
                    (in == Idx::Sph && out == Idx::T)) {
        // tetrad <-> sph
        return v_in;
      } else if constexpr (((in == Idx::T || in == Idx::Sph) && out == Idx::U) ||
                           (in == Idx::D && (out == Idx::T || out == Idx::Sph))){
        // tetrad/sph -> cntrv ; cov -> tetrad/sph
        if constexpr (i == 1){
          return v_in / math::sqrt(h_<i, i>(x)) / Delta(eta2r(x[0] * d_eta + eta_min));
        }else if constexpr (i == 2){
          return v_in / math::sqrt(h_<i, i>(x)) * dpsi_dth;
        }else{
          return v_in / math::sqrt(h_<i, i>(x));
        }
      } else if constexpr ((in == Idx::U && (out == Idx::T || out == Idx::Sph)) ||
                            ((in == Idx::T || in == Idx::Sph) && out == Idx::D)){
        // cntrv -> tetrad/sph ; tetrad/sph -> cov
        if constexpr (i == 1){
          return v_in * math::sqrt(h_<i, i>(x)) * Delta(eta2r(x[0] * d_eta + eta_min));
        }else if constexpr (i == 2){
          return v_in * math::sqrt(h_<i, i>(x)) / dpsi_dth;
        }else{
          return v_in * math::sqrt(h_<i, i>(x));
        }
      } else if constexpr (in == Idx::U && out == Idx::D) {
        // cntrv -> cov
        return v_in * h_<i, i>(x);
      } else if constexpr (in == Idx::D && out == Idx::U) {
        // cov -> cntrv
        return v_in * h<i, i>(x);
      } else if constexpr ((in == Idx::U && out == Idx::PU) ||
                           (in == Idx::PD && out == Idx::D)) {
        // cntrv -> phys cntrv || phys cov -> cov
        if constexpr (i == 1){
          return v_in * Delta(eta2r(x[0] * d_eta + eta_min)) * d_eta ;
        }else if constexpr (i == 2){
          return v_in / dpsi_dth;
        }else{
          return v_in;
        }
      } else if constexpr ((in == Idx::PU && out == Idx::U) ||
                           (in == Idx::D && out == Idx::PD)) {
        // phys cntrv -> cntrv || cov -> phys cov
        if constexpr (i == 1){
          return v_in * d_eta_inv / Delta(eta2r(x[0] * d_eta + eta_min));
        }else if constexpr (i == 2){
          return v_in * dpsi_dth;
        }else{
          return v_in;
        }
      } else {
        raise::KernelError(HERE, "Invalid transformation");
      }
    }

    /**
     * full vector transformations
     */
    template <Idx in, Idx out>
    Inline void transform(const coord_t<D>&      xi,
                          const vec_t<Dim::_3D>& v_in,
                          vec_t<Dim::_3D>&       v_out) const {
      static_assert(in != out, "Invalid vector transformation");
      if constexpr (in != Idx::XYZ && out != Idx::XYZ) {
        v_out[0] = transform<1, in, out>(xi, v_in[0]);
        v_out[1] = transform<2, in, out>(xi, v_in[1]);
        v_out[2] = transform<3, in, out>(xi, v_in[2]);
      } else {
        raise::KernelError(HERE, "Invalid vector transformation");
      }
    }
    
    Inline auto r2eta(const real_t& r) const -> real_t{
      return math::log((r - rh) / (r - rh_m)) / (rh - rh_m);
    }

    Inline auto eta2r(const real_t& eta) const -> real_t{
      return rh_m - TWO * math::sqrt(ONE - SQR(a)) / math::expm1(eta * TWO * math::sqrt(ONE - SQR(a)));
    }
  };

  

}


#endif // METRICS_BOYER_LINDQ_TP_H
