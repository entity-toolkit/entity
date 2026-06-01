/**
 * @file metrics/spherical.h
 * @brief Flat space-time spherical metric class diag(-1, 1, r^2, r^2, sin(th)^2)
 * @implements
 *   - metric::Spherical<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 * !TODO
 *   - 3D version of find_dxMin
 */

 #ifndef METRICS_SPHERICAL_H
 #define METRICS_SPHERICAL_H
 
 #include "enums.h"
 #include "global.h"
 
 #include "arch/kokkos_aliases.h"
 #include "utils/numeric.h"
 
 #include "metrics/metric_base.h"
 
 #include <map>
 #include <string>
 #include <vector>
 
 namespace metric {
  struct Patch {
    enum type : uint8_t {
      INVALID = 0,
      I       = 1,
      II      = 2,
      III     = 3,
      IV      = 4,
      V       = 5,
      VI      = 6,
    };
  }
   template <Dimension D>
   class CubedSphere : public MetricBase<D> {
     static_assert(D != Dim::_1D, "1D cubed sphere not available");
     static_assert(D != Dim::_2D, "2D cubed sphere not fully implemented");
 
     const real_t dr, dtheta, dphi;
     const real_t dr_inv, dtheta_inv, dphi_inv;
     const bool   small_angle;
 
   public:
     static constexpr const char*       Label { "cubed sphere" };
     static constexpr Dimension         PrtlDim { Dim::_3D };
     static constexpr ntt::Metric::type MetricType { ntt::Metric::CubedSphere };
     static constexpr ntt::Coord::type  CoordType { ntt::Coord::Cbd };
     using MetricBase<D>::x1_min;
     using MetricBase<D>::x1_max;
     using MetricBase<D>::x2_min;
     using MetricBase<D>::x2_max;
     using MetricBase<D>::x3_min;
     using MetricBase<D>::x3_max;
     using MetricBase<D>::nx1;
     using MetricBase<D>::nx2;
     using MetricBase<D>::nx3;
     using MetricBase<D>::set_dxMin;
 
     CubedSphere(const std::vector<ncells_t>& res,
               const boundaries_t<real_t>&  ext,
               const std::map<std::string, real_t>& = {})
       : MetricBase<D> { res, ext }
       , dr((x1_max - x1_min) / nx1)
       , dtheta((x2_max - x2_min) / nx2)
       , dphi((x3_max - x3_min) / nx3)
       , dr_inv { ONE / dr }
       , dtheta_inv { ONE / dtheta }
       , dphi_inv { ONE / dphi }
       , small_angle { HALF * dtheta < constant::SMALL_ANGLE } {
       set_dxMin(find_dxMin());
     }
 
     ~CubedSphere() = default;
     
     /**
      * minimum effective cell size for a given metric (in physical units)
      */
     [[nodiscard]]
     auto find_dxMin() const -> real_t override {
       // for 2D
       auto dx1 { dr };
       auto dx2 { x1_min * dtheta };
       return ONE / math::sqrt(ONE / SQR(dx1) + ONE / SQR(dx2));
     }
 
     /**
      * total volume of the region described by the metric (in physical units)
      */
     [[nodiscard]]
     auto totVolume() const -> real_t override {
       if constexpr (D == Dim::_1D) {
         raise::Error("1D spherical metric not applicable", HERE);
       } else if constexpr (D == Dim::_2D) {
         return (SQR(x1_max) - SQR(x1_min)) * (x2_max - x2_min);
       } else {
         return (SQR(x1_max) - SQR(x1_min)) * (x2_max - x2_min) * (x3_max - x3_min);
       }
     }
     /**
      * auxilary variables X,Y,C,D required to compute metric and coordinate transformations 
      * @param x - coordinate component in physical units
      */
     Inline auto get_X(const coord_t<D>& x) const -> real_t {
        return math::tan(x[1] * dxi + x2_min);
     }

     Inline auto get_Y(const coord_t<D>& x) const -> real_t {
        return math::tan(x[2] * deta + x3_min);
     }
     Inline auto get_C(const coord_t<D>& x) const ->real_t{
        const auto X_ = get_X(x);
        return math::sqrt(ONE + X_ * X_);
     }
     Inline auto get_D(const coord_t x) const -> real_t{
        const auto Y_ = get_Y(x);
        return math::sqrt(ONE + Y_ * Y_);
     }
    /**
      * auxilary variables delta required to compute metric and coordinate transformations 
      * @param x - coordinate array in physical units
      */
     Inline auto get_delta(const coord_t<D>& x) const -> real_t{
        if constexpr(D==Dim::_2D){
            const auto X_ = get_X(x);
            const auto Y_ = get_Y(x);
            return ONE + X_ * X_ + Y_ * Y_;
        }else if constexpr(D==Dim::_3D){
            const auto X_ = get_X(x);
            const auto Y_ = get_Y(x);
            return math::sqrt(ONE + X_ * X_+ Y_ * Y_);
        }
        
     }
     /**
      * metric component with lower indices: h_ij
      * @param x coordinate array in code units
      */
     template <idx_t i, idx_t j>
     Inline auto h_(const coord_t<D>& x) const -> real_t {
       static_assert(i > 0 && i <= 3, "Invalid index i");
       static_assert(j > 0 && j <= 3, "Invalid index j");

       if constexpr (i == 1 && j == 1) {
         return ONE;
       } else if constexpr (i == 2 && j == 2) {
         const auto r = x[0] * dr + x1_min;
         const auto C_ = get_C(x);
         const auto D_ = get_D(x);
         const auto delta_ = get_delta(x);
         return r * r * math::pow(C_, 4) * D_ * D_ / math::pow(delta, 4) ;
       } else if constexpr (i == 3 && j == 3) {
         const auto r = x[0] * dr + x1_min; 
         const auto C_ = get_C(x);
         const auto D_ = get_D(x);
         const auto delta_ = get_delta(x);
         return r * r * math::pow(D_, 4) * C_ * C_ / math::pow(delta, 4) ;
         } else if constexpr((i == 2 && j == 3) || (i == 3 && j == 2)) {
          const auto r = x[0] * dr + x1_min;
          const auto C_ = get_C(x);
          const auto D_ = get_D(x);
          const auto delta_ = get_delta(x);
         return - r * r * X_ * Y_ * C_ * C_ * D_ * D_ / math::pow(delta, 4);
       } else {
        return ZERO;
       }
     }
 
     /**
      * sqrt(h_ij)
      * @param x coordinate array in code units
      */
     template <idx_t i, idx_t j>
     Inline auto sqrt_h_(const coord_t<D>& x) const -> real_t {
       static_assert(i > 0 && i <= 3, "Invalid index i");
       static_assert(j > 0 && j <= 3, "Invalid index j");
       if constexpr (i == 1 && j == 1) {
         return dr;
       } else if constexpr (i == 2 && j == 2) {
         return dtheta * (x[0] * dr + x1_min);
       } else if constexpr (i == 3 && j == 3) {
         if constexpr (D == Dim::_2D) {
           return (x[0] * dr + x1_min) * (math::sin(x[1] * dtheta + x2_min));
         } else {
           return dphi * (x[0] * dr + x1_min) * (math::sin(x[1] * dtheta + x2_min));
         }
       } else {
         return ZERO;
       }
     }
 
     /**
      * sqrt(det(h_ij))
      * @param x coordinate array in code units
      */
     Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
        const auto r = x[0] * dr + x1_min;
        const auto C_ = get_C(x);
        const auto D_ = get_D(x);
        const auto delta_ = get_delta(x);
        return r * r * C_ * C_ * D_ * D_ / pow(delta_,3);
     }
 
     /**
      * component-wise coordinate conversions
      */
     template <idx_t i, Crd in, Crd out>
     Inline auto convert(real_t x_in) const -> real_t {
       static_assert(in != out, "Invalid coordinate conversion");
       static_assert(i > 0 && i <= 3, "Invalid index i");
       static_assert((in == Crd::Cd && (out == Crd::Ph || out == Crd::Cbd)) ||
                       ((in == in == Crd::Ph || out == Crd::Cbd) && out == Crd::Cd),
                     "Invalid coordinate conversion");
       if constexpr (in == Crd::Cd && (out == Crd::Cbd || out == Crd::Ph)) {
         // code -> cbd/phys
         if constexpr (i == 1) {
           return x_in * dr + x1_min;
         } else if constexpr (i == 2) {
           return x_in * dxi + x2_min;
         } else {
             return x_in * deta + x3_min;
           }
       } else {
         // cbd/phys -> code
         if constexpr (i == 1) {
           return (x_in - x1_min) * dr_inv;
         } else if constexpr (i == 2) {
           return (x_in - x2_min) * dxi_inv;
         } else {
             return (x_in - x3_min) * deta_inv;
         }
       }
     }
 
     /**
      * full coordinate conversions
      */
     template <Crd in, Crd out>
     Inline void convert(const coord_t<D>& x_in, coord_t<D>& x_out) const {
       static_assert(in != out, "Invalid coordinate conversion");
       static_assert(in != Crd::XYZ && out != Crd::XYZ,
                     "Invalid coordinate conversion: use convert_xyz");
       static_assert(in != Crd::Sph && out != Crd::Sph,
                      "Invalid coordinate conversion: use convert_sph");
       // code <-> cbd/phys
       x_out[0] = convert<1, in, out>(x_in[0]);
       x_out[1] = convert<2, in, out>(x_in[1]);
       x_out[2] = convert<3, in, out>(x_in[2]);
     }
 
     /**
      * full coordinate conversion to/from cartesian FOR CUBED SPHERE
      */
     template <Crd in, Crd out, Patch p>
     Inline void convert_xyz_cbd(const coord_t<PrtlDim>& x_in,
                             coord_t<PrtlDim>&       x_out) const {
       static_assert((in == Crd::Cd && out == Crd::XYZ) ||
                       (in == Crd::XYZ && out == Crd::Cd),
                     "Invalid coordinate conversion");
       if (in == Crd::Cd && out == Crd::XYZ) {
         // code -> cart
         real_t r = convert<1, Crd::Cd, Crd::Cbd>(x_in[0]);
         if constexpr(p == Patch::I){
          real_t X_ = get_X(x_in);
          real_t Y_ = get_Y(x_in);
          real_t delta_ = get_delta(x_in);
          x_out[0] = r / delta_;
          x_out[1] = r * X_ / delta_;
          x_out[2] = r * Y_ / delta_;
         } else if (constexpr(p == Patch::II)){
          real_t X_ = get_X(x_in);
          real_t Y_ = get_Y(x_in);
          real_t delta_ = get_delta(x_in);
          x_out[0] = r / delta_;
          x_out[1] = r * X_ / delta_;
          x_out[2] = r * Y_ / delta_;
         } else if constexpr(p == Patch::III){
          // north pole patch. 
          // Rotated -pi/2 clockwise relative to patch I
          real_t X_ = -get_Y(x_in);
          real_t Y_ = get_X(x_in);
          real_t delta_ = get_delta(x_in);
          x_out[0] = - r * X_ /delta_;
          x_out[1] = - r * Y_ / delta_;
          x_out[2] = r / delta_; 
         } else if constexpr(p == Patch::IV){
          // rotated by +pi/2 clockwise relative to patch I 
          real_t X_ =  get_Y(x_in);
          real_t Y_ = -get_X(x_in);
          real_t delta_ = ;
          x_out[0] = - r / delta_;
          x_out[1] = - r * Y_ / delta_;
          x_out[2] = r * X_ / delta_;
         } else if constexpr(p == Patch::V){
          // rotated by +pi/2 clockwise relative to patch I
          real_t X_ = get_Y(x_in);
          real_t Y_ = -get_X(x_in);
          real_t delta_ = get_delta(x_in);
          x_out[0] =   r * Y_ / delta_;
          x_out[1] = - r / delta_;
          x_out[2] = - r * X_ / delta_;
         } else if constexpr(p == Patch::VI){
          //south pole patch.
          real_t X_ = get_X(x_in);
          real_t Y_ = get_Y(x_in);
          real_t delta_ = get_delta(x_in);
          x_out[0] = r * Y_ / delta_;
          x_out[1] = r * X_ / delta_;
          x_out[2] = - r / delta_;
         } else{
          raise::KernelError(HERE, "Invalid cubed sphere patch");
         }
         x_out[0] = x_Sph[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]);
         x_out[1] = x_Sph[0] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]);
         x_out[2] = x_Sph[0] * math::cos(x_Sph[1]);
       } else {
         // cart -> code
         coord_t<PrtlDim> x_Cbd { ZERO };
         x_Cbd[0] = math::sqrt(SQR(x_in[0]) + SQR(x_in[1]) + SQR(x_in[2]));
         x_out[0] = convert<1, Crd::Cbd, Crd::Cd>(x_Cbd[0]);
        if constexpr(p == Patch::I){ 
          x_Cbd[1] = math::atan(x_in[1] / x_in[0]);
          x_Cbd[2] = math::atan(x_in[2] / x_in[0]);
          
          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(x_Cbd[1]);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(x_Cbd[2]);
        } else if constexpr(p == Patch::II){
          x_Cbd[1] = math::atan(- x_in[0] / x_in[1]);
          x_Cbd[2] = math::atan(x_in[2] / x_in[1]);
          
          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(x_Cbd[1]);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(x_Cbd[2]);
        } else if constexpr(p == Patch::III){
          // north pole patch. 
          // Rotated -pi/2 clockwise relative to patch I
          x_Cbd[1] = math::atan( - x_in[1] / x_in[2]);
          x_Cbd[2] = - math::atan( - x_in[0] / x_in[2]);

          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(x_Cbd[1]);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(x_Cbd[2]);
        } else if constexpr(p == Patch::IV){
          // rotated by +pi/2 clockwise relative to patch I 
          x_Cbd[1] = - math::atan(x_in[1] / x_in[0]);
          x_Cbd[2] = math::atan(x_in[2] / x_in[1]); 

          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(x_Cbd[1]);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(x_Cbd[2]);
        } else if constexpr(p == Patch::V){
          // rotated by +pi/2 clockwise relative to patch I
          x_Cbd[1] = - math::atan(- x_in[0] / x_in[1]);
          x_Cbd[2] = math::atan( x_in[2] / x_in[1] );

          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(x_Cbd[1]);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(x_Cbd[2]);
        } else if constexpr(p == Patch::VI){
          //south pole patch. 
          x_Cbd[1] = math::atan( - x_in[1] / x_in[2]);
          x_Cbd[2] = math::atan( - x_in[0] / x_in[2]);         
          
          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(x_Cbd[1]);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(x_Cbd[2]);
        }else{
          raise::KernelError(HERE, "Invalid cubed sphere patch");
        }
       }
     }

     template <Crd in, Crd out>
     Inline void convert_xyz(const coord_t<PrtlDim>& x_in,
                             coord_t<PrtlDim>&       x_out) const {
       static_assert((in == Crd::Cd && out == Crd::XYZ) ||
                       (in == Crd::XYZ && out == Crd::Cd),
                     "Invalid coordinate conversion");
       if (in == Crd::Cd && out == Crd::XYZ) {
         // code -> cart
         coord_t<PrtlDim> x_Sph { ZERO };
         x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(x_in[0]);
         x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(x_in[1]);
         x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(x_in[2]);
         x_out[0] = x_Sph[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]);
         x_out[1] = x_Sph[0] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]);
         x_out[2] = x_Sph[0] * math::cos(x_Sph[1]);
       } else {
         // cart -> code
         coord_t<PrtlDim> x_Sph { ZERO };
         x_Sph[0] = math::sqrt(SQR(x_in[0]) + SQR(x_in[1]) + SQR(x_in[2]));
         x_Sph[1] = static_cast<real_t>(constant::HALF_PI) -
                    math::atan2(x_in[2], math::sqrt(SQR(x_in[0]) + SQR(x_in[1])));
         x_Sph[2] = static_cast<real_t>(constant::PI) -
                    math::atan2(x_in[1], -x_in[0]);
         x_out[0] = convert<1, Crd::Sph, Crd::Cd>(x_Sph[0]);
         x_out[1] = convert<2, Crd::Sph, Crd::Cd>(x_Sph[1]);
         x_out[2] = convert<3, Crd::Sph, Crd::Cd>(x_Sph[2]);
       }
     }

     /**
      * full coordinate conversion to/from spherical FOR CUBED SPHERE
      */
      template <Crd in, Crd out, Patch p>
      Inline void convert_sph_cbd(const coord_t<PrtlDim>& x_in,
                              coord_t<PrtlDim>&       x_out) const {
        static_assert((in == Crd::Cd && out == Crd::Sph) ||
                        (in == Crd::Sph && out == Crd::Cd),
                      "Invalid coordinate conversion");
        if (in == Crd::Cd && out == Crd::Sph) {
          // code -> sph
          coord_t<PrtlDim> x_Cbd { ZERO };
          x_Cbd[0] = convert<1, Crd::Cd, Crd::Cbd>(x_in[0]);
          x_Cbd[1] = convert<2, Crd::Cd, Crd::Cbd>(x_in[1]);
          x_Cbd[2] = convert<3, Crd::Cd, Crd::Cbd>(x_in[2]);
          if constexpr(p == Patch::I){
            const auto X_ = get_X(x_in);
            const auto Y_ = get_Y(x_in);
            const auto C_ = get_C(x_in);
            x_out[0] = convert<1, Crd::Cd, Crd::Cbd>(x_in[0]);
            x_out[1] = HALF_PI - math::atan(Y_ / C_);
            x_out[2] = math::atan2(X_ / C_, ONE / C_);
          } else if (constexpr(p == Patch::II)){
            const auto X_ = get_X(x_in);
            const auto Y_ = get_Y(x_in);
            const auto C_ = get_C(x_in);
            x_out[0] = convert<1, Crd::Cd, Crd::Cbd>(x_in[0]);
            x_out[1] = HALF_PI - math::atan(Y_ / C_);
            x_out[2] = math::atan2(ONE / C_, - X_ / C_);
          } else if constexpr(p == Patch::III){
            // north pole patch. 
            // Rotated -pi/2 clockwise relative to patch I
            const auto X_ = -get_Y(x_in);
            const auto Y_ = get_X(x_in);
            const auto del = math::sqrt(X_ * X_ + Y_ * Y_);
            x_out[0] = convert<1, Crd::Cd, Crd::Cbd>(x_in[0]);
            if (del == ZERO){
              x_out[1] = ZERO;
              x_out[2] = ZERO;
            } else {
              x_out[1] = HALF_PI - math::atan( ONE / del);
              x_out[2] = math::atan(X_ / del, -Y_ / del);
            }
          } else if constexpr(p == Patch::IV){
            // rotated by +pi/2 clockwise relative to patch I 
            const auto X_ =  get_Y(x_in);
            const auto Y_ = -get_X(x_in);
            x_out[0] = convert<1, Crd::Cd, Crd::Cbd>(x_in[0]);
            x_out[1] = HALF_PI - math::atan(Y_ / math::sqrt(ONE + X_ * X_));
            x_out[2] = math::atan2( - X_ / math::sqrt(ONE + X_ * X_), - ONE / math::sqrt(ONE + X_ * X_));
          } else if constexpr(p == Patch::V){
            // rotated by +pi/2 clockwise relative to patch I
            const auto X_ = get_Y(x_in);
            const auto Y_ = -get_X(x_in);
            x_out[0] = convert<1, Crd::Cd, Crd::Cbd>(x_in[0]);
            x_out[1] = HALF_PI - math::atan(Y_ / math::sqrt(ONE + X_ * X_));
            x_out[2] = math::atan2( - X_ / math::sqrt(ONE + X_ * X_), - ONE / math::sqrt(ONE + X_ * X_));
          } else if constexpr(p == Patch::VI){
            //south pole patch. 
            const auto X_ = get_X(x_in);
            const auto Y_ = get_Y(x_in);
            const auto del = math::sqrt(X_ * X_ + Y_ * Y_);
            x_out[0] = convert<1, Crd::Cd, Crd::Cbd>(x_in[0]);
            if (del == ZERO){
              x_out[1] = PI;
              x_out[2] = ZERO;  
            } else {
              x_out[1] = HALF_PI - math::atan(- ONE / del);
              x_out[2] = math::atan2(X_ / del, Y_ / del);
            }
          } else{
           raise::KernelError(HERE, "Invalid cubed sphere patch");
          }
          x_out[0] = x_Sph[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]);
          x_out[1] = x_Sph[0] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]);
          x_out[2] = x_Sph[0] * math::cos(x_Sph[1]);
        } else {
          // sph -> code
          coord_t<PrtlDim> x_Cbd { ZERO };
         if constexpr(p == Patch::I){
          real_t eta = math::atan(ONE / (math::tan(x_in[1]) * math::cos(x_in[2])));
          x_out[0] = convert<1, Crd::Cbd, Crd::Cd>(x_in[0]);
          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(x_in[2]); //xi = phi for patch I
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(eta);

         } else if constexpr(p == Patch::II){
          real_t xi = math::atan(- ONE / math:tan(x[2]));
          real_t eta = math::atan( ONE / (math::tan(x[1]) * math::sin(x[2])));
          x_out[0] = convert<1, Crd::Cbd, Crd::Cd>(x_in[0]);
          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(xi);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(eta);

         } else if constexpr(p == Patch::III){
          //north pole patch
          // Rotated -pi/2 clockwise relative to patch I
          real_t xi = math::atan(- math::tan(x_in[1]) * math::cos(x_in[2]));
          real_t eta = math::atan(math::tan(x_in[1]) * math::sin(x_in[2]));
          x_out[0] = convert<1, Crd::Cbd, Crd::Cd>(x_in[0]);
          x_out[1] = convert<2, Crd::Cbd, Crd::Cd>(xi);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(eta);

         } else if constexpr(p == Patch::IV){
          // rotated by +pi/2 clockwise relative to patch I 
          real_t xi = - math::atan(- ONE / (math::tan(x_in[1]) * math::cos(x_in[2])));
          x_out[0] = convert<1, Crd::Cbd, Crd::Cd>(x_in[0]);
          x_out[1] = convert<1, Crd::Cbd, Crd::Cd>(xi);
          x_out[2] = convert<3, Crd::Cbd, Crd::Cd>(x_in[2]); //eta = phi for patch IV
         } else if constexpr(p == Patch::V){
          // rotated by +pi/2 clockwise relative to patch I
          real_t xi = - math::atan(- ONE / (math::tan(x_in[1]) * math::cos(x_in[2])));
          real_t eta = math::atan(- ONE / math::tan(x_in[2]));
          x_out[0] = convert<1, Crd::Cbd, Crd::Cd>(x_in[0]);
          x_out[1] = convert<1, Crd::Cbd, Crd::Cd>(xi);
          x_out[2] = convert<1, Crd::Cbd, Crd::Cd>(eta);

         } else if constexpr(p == Patch::VI){
          //south pole patch
          real_t xi  = math::atan(- math::tan(x_in[1]) * math::sin(x_in[2]));
          real_t eta = math::atan(- math::tan(x_in[1]) * math::cos(x_in[2]));
          x_out[0] = convert<1, Crd::Cbd, Crd::Cd>(x_in[0]);
          x_out[1] = convert<1, Crd::Cbd, Crd::Cd>(xi);
          x_out[2] = convert<1, Crd::Cbd, Crd::Cd>(eta);

         } else{
           raise::KernelError(HERE, "Invalid cubed sphere patch");
         }
        }
      }

     /**
      * component-wise vector transformations
      * @note tetrad/sph <-> cntrv <-> cov
      */
     template <idx_t i, Idx in, Idx out>
     Inline auto transform(const coord_t<D>& xi, real_t v_in) const -> real_t {
       static_assert(i > 0 && i <= 3, "Invalid index i");
       static_assert(in != out, "Invalid vector transformation");
       if constexpr ((in == Idx::T && out == Idx::Sph) ||
                     (in == Idx::Sph && out == Idx::T)) {
         // tetrad <-> sph
         return v_in;
       } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::U) {
         // tetrad/sph -> cntrv
         return v_in / sqrt_h_<i, i>(xi);
       } else if constexpr (in == Idx::U && (out == Idx::T || out == Idx::Sph)) {
         // cntrv -> tetrad/sph
         return v_in * sqrt_h_<i, i>(xi);
       } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::D) {
         // tetrad/sph -> cov
         return v_in * sqrt_h_<i, i>(xi);
       } else if constexpr (in == Idx::D && (out == Idx::T || out == Idx::Sph)) {
         // cov -> tetrad/sph
         return v_in / sqrt_h_<i, i>(xi);
       } else if constexpr (in == Idx::U && out == Idx::D) {
         // cntrv -> cov
         return v_in * h_<i, i>(xi);
       } else if constexpr (in == Idx::D && out == Idx::U) {
         // cov -> cntrv
         return v_in / h_<i, i>(xi);
       } else if constexpr ((in == Idx::U && out == Idx::PU) ||
                            (in == Idx::PD && out == Idx::D)) {
         // cntrv -> phys cntrv || phys cov -> cov
         if constexpr (i == 1) {
           return v_in * dr;
         } else if constexpr (i == 2) {
           return v_in * dtheta;
         } else if constexpr (D == Dim::_2D) {
           return v_in;
         } else {
           return v_in * dphi;
         }
       } else if constexpr ((in == Idx::PU && out == Idx::U) ||
                            (in == Idx::D && out == Idx::PD)) {
         // phys cntrv -> cntrv || cov -> phys cov
         if constexpr (i == 1) {
           return v_in * dr_inv;
         } else if constexpr (i == 2) {
           return v_in * dtheta_inv;
         } else if constexpr (D == Dim::_2D) {
           return v_in;
         } else {
           return v_in * dphi_inv;
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
 
     /**
      * full vector transformation to/from cartesian
      */
     template <Idx in, Idx out>
     Inline void transform_xyz(const coord_t<PrtlDim>& xi,
                               const vec_t<Dim::_3D>&  v_in,
                               vec_t<Dim::_3D>&        v_out) const {
       static_assert(in != out, "Invalid vector transformation");
       static_assert(in == Idx::XYZ || out == Idx::XYZ,
                     "Invalid vector transformation");
       if constexpr (in == Idx::T && out == Idx::XYZ) {
         // tetrad -> cart
         coord_t<PrtlDim> x_Sph { ZERO };
         x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(xi[0]);
         x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(xi[1]);
         x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(xi[2]);
         v_out[0] = v_in[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]) +
                    v_in[1] * math::cos(x_Sph[1]) * math::cos(x_Sph[2]) -
                    v_in[2] * math::sin(x_Sph[2]);
         v_out[1] = v_in[0] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]) +
                    v_in[1] * math::cos(x_Sph[1]) * math::sin(x_Sph[2]) +
                    v_in[2] * math::cos(x_Sph[2]);
         v_out[2] = v_in[0] * math::cos(x_Sph[1]) - v_in[1] * math::sin(x_Sph[1]);
       } else if constexpr (in == Idx::XYZ && out == Idx::T) {
         // cart -> tetrad
         coord_t<PrtlDim> x_Sph { ZERO };
         x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(xi[0]);
         x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(xi[1]);
         x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(xi[2]);
         v_out[0] = v_in[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]) +
                    v_in[1] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]) +
                    v_in[2] * math::cos(x_Sph[1]);
         v_out[1] = v_in[0] * math::cos(x_Sph[1]) * math::cos(x_Sph[2]) +
                    v_in[1] * math::cos(x_Sph[1]) * math::sin(x_Sph[2]) -
                    v_in[2] * math::sin(x_Sph[1]);
         v_out[2] = -v_in[0] * math::sin(x_Sph[2]) + v_in[1] * math::cos(x_Sph[2]);
       } else if (in == Idx::XYZ) {
         // cart -> cov/cntrv
         vec_t<Dim::_3D> v_Tetrad { ZERO };
         transform_xyz<Idx::XYZ, Idx::T>(xi, v_in, v_Tetrad);
         if constexpr (D == Dim::_2D) {
           transform<Idx::T, out>({ xi[0], xi[1] }, v_Tetrad, v_out);
         } else {
           transform<Idx::T, out>(xi, v_Tetrad, v_out);
         }
       } else if (out == Idx::XYZ) {
         // cov/cntrv -> cart
         vec_t<Dim::_3D> v_Tetrad { ZERO };
         if constexpr (D == Dim::_2D) {
           transform<in, Idx::T>({ xi[0], xi[1] }, v_in, v_Tetrad);
         } else {
           transform<in, Idx::T>(xi, v_in, v_Tetrad);
         }
         transform_xyz<Idx::T, Idx::XYZ>(xi, v_Tetrad, v_out);
       }
     }
   };
 
 } // namespace metric
 
 #endif // METRICS_SPHERICAL_H
 
