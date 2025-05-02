/**
 * @file kernels/reduced_stats.hpp
 * @brief Compute reduced field/moment quantities for stats output
 * @implements
 *   - kernel::PrtlToPhys_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_REDUCED_STATS_HPP
#define KERNELS_REDUCED_STATS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  template <SimEngine::type S, class M, StatsID::type F, unsigned short I = 0>
  class ReducedFields_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(I <= 3,
                  "I must be less than or equal to 3 for ReducedFields_kernel");
    static constexpr auto D = M::Dim;

    ndfield_t<D, 6> EM;
    ndfield_t<D, 3> J;
    const M         metric;

  public:
    ReducedFields_kernel(const ndfield_t<D, 6>& EM,
                         const ndfield_t<D, 3>& J,
                         const M&               metric)
      : EM { EM }
      , J { J }
      , metric { metric } {}

    Inline void operator()(index_t i1, real_t& buff) const {
      const auto i1_ = COORD(i1);
      if constexpr (F == StatsID::B2) {
        if constexpr (I == 1) {
          const auto b1_u = EM(i1, em::bx1);
          const auto b1_d = metric.template transform<1, Idx::U, Idx::D>({ i1_ },
                                                                         b1_u);
          buff += b1_u * b1_d;
        } else if constexpr (I == 2) {
          const auto b2_u = EM(i1, em::bx2);
          const auto b2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_ + HALF },
            b2_u);
          buff += b2_u * b2_d;
        } else {
          const auto b3_u = EM(i1, em::bx3);
          const auto b3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_ + HALF },
            b3_u);
          buff += b3_u * b3_d;
        }
      } else if constexpr (F == StatsID::E2) {
        if constexpr (I == 1) {
          const auto e1_u = EM(i1, em::ex1);
          const auto e1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_ + HALF },
            e1_u);
          buff += e1_u * e1_d;
        } else if constexpr (I == 2) {
          const auto e2_u = EM(i1, em::ex2);
          const auto e2_d = metric.template transform<2, Idx::U, Idx::D>({ i1_ },
                                                                         e2_u);
          buff += e2_u * e2_d;
        } else {
          const auto e3_u = EM(i1, em::ex3);
          const auto e3_d = metric.template transform<3, Idx::U, Idx::D>({ i1_ },
                                                                         e3_u);
          buff += e3_u * e3_d;
        }
      } else if constexpr (F == StatsID::ExB) {
        if constexpr (I == 1) {
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::ex2) + EM(i1 + 1, em::ex2)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::ex3) + EM(i1 + 1, em::ex3)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::bx2));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::bx3));
          buff += e2_t * b3_t - e3_t * b2_t;
        } else if constexpr (I == 2) {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::ex1));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::ex3) + EM(i1 + 1, em::ex3)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::bx1) + EM(i1 + 1, em::bx1)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::bx3));
          buff += e3_t * b1_t - e1_t * b3_t;
        } else {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::ex1));
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::ex2) + EM(i1 + 1, em::ex2)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF },
            HALF * (EM(i1, em::bx1) + EM(i1 + 1, em::bx1)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF },
            EM(i1, em::bx2));
          buff += e1_t * b2_t - e2_t * b1_t;
        }
      } else if constexpr (F == StatsID::JdotE) {
        vec_t<Dim::_3D> e_t { ZERO };
        vec_t<Dim::_3D> j_t { ZERO };
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF },
          { EM(i1, em::ex1),
            HALF * (EM(i1, em::ex2) + EM(i1 + 1, em::ex2)),
            HALF * (EM(i1, em::ex3) + EM(i1 + 1, em::ex3)) },
          e_t);
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF },
          { J(i1, cur::jx1),
            HALF * (J(i1, cur::jx2) + J(i1 + 1, cur::jx2)),
            HALF * (J(i1, cur::jx3) + J(i1 + 1, cur::jx3)) },
          j_t);
        buff += e_t[0] * j_t[0] + e_t[1] * j_t[1] + e_t[2] * j_t[2];
      }
    }

    Inline void operator()(index_t i1, index_t i2, real_t& buff) const {
      const auto i1_ = COORD(i1);
      const auto i2_ = COORD(i2);
      if constexpr (F == StatsID::B2) {
        if constexpr (I == 1) {
          const auto b1_u = EM(i1, i2, em::bx1);
          const auto b1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_, i2_ + HALF },
            b1_u);
          buff += b1_u * b1_d;
        } else if constexpr (I == 2) {
          const auto b2_u = EM(i1, i2, em::bx2);
          const auto b2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_ + HALF, i2_ },
            b2_u);
          buff += b2_u * b2_d;
        } else {
          const auto b3_u = EM(i1, i2, em::bx3);
          const auto b3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_ + HALF, i2_ + HALF },
            b3_u);
          buff += b3_u * b3_d;
        }
      } else if constexpr (F == StatsID::E2) {
        if constexpr (I == 1) {
          const auto e1_u = EM(i1, i2, em::ex1);
          const auto e1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_ + HALF, i2_ },
            e1_u);
          buff += e1_u * e1_d;
        } else if constexpr (I == 2) {
          const auto e2_u = EM(i1, i2, em::ex2);
          const auto e2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_, i2_ + HALF },
            e2_u);
          buff += e2_u * e2_d;
        } else {
          const auto e3_u = EM(i1, i2, em::ex3);
          const auto e3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_, i2_ },
            e3_u);
          buff += e3_u * e3_d;
        }
      } else if constexpr (F == StatsID::ExB) {
        if constexpr (I == 1) {
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::ex2) + EM(i1 + 1, i2, em::ex2)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            INV_4 * (EM(i1, i2, em::ex3) + EM(i1 + 1, i2, em::ex3) +
                     EM(i1, i2 + 1, em::ex3) + EM(i1 + 1, i2 + 1, em::ex3)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::bx2) + EM(i1, i2 + 1, em::bx2)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            EM(i1, i2, em::bx3));
          buff += e2_t * b3_t - e3_t * b2_t;
        } else if constexpr (I == 2) {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::ex1) + EM(i1, i2 + 1, em::ex1)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            INV_4 * (EM(i1, i2, em::ex3) + EM(i1 + 1, i2, em::ex3) +
                     EM(i1, i2 + 1, em::ex3) + EM(i1 + 1, i2 + 1, em::ex3)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::bx1) + EM(i1 + 1, i2, em::bx1)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            EM(i1, i2, em::bx3));
          buff += e3_t * b1_t - e1_t * b3_t;
        } else {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::ex1) + EM(i1, i2 + 1, em::ex1)));
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::ex2) + EM(i1 + 1, i2, em::ex2)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::bx1) + EM(i1 + 1, i2, em::bx1)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF },
            HALF * (EM(i1, i2, em::bx2) + EM(i1, i2 + 1, em::bx2)));
          buff += e1_t * b2_t - e2_t * b1_t;
        }
      } else if constexpr (F == StatsID::JdotE) {
        vec_t<Dim::_3D> e_t { ZERO };
        vec_t<Dim::_3D> j_t { ZERO };
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF, i2_ + HALF },
          { HALF * (EM(i1, i2, em::ex1) + EM(i1, i2 + 1, em::ex1)),
            HALF * (EM(i1, i2, em::ex2) + EM(i1 + 1, i2, em::ex2)),
            INV_4 * (EM(i1, i2, em::ex3) + EM(i1 + 1, i2, em::ex3) +
                     EM(i1, i2 + 1, em::ex3) + EM(i1 + 1, i2 + 1, em::ex3)) },
          e_t);
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF, i2_ + HALF },
          { HALF * (J(i1, i2, cur::jx1) + J(i1, i2 + 1, cur::jx1)),
            HALF * (J(i1, i2, cur::jx2) + J(i1 + 1, i2, cur::jx2)),
            INV_4 * (J(i1, i2, cur::jx3) + J(i1 + 1, i2, cur::jx3) +
                     J(i1, i2 + 1, cur::jx3) + J(i1 + 1, i2 + 1, cur::jx3)) },
          j_t);
        buff += e_t[0] * j_t[0] + e_t[1] * j_t[1] + e_t[2] * j_t[2];
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3, real_t& buff) const {
      const auto i1_ = COORD(i1);
      const auto i2_ = COORD(i2);
      const auto i3_ = COORD(i3);
      if constexpr (F == StatsID::B2) {
        if constexpr (I == 1) {
          const auto b1_u = EM(i1, i2, i3, em::bx1);
          const auto b1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_, i2_ + HALF, i3_ + HALF },
            b1_u);
          buff += b1_u * b1_d;
        } else if constexpr (I == 2) {
          const auto b2_u = EM(i1, i2, i3, em::bx2);
          const auto b2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_ + HALF, i2_, i3_ + HALF },
            b2_u);
          buff += b2_u * b2_d;
        } else {
          const auto b3_u = EM(i1, i2, i3, em::bx3);
          const auto b3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_ + HALF, i2_ + HALF, i3_ },
            b3_u);
          buff += b3_u * b3_d;
        }
      } else if constexpr (F == StatsID::E2) {
        if constexpr (I == 1) {
          const auto e1_u = EM(i1, i2, i3, em::ex1);
          const auto e1_d = metric.template transform<1, Idx::U, Idx::D>(
            { i1_ + HALF, i2_, i3_ },
            e1_u);
          buff += e1_u * e1_d;
        } else if constexpr (I == 2) {
          const auto e2_u = EM(i1, i2, i3, em::ex2);
          const auto e2_d = metric.template transform<2, Idx::U, Idx::D>(
            { i1_, i2_ + HALF, i3_ },
            e2_u);
          buff += e2_u * e2_d;
        } else {
          const auto e3_u = EM(i1, i2, i3, em::ex3);
          const auto e3_d = metric.template transform<3, Idx::U, Idx::D>(
            { i1_, i2_, i3_ + HALF },
            e3_u);
          buff += e3_u * e3_d;
        }
      } else if constexpr (F == StatsID::ExB) {
        if constexpr (I == 1) {
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex2) + EM(i1 + 1, i2, i3, em::ex2) +
               EM(i1, i2, i3 + 1, em::ex2) + EM(i1 + 1, i2, i3 + 1, em::ex2)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex3) + EM(i1 + 1, i2, i3, em::ex3) +
               EM(i1, i2 + 1, i3, em::ex3) + EM(i1 + 1, i2 + 1, i3, em::ex3)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx2) + EM(i1, i2 + 1, i3, em::bx2)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx3) + EM(i1, i2, i3 + 1, em::bx3)));
          buff += e2_t * b3_t - e3_t * b2_t;
        } else if constexpr (I == 2) {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex1) + EM(i1, i2 + 1, i3, em::ex1) +
               EM(i1, i2, i3 + 1, em::ex1) + EM(i1, i2 + 1, i3 + 1, em::ex1)));
          const auto e3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex3) + EM(i1 + 1, i2, i3, em::ex3) +
               EM(i1, i2 + 1, i3, em::ex3) + EM(i1 + 1, i2 + 1, i3, em::ex3)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx1) + EM(i1 + 1, i2, i3, em::bx1)));
          const auto b3_t = metric.template transform<3, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx3) + EM(i1, i2, i3 + 1, em::bx3)));
          buff += e3_t * b1_t - e1_t * b3_t;
        } else {
          const auto e1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex1) + EM(i1, i2 + 1, i3, em::ex1) +
               EM(i1, i2, i3 + 1, em::ex1) + EM(i1, i2 + 1, i3 + 1, em::ex1)));
          const auto e2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            INV_4 *
              (EM(i1, i2, i3, em::ex2) + EM(i1 + 1, i2, i3, em::ex2) +
               EM(i1, i2, i3 + 1, em::ex2) + EM(i1 + 1, i2, i3 + 1, em::ex2)));
          const auto b1_t = metric.template transform<1, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx1) + EM(i1 + 1, i2, i3, em::bx1)));
          const auto b2_t = metric.template transform<2, Idx::U, Idx::T>(
            { i1_ + HALF, i2_ + HALF, i3_ + HALF },
            HALF * (EM(i1, i2, i3, em::bx2) + EM(i1, i2 + 1, i3, em::bx2)));
          buff += e1_t * b2_t - e2_t * b1_t;
        }
      } else if constexpr (F == StatsID::JdotE) {
        vec_t<Dim::_3D> e_t { ZERO };
        vec_t<Dim::_3D> j_t { ZERO };
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF, i2_ + HALF, i3_ + HALF },
          { INV_4 * (EM(i1, i2, i3, em::ex1) + EM(i1, i2 + 1, i3, em::ex1) +
                     EM(i1, i2, i3 + 1, em::ex1) + EM(i1, i2 + 1, i3 + 1, em::ex1)),
            INV_4 * (EM(i1, i2, i3, em::ex2) + EM(i1 + 1, i2, i3, em::ex2) +
                     EM(i1, i2, i3 + 1, em::ex2) + EM(i1 + 1, i2, i3 + 1, em::ex2)),
            INV_4 * (EM(i1, i2, i3, em::ex3) + EM(i1 + 1, i2, i3, em::ex3) +
                     EM(i1, i2 + 1, i3, em::ex3) + EM(i1 + 1, i2 + 1, i3, em::ex3)) },
          e_t);
        metric.template transform<Idx::U, Idx::T>(
          { i1_ + HALF, i2_ + HALF, i3_ + HALF },
          { INV_4 * (J(i1, i2, i3, cur::jx1) + J(i1, i2 + 1, i3, cur::jx1) +
                     J(i1, i2, i3 + 1, cur::jx1) + J(i1, i2 + 1, i3 + 1, cur::jx1)),
            INV_4 * (J(i1, i2, i3, cur::jx2) + J(i1 + 1, i2, i3, cur::jx2) +
                     J(i1, i2, i3 + 1, cur::jx2) + J(i1 + 1, i2, i3 + 1, cur::jx2)),
            INV_4 * (J(i1, i2, i3, cur::jx3) + J(i1 + 1, i2, i3, cur::jx3) +
                     J(i1, i2 + 1, i3, cur::jx3) + J(i1 + 1, i2 + 1, i3, cur::jx3)) },
          j_t);
        buff += e_t[0] * j_t[0] + e_t[1] * j_t[1] + e_t[2] * j_t[2];
      }
    }
  };

} // namespace kernel

#endif
