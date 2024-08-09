#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics { traits::compatible_with<Metric::Minkowski>::value };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    array_t<real_t**> cbuff;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>&)
      : arch::ProblemGenerator<S, M>(p) {}

    inline PGen() {}

    void CustomPostStep(std::size_t step, long double, Domain<S, M>& domain) {
      if (step == 0) {
        // allocate the array at time = 0
        cbuff = array_t<real_t**>("cbuff",
                                  domain.mesh.n_all(in::x1),
                                  domain.mesh.n_all(in::x2));
      }
      // fill with zeros
      Kokkos::deep_copy(cbuff, ZERO);
      // populate the array atomically (here it's not strictly necessary)
      auto cbuff_sc = Kokkos::Experimental::create_scatter_view(cbuff);
      Kokkos::parallel_for(
        "FillCbuff",
        domain.mesh.rangeActiveCells(),
        Lambda(index_t i1, index_t i2) {
          auto cbuff_acc     = cbuff_sc.access();
          cbuff_acc(i1, i2) += static_cast<real_t>(i1 + i2);
        });
      Kokkos::Experimental::contribute(cbuff, cbuff_sc);
    }

    void CustomFieldOutput(const std::string&   name,
                           ndfield_t<M::Dim, 6> buffer,
                           std::size_t          index,
                           const Domain<S, M>&  domain) {
      printf("CustomFieldOutput: %s\n", name.c_str());
      // examples for 2D
      if (name == "mybuff") {
        // copy the custom buffer to the buffer output
        Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL, Kokkos::ALL, index),
                          cbuff);
      } else if (name == "EdotB+1") {
        // calculate the custom buffer from EM fields
        const auto& EM = domain.fields.em;
        Kokkos::parallel_for(
          "EdotB+1",
          domain.mesh.rangeActiveCells(),
          Lambda(index_t i1, index_t i2) {
            buffer(i1, i2, index) = EM(i1, i2, em::ex1) * EM(i1, i2, em::bx1) +
                                    EM(i1, i2, em::ex2) * EM(i1, i2, em::bx2) +
                                    EM(i1, i2, em::ex3) * EM(i1, i2, em::bx3) +
                                    ONE;
          });
      } else {
        raise::Error("Custom output not provided", HERE);
      }
    }
  };

} // namespace user

#endif
