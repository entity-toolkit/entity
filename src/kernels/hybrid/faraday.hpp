#ifndef KERNELS_HYBRID_FARADAY_HPP
#define KERNELS_HYBRID_FARADAY_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace kernel::hybrid {

  template <Dimension D, uint8_t N>
  class Faraday_kernel {
    ndfield_t<D, 6> Ein;
    ndfield_t<D, 6> Bin;
    ndfield_t<D, N> Bout;

    const uint8_t comp_Ein;
    const uint8_t comp_Bin;
    const uint8_t comp_Bout;

    const real_t dt;

  public:
    Faraday_kernel(const ndfield_t<D, 6>& Ein,
                   const ndfield_t<D, 6>& Bin,
                   ndfield_t<D, N>&       Bout,
                   uint8_t                comp_Ein,
                   uint8_t                comp_Bin,
                   uint8_t                comp_Bout,
                   real_t                 dt)
      : Ein { Ein }
      , Bin { Bin }
      , Bout { Bout }
      , comp_Ein { comp_Ein }
      , comp_Bin { comp_Bin }
      , comp_Bout { comp_Bout }
      , dt { dt } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        // B_x is constant in 1D (dB_x/dt = -(curl E)_x = 0). Bout here is a
        // SEPARATE scratch buffer (cur = Bf*/Bf** for the predictor pushes),
        // NOT in-place on `em`, and is zero-initialized -- so B_x must be
        // copied through, else the subsequent EMF (Ohm's-law Hall/motional
        // terms) would read Bf*_x = 0 instead of B_x^n.
        Bout(i1, comp_Bout + 0) = Bin(i1, comp_Bin + 0);
        Bout(i1, comp_Bout + 1) = Bin(i1, comp_Bin + 1) -
                                  dt * (-Ein(i1 + 1, comp_Ein + 2) +
                                        Ein(i1, comp_Ein + 2));
        Bout(i1, comp_Bout + 2) = Bin(i1, comp_Bin + 2) -
                                  dt * (Ein(i1 + 1, comp_Ein + 1) -
                                        Ein(i1, comp_Ein + 1));
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        Bout(i1, i2, comp_Bout + 0) = Bin(i1, i2, comp_Bin + 0) -
                                      dt * (Ein(i1, i2 + 1, comp_Ein + 2) -
                                            Ein(i1, i2, comp_Ein + 2));
        Bout(i1, i2, comp_Bout + 1) = Bin(i1, i2, comp_Bin + 1) -
                                      dt * (-Ein(i1 + 1, i2, comp_Ein + 2) +
                                            Ein(i1, i2, comp_Ein + 2));
        Bout(i1, i2, comp_Bout + 2) = Bin(i1, i2, comp_Bin + 2) -
                                      dt * (-Ein(i1, i2 + 1, comp_Ein + 0) +
                                            Ein(i1, i2, comp_Ein + 0) +
                                            Ein(i1 + 1, i2, comp_Ein + 1) -
                                            Ein(i1, i2, comp_Ein + 1));
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        Bout(i1, i2, i3, comp_Bout + 0) = Bin(i1, i2, i3, comp_Bin + 0) -
                                          dt *
                                            (-Ein(i1, i2, i3 + 1, comp_Ein + 1) +
                                             Ein(i1, i2, i3, comp_Ein + 1) +
                                             Ein(i1, i2 + 1, i3, comp_Ein + 2) -
                                             Ein(i1, i2, i3, comp_Ein + 2));
        Bout(i1, i2, i3, comp_Bout + 1) = Bin(i1, i2, i3, comp_Bin + 1) -
                                          dt * (Ein(i1, i2, i3 + 1, comp_Ein + 0) -
                                                Ein(i1, i2, i3, comp_Ein + 0) -
                                                Ein(i1 + 1, i2, i3, comp_Ein + 2) +
                                                Ein(i1, i2, i3, comp_Ein + 2));
        Bout(i1, i2, i3, comp_Bout + 2) = Bin(i1, i2, i3, comp_Bin + 2) -
                                          dt *
                                            (-Ein(i1, i2 + 1, i3, comp_Ein + 0) +
                                             Ein(i1, i2, i3, comp_Ein + 0) +
                                             Ein(i1 + 1, i2, i3, comp_Ein + 1) -
                                             Ein(i1, i2, i3, comp_Ein + 1));
      } else {
        raise::KernelError(HERE, "Faraday_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::hybrid

#endif // KERNELS_HYBRID_FARADAY_HPP
