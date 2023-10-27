#ifndef FRAMEWORK_IO_OUTPUT_PRTLS_H
#define FRAMEWORK_IO_OUTPUT_PRTLS_H

#include "wrapper.h"

#include "particle_macros.h"

#include "meshblock/meshblock.h"
#include "meshblock/particles.h"

namespace ntt {
  struct OutputPositions_t {};

  struct OutputVelocities_t {};

  template <Dimension D, SimulationEngine S>
  class PreparePrtlQuantities_kernel {
    Meshblock<D, S>   m_mblock;
    Particles<D, S>   m_particles;
    array_t<real_t*>  m_buffer;
    const std::size_t m_stride;
    const short       m_component;

  public:
    PreparePrtlQuantities_kernel(const Meshblock<D, S>& mblock,
                                 const Particles<D, S>& particles,
                                 array_t<real_t*>&      buffer,
                                 const std::size_t&     stride,
                                 const short&           component) :
      m_mblock(mblock),
      m_particles(particles),
      m_buffer(buffer),
      m_stride(stride),
      m_component(component) {}

    Inline auto operator()(const OutputPositions_t&, index_t) const -> void;
    Inline auto operator()(const OutputVelocities_t&, index_t) const -> void;
  };

  // default implementation
  template <Dimension D, SimulationEngine S>
  Inline auto PreparePrtlQuantities_kernel<D, S>::operator()(
    const OutputPositions_t&,
    index_t p) const -> void {
    coord_t<FullD> xcode { ZERO }, xph { ZERO };
    if (m_component == 0) {
      xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    } else if (m_component == 1) {
      xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    } else if (m_component == 2) {
      xcode[2] = get_prtl_x3(m_particles, p * m_stride);
    }
    m_mblock.metric.x_Code2Phys(xcode, xph);
    m_buffer(p) = xph[m_component];
  }

  template <Dimension D, SimulationEngine S>
  Inline auto PreparePrtlQuantities_kernel<D, S>::operator()(
    const OutputVelocities_t&,
    index_t p) const -> void {
    if (m_component == 0) {
      m_buffer(p) = m_particles.ux1(p * m_stride);
    } else if (m_component == 1) {
      m_buffer(p) = m_particles.ux2(p * m_stride);
    } else if (m_component == 2) {
      m_buffer(p) = m_particles.ux3(p * m_stride);
    }
  }

#ifndef MINKOWSKI_METRIC

  /* ------------------------------ PIC positions ----------------------------- */

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, PICEngine>::operator()(
    const OutputPositions_t&,
    index_t p) const {
    coord_t<Dim3> xcode { ZERO }, xph { ZERO };
    xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    xcode[2] = m_particles.phi(p * m_stride);
    m_mblock.metric.x_Code2Sph(xcode, xph);
    m_buffer(p) = xph[m_component];
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, SANDBOXEngine>::operator()(
    const OutputPositions_t&,
    index_t p) const {
    coord_t<Dim3> xcode { ZERO }, xph { ZERO };
    xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    xcode[2] = m_particles.phi(p * m_stride);
    m_mblock.metric.x_Code2Sph(xcode, xph);
    m_buffer(p) = xph[m_component];
  }

  /* ----------------------------- PIC velocities ----------------------------- */

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim1, PICEngine>::operator()(
    const OutputVelocities_t&,
    index_t) const {
    NTTError("not applicable");
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, PICEngine>::operator()(
    const OutputVelocities_t&,
    index_t p) const {
    coord_t<Dim3> xcode { ZERO };
    vec_t<Dim3>   vhat { ZERO };
    xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    xcode[2] = m_particles.phi(p * m_stride);
    m_mblock.metric.v3_Cart2Hat(xcode,
                                { m_particles.ux1(p * m_stride),
                                  m_particles.ux2(p * m_stride),
                                  m_particles.ux3(p * m_stride) },
                                vhat);
    m_buffer(p) = vhat[m_component];
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim3, PICEngine>::operator()(
    const OutputVelocities_t&,
    index_t p) const {
    coord_t<Dim3> xcode { ZERO };
    vec_t<Dim3>   vhat { ZERO };
    xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    xcode[2] = get_prtl_x3(m_particles, p * m_stride);
    m_mblock.metric.v3_Cart2Hat(xcode,
                                { m_particles.ux1(p * m_stride),
                                  m_particles.ux2(p * m_stride),
                                  m_particles.ux3(p * m_stride) },
                                vhat);
    m_buffer(p) = vhat[m_component];
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, SANDBOXEngine>::operator()(
    const OutputVelocities_t&,
    index_t p) const {
    coord_t<Dim3> xcode { ZERO };
    vec_t<Dim3>   vhat { ZERO };
    xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    xcode[2] = m_particles.phi(p * m_stride);
    m_mblock.metric.v3_Cart2Hat(xcode,
                                { m_particles.ux1(p * m_stride),
                                  m_particles.ux2(p * m_stride),
                                  m_particles.ux3(p * m_stride) },
                                vhat);
    m_buffer(p) = vhat[m_component];
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim3, SANDBOXEngine>::operator()(
    const OutputVelocities_t&,
    index_t p) const {
    coord_t<Dim3> xcode { ZERO };
    vec_t<Dim3>   vhat { ZERO };
    xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    xcode[2] = get_prtl_x3(m_particles, p * m_stride);
    m_mblock.metric.v3_Cart2Hat(xcode,
                                { m_particles.ux1(p * m_stride),
                                  m_particles.ux2(p * m_stride),
                                  m_particles.ux3(p * m_stride) },
                                vhat);
    m_buffer(p) = vhat[m_component];
  }

  /* ----------------------------- GRPIC positions ---------------------------- */

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, GRPICEngine>::operator()(
    const OutputPositions_t&,
    index_t p) const {
    // x is taken at (n - 1/2)
    coord_t<Dim3> xcode { ZERO }, xcode_prev { ZERO }, xph { ZERO };
    xcode[0]      = get_prtl_x1(m_particles, p * m_stride);
    xcode_prev[0] = get_prtl_x1_prev(m_particles, p * m_stride);
    xcode[0]      = HALF * (xcode[0] + xcode_prev[0]);
    xcode[1]      = get_prtl_x2(m_particles, p * m_stride);
    xcode_prev[1] = get_prtl_x2_prev(m_particles, p * m_stride);
    xcode[1]      = HALF * (xcode[1] + xcode_prev[1]);
    xcode[2]      = m_particles.phi(p * m_stride);
    m_mblock.metric.x_Code2Sph(xcode, xph);
    m_buffer(p) = xph[m_component];
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim3, GRPICEngine>::operator()(
    const OutputPositions_t&,
    index_t p) const {
    coord_t<Dim3> xcode { ZERO }, xcode_prev { ZERO }, xph { ZERO };
    // x is taken at (n - 1/2)
    if (m_component == 0) {
      xcode[0]      = get_prtl_x1(m_particles, p * m_stride);
      xcode_prev[0] = get_prtl_x1_prev(m_particles, p * m_stride);
      xcode[0]      = HALF * (xcode[0] + xcode_prev[0]);
    } else if (m_component == 1) {
      xcode[1]      = get_prtl_x2(m_particles, p * m_stride);
      xcode_prev[1] = get_prtl_x2_prev(m_particles, p * m_stride);
      xcode[1]      = HALF * (xcode[1] + xcode_prev[1]);
    } else if (m_component == 2) {
      xcode[2]      = get_prtl_x3(m_particles, p * m_stride);
      xcode_prev[2] = get_prtl_x3_prev(m_particles, p * m_stride);
      xcode[2]      = HALF * (xcode[2] + xcode_prev[2]);
    }
    m_mblock.metric.x_Code2Sph(xcode, xph);
    m_buffer(p) = xph[m_component];
  }

  /* ---------------------------- GRPIC velocities ---------------------------- */

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, GRPICEngine>::operator()(
    const OutputVelocities_t&,
    index_t p) const {
    // velocity should be at (n - 1/2)
    // velocity at (n - 1/2)
    coord_t<Dim2> xcode { ZERO }, xcode_prev { ZERO };
    vec_t<Dim3>   vcov_sph { ZERO };
    xcode[0]      = get_prtl_x1(m_particles, p * m_stride);
    xcode[1]      = get_prtl_x2(m_particles, p * m_stride);
    xcode_prev[0] = get_prtl_x1_prev(m_particles, p * m_stride);
    xcode_prev[1] = get_prtl_x2_prev(m_particles, p * m_stride);
    // x at (n - 1/2)
    xcode[0]      = HALF * (xcode[0] + xcode_prev[0]);
    xcode[1]      = HALF * (xcode[1] + xcode_prev[1]);
    m_mblock.metric.v3_Cov2PhysCov(xcode,
                                   { m_particles.ux1(p * m_stride),
                                     m_particles.ux2(p * m_stride),
                                     m_particles.ux3(p * m_stride) },
                                   vcov_sph);
    m_buffer(p) = vcov_sph[m_component];
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim3, GRPICEngine>::operator()(
    const OutputVelocities_t&,
    index_t p) const {
    // velocity should be at (n - 1/2)
    coord_t<Dim3> xcode { ZERO }, xcode_prev { ZERO };
    vec_t<Dim3>   vcov_sph { ZERO };
    xcode[0]      = get_prtl_x1(m_particles, p * m_stride);
    xcode[1]      = get_prtl_x2(m_particles, p * m_stride);
    xcode[2]      = get_prtl_x3(m_particles, p * m_stride);
    xcode_prev[0] = get_prtl_x1_prev(m_particles, p * m_stride);
    xcode_prev[1] = get_prtl_x2_prev(m_particles, p * m_stride);
    xcode_prev[2] = get_prtl_x3_prev(m_particles, p * m_stride);
    // x at (n - 1/2)
    xcode[0]      = HALF * (xcode[0] + xcode_prev[0]);
    xcode[1]      = HALF * (xcode[1] + xcode_prev[1]);
    xcode[2]      = HALF * (xcode[2] + xcode_prev[2]);
    m_mblock.metric.v3_Cov2PhysCov(xcode,
                                   { m_particles.ux1(p * m_stride),
                                     m_particles.ux2(p * m_stride),
                                     m_particles.ux3(p * m_stride) },
                                   vcov_sph);
    m_buffer(p) = vcov_sph[m_component];
  }

#endif

} // namespace ntt

#endif // FRAMEWORK_IO_OUTPUT_PRTLS_H