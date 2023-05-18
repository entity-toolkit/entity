#ifndef FRAMEWORK_IO_OUTPUT_PRTLS_H
#define FRAMEWORK_IO_OUTPUT_PRTLS_H

#include "wrapper.h"

#include "meshblock.h"
#include "particle_macros.h"
#include "particles.h"

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
                                 const short&           component)
      : m_mblock(mblock),
        m_particles(particles),
        m_buffer(buffer),
        m_stride(stride),
        m_component(component) {}

#ifdef MINKOWSKI_METRIC
    Inline void operator()(const OutputPositions_t&, index_t p) const {
      coord_t<D> xcode { ZERO }, xph { ZERO };
      if (m_component == 0) {
        xcode[0] = get_prtl_x1(m_particles, p * m_stride);
      } else if (m_component == 1) {
        xcode[1] = get_prtl_x2(m_particles, p * m_stride);
      } else if (m_component == 2) {
        xcode[2] = get_prtl_x3(m_particles, p * m_stride);
      }
      m_mblock.metric.x_Code2Cart(xcode, xph);
      m_buffer(p) = xph[m_component];
    }

    Inline void operator()(const OutputVelocities_t&, index_t p) const {
      if (m_component == 0) {
        m_buffer(p) = m_particles.ux1(p * m_stride);
      } else if (m_component == 1) {
        m_buffer(p) = m_particles.ux2(p * m_stride);
      } else if (m_component == 2) {
        m_buffer(p) = m_particles.ux3(p * m_stride);
      }
    }
#else
    // defined below
    Inline void operator()(const OutputVelocities_t&, index_t p) const {}
    Inline void operator()(const OutputPositions_t&, index_t p) const {}
#endif
  };

#ifndef MINKOWSKI_METRIC

  /* ------------------------------ PIC positions ----------------------------- */

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, PICEngine>::operator()(
    const OutputPositions_t&, index_t p) const {
    if (m_component == 2) {
      m_buffer(p) = m_particles.phi(p);
    } else {
      coord_t<Dim2> xcode { ZERO }, xph { ZERO };
      if (m_component == 0) {
        xcode[0] = get_prtl_x1(m_particles, p * m_stride);
      } else if (m_component == 1) {
        xcode[1] = get_prtl_x2(m_particles, p * m_stride);
      }
      m_mblock.metric.x_Code2Sph(xcode, xph);
      m_buffer(p) = xph[m_component];
    }
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim3, PICEngine>::operator()(
    const OutputPositions_t&, index_t p) const {
    coord_t<Dim3> xcode { ZERO }, xph { ZERO };
    if (m_component == 0) {
      xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    } else if (m_component == 1) {
      xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    } else if (m_component == 2) {
      xcode[2] = get_prtl_x3(m_particles, p * m_stride);
    }
    m_mblock.metric.x_Code2Sph(xcode, xph);
    m_buffer(p) = xph[m_component];
  }

  /* ----------------------------- GRPIC positions ---------------------------- */

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, GRPICEngine>::operator()(
    const OutputPositions_t&, index_t p) const {
    if (m_component == 2) {
      m_buffer(p) = m_particles.phi(p);
    } else {
      coord_t<Dim2> xcode { ZERO }, xph { ZERO };
      if (m_component == 0) {
        xcode[0] = get_prtl_x1(m_particles, p * m_stride);
      } else if (m_component == 1) {
        xcode[1] = get_prtl_x2(m_particles, p * m_stride);
      }
      m_mblock.metric.x_Code2Sph(xcode, xph);
      m_buffer(p) = xph[m_component];
    }
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim3, GRPICEngine>::operator()(
    const OutputPositions_t&, index_t p) const {
    coord_t<Dim3> xcode { ZERO }, xph { ZERO };
    if (m_component == 0) {
      xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    } else if (m_component == 1) {
      xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    } else if (m_component == 2) {
      xcode[2] = get_prtl_x3(m_particles, p * m_stride);
    }
    m_mblock.metric.x_Code2Sph(xcode, xph);
    m_buffer(p) = xph[m_component];
  }

  /* ----------------------------- PIC velocities ----------------------------- */

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim1, PICEngine>::operator()(
    const OutputVelocities_t&, index_t) const {
    NTTError("not applicable");
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim2, PICEngine>::operator()(
    const OutputVelocities_t&, index_t p) const {
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
    const OutputVelocities_t&, index_t p) const {
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
  Inline void PreparePrtlQuantities_kernel<Dim2, GRPICEngine>::operator()(
    const OutputVelocities_t&, index_t p) const {
    coord_t<Dim2> xcode { ZERO };
    vec_t<Dim3>   vcov_sph { ZERO };
    xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    m_mblock.metric.v3_Cov2SphCov(xcode,
                                  { m_particles.ux1(p * m_stride),
                                    m_particles.ux2(p * m_stride),
                                    m_particles.ux3(p * m_stride) },
                                  vcov_sph);
    m_buffer(p) = vcov_sph[m_component];
  }

  template <>
  Inline void PreparePrtlQuantities_kernel<Dim3, GRPICEngine>::operator()(
    const OutputVelocities_t&, index_t p) const {
    coord_t<Dim3> xcode { ZERO };
    vec_t<Dim3>   vcov_sph { ZERO };
    xcode[0] = get_prtl_x1(m_particles, p * m_stride);
    xcode[1] = get_prtl_x2(m_particles, p * m_stride);
    xcode[2] = get_prtl_x3(m_particles, p * m_stride);
    m_mblock.metric.v3_Cov2SphCov(xcode,
                                  { m_particles.ux1(p * m_stride),
                                    m_particles.ux2(p * m_stride),
                                    m_particles.ux3(p * m_stride) },
                                  vcov_sph);
    m_buffer(p) = vcov_sph[m_component];
  }

#endif

}    // namespace ntt

#endif    // FRAMEWORK_IO_OUTPUT_PRTLS_H