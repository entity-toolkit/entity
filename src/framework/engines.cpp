/**
 * Engine specific instantiations
 */
#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "output.h"
#include "particles.h"
#include "simulation.h"
#include "writer.h"

#include "fields.cpp"
#include "meshblock.cpp"
#include "output_flds.cpp"
#include "output_prtls.cpp"
#include "particles.cpp"
#include "simulation.cpp"
#include "species.cpp"
#include "writer.cpp"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

/**
 * SANDBOX Engine
 */
template class ntt::Simulation<ntt::Dim1, ntt::SANDBOXEngine>;
template class ntt::Simulation<ntt::Dim2, ntt::SANDBOXEngine>;
template class ntt::Simulation<ntt::Dim3, ntt::SANDBOXEngine>;

template struct ntt::Fields<ntt::Dim1, ntt::SANDBOXEngine>;
template struct ntt::Fields<ntt::Dim2, ntt::SANDBOXEngine>;
template struct ntt::Fields<ntt::Dim3, ntt::SANDBOXEngine>;

template class ntt::Meshblock<ntt::Dim1, ntt::SANDBOXEngine>;
template class ntt::Meshblock<ntt::Dim2, ntt::SANDBOXEngine>;
template class ntt::Meshblock<ntt::Dim3, ntt::SANDBOXEngine>;

template struct ntt::Particles<ntt::Dim1, ntt::SANDBOXEngine>;
template struct ntt::Particles<ntt::Dim2, ntt::SANDBOXEngine>;
template struct ntt::Particles<ntt::Dim3, ntt::SANDBOXEngine>;

template class ntt::Writer<ntt::Dim1, ntt::SANDBOXEngine>;
template class ntt::Writer<ntt::Dim2, ntt::SANDBOXEngine>;
template class ntt::Writer<ntt::Dim3, ntt::SANDBOXEngine>;

/**
 * PIC Engine
 */
template class ntt::Simulation<ntt::Dim1, ntt::PICEngine>;
template class ntt::Simulation<ntt::Dim2, ntt::PICEngine>;
template class ntt::Simulation<ntt::Dim3, ntt::PICEngine>;

template struct ntt::Fields<ntt::Dim1, ntt::PICEngine>;
template struct ntt::Fields<ntt::Dim2, ntt::PICEngine>;
template struct ntt::Fields<ntt::Dim3, ntt::PICEngine>;

template class ntt::Meshblock<ntt::Dim1, ntt::PICEngine>;
template class ntt::Meshblock<ntt::Dim2, ntt::PICEngine>;
template class ntt::Meshblock<ntt::Dim3, ntt::PICEngine>;

template struct ntt::Particles<ntt::Dim1, ntt::PICEngine>;
template struct ntt::Particles<ntt::Dim2, ntt::PICEngine>;
template struct ntt::Particles<ntt::Dim3, ntt::PICEngine>;

template class ntt::Writer<ntt::Dim1, ntt::PICEngine>;
template class ntt::Writer<ntt::Dim2, ntt::PICEngine>;
template class ntt::Writer<ntt::Dim3, ntt::PICEngine>;

/**
 * GRPIC Engine
 */
template class ntt::Simulation<ntt::Dim2, ntt::GRPICEngine>;
template class ntt::Simulation<ntt::Dim3, ntt::GRPICEngine>;

template struct ntt::Fields<ntt::Dim2, ntt::GRPICEngine>;
template struct ntt::Fields<ntt::Dim3, ntt::GRPICEngine>;

template class ntt::Meshblock<ntt::Dim2, ntt::GRPICEngine>;
template class ntt::Meshblock<ntt::Dim3, ntt::GRPICEngine>;

template struct ntt::Particles<ntt::Dim2, ntt::GRPICEngine>;
template struct ntt::Particles<ntt::Dim3, ntt::GRPICEngine>;

template class ntt::Writer<ntt::Dim2, ntt::GRPICEngine>;
template class ntt::Writer<ntt::Dim3, ntt::GRPICEngine>;

#ifdef OUTPUT_ENABLED

template void ntt::OutputField::put<ntt::Dim1, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim1, ntt::SANDBOXEngine>&) const;
template void ntt::OutputField::put<ntt::Dim2, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim2, ntt::SANDBOXEngine>&) const;
template void ntt::OutputField::put<ntt::Dim3, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim3, ntt::SANDBOXEngine>&) const;

template void ntt::OutputField::put<ntt::Dim1, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim1, ntt::PICEngine>&) const;
template void ntt::OutputField::put<ntt::Dim2, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim2, ntt::PICEngine>&) const;
template void ntt::OutputField::put<ntt::Dim3, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim3, ntt::PICEngine>&) const;

template void ntt::OutputField::put<ntt::Dim2, ntt::GRPICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim2, ntt::GRPICEngine>&) const;
template void ntt::OutputField::put<ntt::Dim3, ntt::GRPICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim3, ntt::GRPICEngine>&) const;

template void ntt::OutputParticles::put<ntt::Dim1, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim1, ntt::SANDBOXEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim2, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim2, ntt::SANDBOXEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim3, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim3, ntt::SANDBOXEngine>&) const;

template void ntt::OutputParticles::put<ntt::Dim1, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim1, ntt::PICEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim2, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim2, ntt::PICEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim3, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim3, ntt::PICEngine>&) const;

template void ntt::OutputParticles::put<ntt::Dim2, ntt::GRPICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim2, ntt::GRPICEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim3, ntt::GRPICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim3, ntt::GRPICEngine>&) const;

#endif