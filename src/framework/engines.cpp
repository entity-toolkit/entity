/**
 * Engine specific instantiations
 */
#include "wrapper.h"

#include "simulation.h"

#include "io/output.h"
#include "io/writer.h"
#include "meshblock/fields.h"
#include "meshblock/meshblock.h"
#include "meshblock/particles.h"

#include "communications/comm_nompi.cpp"
#include "communications/comm_mpi.cpp"
#include "communications/currents_sync.cpp"
#include "io/output_flds.cpp"
#include "io/output_prtls.cpp"
#include "io/writer.cpp"
#include "meshblock/fields.cpp"
#include "meshblock/meshblock.cpp"
#include "meshblock/meshblock_aux.cpp"
#include "meshblock/particles.cpp"
#include "meshblock/species.cpp"
#include "simulation.cpp"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <vector>

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

template void ntt::OutputField::compute<ntt::Dim1, ntt::SANDBOXEngine>(
  const ntt::SimulationParams&, ntt::Meshblock<ntt::Dim1, ntt::SANDBOXEngine>&) const;
template void ntt::OutputField::compute<ntt::Dim2, ntt::SANDBOXEngine>(
  const ntt::SimulationParams&, ntt::Meshblock<ntt::Dim2, ntt::SANDBOXEngine>&) const;
template void ntt::OutputField::compute<ntt::Dim3, ntt::SANDBOXEngine>(
  const ntt::SimulationParams&, ntt::Meshblock<ntt::Dim3, ntt::SANDBOXEngine>&) const;

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

template void ntt::OutputField::compute<ntt::Dim1, ntt::PICEngine>(
  const ntt::SimulationParams&, ntt::Meshblock<ntt::Dim1, ntt::PICEngine>&) const;
template void ntt::OutputField::compute<ntt::Dim2, ntt::PICEngine>(
  const ntt::SimulationParams&, ntt::Meshblock<ntt::Dim2, ntt::PICEngine>&) const;
template void ntt::OutputField::compute<ntt::Dim3, ntt::PICEngine>(
  const ntt::SimulationParams&, ntt::Meshblock<ntt::Dim3, ntt::PICEngine>&) const;

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

template void ntt::OutputField::compute<ntt::Dim2, ntt::GRPICEngine>(
  const ntt::SimulationParams&, ntt::Meshblock<ntt::Dim2, ntt::GRPICEngine>&) const;
template void ntt::OutputField::compute<ntt::Dim3, ntt::GRPICEngine>(
  const ntt::SimulationParams&, ntt::Meshblock<ntt::Dim3, ntt::GRPICEngine>&) const;

#ifdef OUTPUT_ENABLED
template void ntt::OutputField::put<ntt::Dim1, ntt::SANDBOXEngine>(
  adios2::IO&, adios2::Engine&, ntt::Meshblock<ntt::Dim1, ntt::SANDBOXEngine>&) const;
template void ntt::OutputField::put<ntt::Dim2, ntt::SANDBOXEngine>(
  adios2::IO&, adios2::Engine&, ntt::Meshblock<ntt::Dim2, ntt::SANDBOXEngine>&) const;
template void ntt::OutputField::put<ntt::Dim3, ntt::SANDBOXEngine>(
  adios2::IO&, adios2::Engine&, ntt::Meshblock<ntt::Dim3, ntt::SANDBOXEngine>&) const;

template void ntt::OutputParticles::put<ntt::Dim1, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  const ntt::Metadomain<ntt::Dim1>&,
  ntt::Meshblock<ntt::Dim1, ntt::SANDBOXEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim2, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  const ntt::Metadomain<ntt::Dim2>&,
  ntt::Meshblock<ntt::Dim2, ntt::SANDBOXEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim3, ntt::SANDBOXEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  const ntt::Metadomain<ntt::Dim3>&,
  ntt::Meshblock<ntt::Dim3, ntt::SANDBOXEngine>&) const;

template void ntt::OutputField::put<ntt::Dim1, ntt::PICEngine>(
  adios2::IO&, adios2::Engine&, ntt::Meshblock<ntt::Dim1, ntt::PICEngine>&) const;
template void ntt::OutputField::put<ntt::Dim2, ntt::PICEngine>(
  adios2::IO&, adios2::Engine&, ntt::Meshblock<ntt::Dim2, ntt::PICEngine>&) const;
template void ntt::OutputField::put<ntt::Dim3, ntt::PICEngine>(
  adios2::IO&, adios2::Engine&, ntt::Meshblock<ntt::Dim3, ntt::PICEngine>&) const;

template void ntt::OutputParticles::put<ntt::Dim1, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  const ntt::Metadomain<ntt::Dim1>&,
  ntt::Meshblock<ntt::Dim1, ntt::PICEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim2, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  const ntt::Metadomain<ntt::Dim2>&,
  ntt::Meshblock<ntt::Dim2, ntt::PICEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim3, ntt::PICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  const ntt::Metadomain<ntt::Dim3>&,
  ntt::Meshblock<ntt::Dim3, ntt::PICEngine>&) const;

template void ntt::OutputField::put<ntt::Dim2, ntt::GRPICEngine>(
  adios2::IO&, adios2::Engine&, ntt::Meshblock<ntt::Dim2, ntt::GRPICEngine>&) const;
template void ntt::OutputField::put<ntt::Dim3, ntt::GRPICEngine>(
  adios2::IO&, adios2::Engine&, ntt::Meshblock<ntt::Dim3, ntt::GRPICEngine>&) const;

template void ntt::OutputParticles::put<ntt::Dim2, ntt::GRPICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  const ntt::Metadomain<ntt::Dim2>&,
  ntt::Meshblock<ntt::Dim2, ntt::GRPICEngine>&) const;
template void ntt::OutputParticles::put<ntt::Dim3, ntt::GRPICEngine>(
  adios2::IO&,
  adios2::Engine&,
  const ntt::SimulationParams&,
  const ntt::Metadomain<ntt::Dim3>&,
  ntt::Meshblock<ntt::Dim3, ntt::GRPICEngine>&) const;
#endif    // OUTPUT_ENABLED