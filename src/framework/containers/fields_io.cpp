#include "enums.h"
#include "global.h"

#include "utils/log.h"

#include "framework/containers/fields.h"
#include "output/utils/readers.h"
#include "output/utils/writers.h"

#include <adios2.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <vector>

namespace ntt {

  template <Dimension D, SimEngine::type S>
  void Fields<D, S>::CheckpointDeclare(
    adios2::IO&                  io,
    const std::vector<ncells_t>& local_shape,
    const std::vector<ncells_t>& global_shape,
    const std::vector<ncells_t>& local_offset) const {
    logger::Checkpoint("Declaring fields checkpoint", HERE);

    auto gs6 = std::vector<ncells_t>(global_shape.begin(), global_shape.end());
    auto lo6 = std::vector<ncells_t>(local_offset.begin(), local_offset.end());
    auto ls6 = std::vector<ncells_t>(local_shape.begin(), local_shape.end());
    gs6.push_back(6);
    lo6.push_back(0);
    ls6.push_back(6);

    io.DefineVariable<real_t>("em", gs6, lo6, ls6);
    if (S == ntt::SimEngine::GRPIC) {
      io.DefineVariable<real_t>("em0", gs6, lo6, ls6);
      auto gs3 = std::vector<ncells_t>(global_shape.begin(), global_shape.end());
      auto lo3 = std::vector<ncells_t>(local_offset.begin(), local_offset.end());
      auto ls3 = std::vector<ncells_t>(local_shape.begin(), local_shape.end());
      gs3.push_back(3);
      lo3.push_back(0);
      ls3.push_back(3);
      io.DefineVariable<real_t>("cur", gs3, lo3, ls3);
    }
  }

  template <Dimension D, SimEngine::type S>
  void Fields<D, S>::CheckpointRead(adios2::IO&                      io,
                                    adios2::Engine&                  reader,
                                    const adios2::Box<adios2::Dims>& range) {
    logger::Checkpoint("Reading fields checkpoint", HERE);

    auto range6 = adios2::Box<adios2::Dims>(range.first, range.second);
    range6.first.push_back(0);
    range6.second.push_back(6);
    out::ReadNDField<D, 6>(io, reader, "em", em, range6);
    if (S == ntt::SimEngine::GRPIC) {
      out::ReadNDField<D, 6>(io, reader, "em0", em0, range6);
      auto range3 = adios2::Box<adios2::Dims>(range.first, range.second);
      range3.first.push_back(0);
      range3.second.push_back(3);
      out::ReadNDField<D, 3>(io, reader, "cur", cur, range3);
    }
  }

  template <Dimension D, SimEngine::type S>
  void Fields<D, S>::CheckpointWrite(adios2::IO& io, adios2::Engine& writer) const {
    logger::Checkpoint("Writing fields checkpoint", HERE);

    out::WriteNDField<D, 6>(io, writer, "em", em);
    if (S == ntt::SimEngine::GRPIC) {
      out::WriteNDField<D, 6>(io, writer, "em0", em0);
      out::WriteNDField<D, 3>(io, writer, "cur", cur);
    }
  }

#define FIELDS_CHECKPOINTS(D, S)                                                \
  template void Fields<D, S>::CheckpointDeclare(adios2::IO&,                    \
                                                const std::vector<ncells_t>&,   \
                                                const std::vector<ncells_t>&,   \
                                                const std::vector<ncells_t>&)   \
    const;                                                                      \
  template void Fields<D, S>::CheckpointRead(adios2::IO&,                       \
                                             adios2::Engine&,                   \
                                             const adios2::Box<adios2::Dims>&); \
  template void Fields<D, S>::CheckpointWrite(adios2::IO&, adios2::Engine&) const;

  FIELDS_CHECKPOINTS(Dim::_1D, SimEngine::SRPIC)
  FIELDS_CHECKPOINTS(Dim::_2D, SimEngine::SRPIC)
  FIELDS_CHECKPOINTS(Dim::_3D, SimEngine::SRPIC)
  FIELDS_CHECKPOINTS(Dim::_2D, SimEngine::GRPIC)
  FIELDS_CHECKPOINTS(Dim::_3D, SimEngine::GRPIC)
#undef FIELDS_CHECKPOINTS

} // namespace ntt
