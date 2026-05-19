#include "output/utils/tuning.h"

#include "utils/formatting.h"

#include <adios2.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <algorithm>
#include <string>

namespace out {

  auto TotalAggregators(int aggregators_per_node) -> int {
    if (aggregators_per_node <= 0) {
      return 0;
    }
    int num_nodes = 1;
#if defined(MPI_ENABLED)
    int      world_size = 1, local_size = 1;
    MPI_Comm shm;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm);
    MPI_Comm_size(shm, &local_size);
    MPI_Comm_free(&shm);
    num_nodes = (local_size > 0) ? (world_size / local_size) : 1;
#endif
    return std::max(1, num_nodes * aggregators_per_node);
  }

  void ApplyBp5Tuning(adios2::IO&        io,
                      const std::string& engine,
                      const Bp5Tuning&   bp5) {
    // BP5 tuning for large-scale parallel filesystems.
    // Per-node aggregator count scales with the NIC layout;
    // aggregators_per_node == 0 leaves the ADIOS2 default (one per node).
    const auto eng = fmt::toLower(engine);
    if (eng != "bpfile" && eng != "bp5") {
      return;
    }
    const auto num_agg = std::to_string(TotalAggregators(bp5.aggregators_per_node));
    io.SetParameter("AggregationType", "TwoLevelShm");
    io.SetParameter("NumAggregators", num_agg);
    io.SetParameter("NumSubFiles", num_agg);
    io.SetParameter("BufferChunkSize", std::to_string(bp5.buffer_chunk_size));
    io.SetParameter("MaxShmSize", std::to_string(bp5.max_shm_size));
    io.SetParameter("AsyncOpen", "true");
    io.SetParameter("AsyncWrite", "true");
    io.SetParameter("OpenTimeoutSecs", "600");
  }

} // namespace out
