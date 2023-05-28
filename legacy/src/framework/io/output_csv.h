#ifndef IO_OUTPUT_CSV_H
#define IO_OUTPUT_CSV_H

#include "wrapper.h"
#include "io/output.h"
#include "meshblock/meshblock.h"

namespace ntt {
  // enum class OutputMode { UNDEFINED, WRITE, APPEND };

  namespace csv {
    // /**
    //  * @brief Write a field component to a csv file.
    //  * @param[in] fname Filename to write to.
    //  * @param[in] mblock Meshblock.
    //  * @param[in] em Field component to output.
    //  */
    // template <Dimension D, SimulationType S>
    // void writeField(const std::string&, const Meshblock<D, S>&, const em&);

    // /**
    //  * @brief Write a current component to a csv file.
    //  * @overload
    //  * @param[in] fname Filename to write to.
    //  * @param[in] mblock Meshblock.
    //  * @param[in] cur Current component to output.
    //  */
    // template <Dimension D, SimulationType S>
    // void writeField(const std::string&, const Meshblock<D, S>&, const cur&);

    // /**
    //  * @brief Write a particle data to a csv file.
    //  * @param[in] fname Filename to write to.
    //  * @param[in] mblock Meshblock.
    //  * @param[in] spec_id Species id.
    //  * @param[in] prtl_id Particle id.
    //  * @param[in] mode Write mode {WRITE, APPEND}.
    //  */
    // template <Dimension D, SimulationType S>
    // void writeParticle(std::string,
    //                    const Meshblock<D, S>&,
    //                    const std::size_t&,
    //                    const std::size_t&,
    //                    const OutputMode& mode = OutputMode::WRITE);

    // /**
    //  * @brief Ensure the file exists (raises an error if not).
    //  */
    // void ensureFileExists(const std::string&);
  } // namespace csv
} // namespace ntt

#endif