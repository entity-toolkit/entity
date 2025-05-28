/**
 * @file utils/diag.h
 * @brief Routines for diagnostics output at every step
 * @implements
 *   - diag::printDiagnostics -> void
 * @cpp:
 *   - diag.cpp
 * @namespces:
 *   - diag::
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef GLOBAL_UTILS_DIAG_H
#define GLOBAL_UTILS_DIAG_H

#include "utils/progressbar.h"
#include "utils/timer.h"

#include <string>
#include <vector>

namespace diag {

  /**
   * @brief Print diagnostics to the console
   * @param step
   * @param tot_steps
   * @param time
   * @param dt
   * @param timers
   * @param duration_history
   * @param ncells (total)
   * @param species_labels (vector of particle labels)
   * @param npart (per each species)
   * @param maxnpart (per each species)
   * @param prtlclear (if true, dead particles were removed)
   * @param output (if true, output was written)
   * @param checkpoint (if true, checkpoint was written)
   * @param colorful_print (if true, print with colors)
   */
  void printDiagnostics(timestep_t,
                        timestep_t,
                        simtime_t,
                        simtime_t,
                        timer::Timers&,
                        pbar::DurationHistory&,
                        ncells_t,
                        const std::vector<std::string>&,
                        const std::vector<npart_t>&,
                        const std::vector<npart_t>&,
                        bool,
                        bool,
                        bool,
                        bool);

} // namespace diag

#endif // GLOBAL_UTILS_DIAG_H
