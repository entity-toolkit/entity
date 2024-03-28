/**
 * @file defaults.h
 * @brief Default values for the simulation parameters
 * @depends:
 *   - global.h
 * @namespaces:
 *   - ntt::defaults
 * @note These values are used when the user does not provide them in the input file
 */

#ifndef GLOBAL_DEFAULTS_H
#define GLOBAL_DEFAULTS_H

#include "global.h"

#include <string>
#include <vector>

#include <string_view>

namespace ntt::defaults {
  constexpr std::string_view input_filename = "input";
  constexpr std::string_view output_path    = "output";

  const real_t correction = 1.0;
  const real_t cfl        = 0.95;

  const unsigned short current_filters = 0;

  const std::string  em_pusher     = "Boris";
  const std::string  ph_pusher     = "Photon";
  const unsigned int sort_interval = 100;

  namespace qsph {
    const real_t r0 = 0.0;
    const real_t h  = 0.0;
  } // namespace qsph

  namespace ks {
    const real_t a = 0.0;
  } // namespace ks

  namespace gr {
    const real_t pusher_eps   = 1e-6;
    const int    pusher_niter = 10;
  } // namespace gr

  namespace bc {
    const real_t d_absorb_frac = 0.01;
    const real_t absorb_coeff  = 1.0;
  } // namespace bc

  namespace output {
    const std::string format      = "hdf5";
    const int         interval    = 1;
    const int         mom_smooth  = 1;
    const std::size_t prtl_stride = 100;
  } // namespace output

  const std::string log_level       = "info";
  const int         diag_interval   = 1;
  const bool        blocking_timers = false;

  namespace gca {
    const real_t EovrB_max = 0.9;
  } // namespace gca
} // namespace ntt::defaults

#endif // GLOBAL_DEFAULTS_H