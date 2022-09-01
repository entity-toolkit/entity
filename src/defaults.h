#include "definitions.h"

#include <string>
#include <string_view>

namespace ntt {
  // defaults
  namespace defaults {
    constexpr std::string_view input_filename {"input"};
    constexpr std::string_view output_path {"output"};

    const std::string title {"PIC_Sim"};
    const int         n_species {0};
    const std::string pusher {"Boris"};
    const std::string metric {"minkowski"};

    const real_t runtime {1e10};
    const real_t correction {1.0};
    const real_t cfl {0.95};

    const unsigned short current_filters {0};

    const std::string output_format {"disabled"};
    const int         output_interval {1};
  } // namespace defaults
} // namespace ntt
