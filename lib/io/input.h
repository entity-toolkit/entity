#ifndef IO_INPUT_H
#define IO_INPUT_H

#include "global.h"

#include "toml/toml.hpp"

#include <string_view>

namespace ntt {
  namespace io {
    template<typename T>
    T readFromInput(std::string_view blockname, std::string_view variable) {
    }
  }
}

#endif
