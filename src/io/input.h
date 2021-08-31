#ifndef IO_INPUT_H
#define IO_INPUT_H

#include "global.h"

#include <toml/toml.hpp>
#include <plog/Log.h>

#include <string>
#include <exception>

namespace ntt::io {
void dataExistsInToml(toml::value inputdata, const std::string &blockname, const std::string &variable) {
  if (inputdata.contains(blockname)) {
    auto &val_block = toml::find(inputdata, blockname);
    if (!val_block.contains(variable)) {
      PLOGE << "Cannot find variable <" << variable << "> from block [" << blockname << "] in the input file.";
      throw std::invalid_argument("Cannot find variable in input file.");
    }
  } else {
    PLOGE << "Cannot find block [" << blockname << "] in the input file.";
    throw std::invalid_argument("Cannot find blockname in input file.");
  }
}

template <typename T>
auto readTomlData(toml::value inputdata, const std::string &blockname, const std::string &variable) -> T {
  dataExistsInToml(inputdata, blockname, variable);
  auto &val_block = toml::find(inputdata, blockname);
  return toml::find<T>(val_block, variable);
}
} // namespace ntt::io

#endif
