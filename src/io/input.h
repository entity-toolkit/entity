#ifndef IO_INPUT_H
#define IO_INPUT_H

#include <toml/toml.hpp>
#include <plog/Log.h>

#include <string>
#include <stdexcept>

namespace ntt {
namespace {
  void dataExistsInToml(const toml::value &inputdata, const std::string &blockname, const std::string &variable) {
    if (inputdata.contains(blockname)) {
      auto &val_block = toml::find(inputdata, blockname);
      if (!val_block.contains(variable)) {
        PLOGI << "Cannot find variable <" << variable << "> from block [" << blockname << "] in the input file.";
        throw std::invalid_argument("Cannot find variable in input file.");
      }
    } else {
      PLOGI << "Cannot find block [" << blockname << "] in the input file.";
      throw std::invalid_argument("Cannot find blockname in input file.");
    }
  }
}

template <typename T> auto readFromInput(const toml::value &inputdata, const std::string &blockname, const std::string &variable) -> T {
  dataExistsInToml(inputdata, blockname, variable);
  auto &val_block = toml::find(inputdata, blockname);
  return toml::find<T>(val_block, variable);
}
template <typename T>
auto readFromInput(const toml::value &inputdata, const std::string &blockname, const std::string &variable, const T &defval) -> T {
  try {
    return readFromInput<T>(inputdata, blockname, variable);
  } catch (std::exception &err) {
    PLOGI << "Variable <" << variable << "> of [" << blockname << "] not found. Falling back to default value.";
    return defval;
  }
}

}

#endif
