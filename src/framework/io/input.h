#ifndef IO_INPUT_H
#define IO_INPUT_H

#include <plog/Log.h>
#include <toml.hpp>

#include <stdexcept>
#include <string>

namespace ntt {
  namespace {
    auto dataExistsInToml(const toml::value& inputdata,
                          const std::string& blockname,
                          const std::string& variable) -> bool {
      if (inputdata.contains(blockname)) {
        auto& val_block = toml::find(inputdata, blockname);
        return val_block.contains(variable);
      }
      return false;
    }
  }    // namespace

  template <typename T>
  auto readFromInput(const toml::value& inputdata,
                     const std::string& blockname,
                     const std::string& variable) -> T {
    if (dataExistsInToml(inputdata, blockname, variable)) {
      auto& val_block = toml::find(inputdata, blockname);
      return toml::find<T>(val_block, variable);
    }
    PLOGE_(LogFile) << "Cannot find variable <" << variable << "> from block [" << blockname
                    << "] in the input file.";
    throw std::invalid_argument("cannot find variable in the input");
  }
  template <typename T>
  auto readFromInput(const toml::value& inputdata,
                     const std::string& blockname,
                     const std::string& variable,
                     const T&           defval) -> T {
    if (dataExistsInToml(inputdata, blockname, variable)) {
      auto& val_block = toml::find(inputdata, blockname);
      return toml::find<T>(val_block, variable);
    }
    return defval;
  }

}    // namespace ntt

#endif    // IO_INPUT_H
