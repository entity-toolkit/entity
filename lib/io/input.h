#ifndef IO_INPUT_H
#define IO_INPUT_H

#include <string_view>
#include <string>
#include <vector>

namespace io {
  class CommandLineArguments {
  private:
    bool _initialized = false;
    std::vector<std::string_view> _args;
  public:
    void readCommandLineArguments(int argc, char *argv[]);
    std::string_view getArgument (std::string_view key, std::string_view def);
    std::string_view getArgument (std::string_view key);
    bool isSpecified(std::string_view key);
  };
  extern CommandLineArguments cl_args;
  extern std::string_view InputFile;
}

#endif
