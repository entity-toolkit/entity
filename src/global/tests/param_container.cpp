#include "utils/param_container.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

auto main() -> int {
  auto p = prm::Parameters();

  const auto d_vec = std::vector<bool> { true, true, false };
  const auto e_vec = std::vector<std::vector<std::string>> {
    { "a" },
    { "b", "c" }
  };
  const auto nonexist_vec = std::vector<int> { 1, 2, 3 };

  p.set("a", 1);
  p.set("b", std::string("hello world"));
  p.set("c", 3.14);
  p.set("d", d_vec);
  p.set("e", e_vec);
  try {
    errorIf(p.get<int>("a") != 1, "Failed to get int");
    errorIf(p.get<std::string>("b") != "hello world", "Failed to get string");
    errorIf(p.get<double>("c") != 3.14, "Failed to get double");
    errorIf(p.get<std::vector<bool>>("d") != d_vec,
            "Failed to get vector of bools");
    errorIf(p.get<std::vector<std::vector<std::string>>>("e") != e_vec,
            "Failed to get vector of vector of strings");
    errorIf(p.get<std::vector<int>>("nonexist", nonexist_vec) != nonexist_vec,
            "Failed to fallback to default");

    errorIf(p.stringize<int>("a") != "1", "Wrong stringize for int");
    errorIf(p.stringize<std::string>("b") != "hello world",
            "Wrong stringize for string");
    errorIf(p.stringize<double>("c") != "3.14", "Wrong stringize for double");
    errorIf(p.stringize<bool>("d") != "[true, true, false]",
            "Wrong stringize for vector of bools");
    errorIf(p.stringize<std::string>("e") != "[{a}, {b, c}]",
            "Wrong stringize for vector of vector of strings");

  } catch (std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
  return 0;
}