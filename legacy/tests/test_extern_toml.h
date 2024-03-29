#ifndef TEST_EXTERN_TOML_H
#define TEST_EXTERN_TOML_H

#include <doctest.h>
#include <toml/toml.hpp>

#include <vector>

TEST_CASE("testing toml library") {
  std::string        valid_toml = R"TOML(
    title = "brand new code"
    [apple]
    name = "Mr. Apple"
    friends = ["Mr. Orange"]
    married = false

    [orange]
    name = "Mr. Orange"
    color = "orange"
    age = 2
  )TOML";
  std::istringstream is(valid_toml, std::ios_base::binary | std::ios_base::in);
  auto               data = toml::parse(is, "std::string");

  std::string title = toml::find_or<std::string>(data, "title", "NA");

  CHECK(title == "brand new code");

  auto& orange_block = toml::find(data, "orange");
  auto  orange_name  = toml::find<std::string>(orange_block, "name");
  auto  orange_age   = toml::find<int>(orange_block, "age");

  CHECK(orange_name == "Mr. Orange");
  CHECK(orange_age == 2);

  auto& apple_block   = toml::find(data, "apple");
  auto  apple_name    = toml::find<std::string>(apple_block, "name");
  auto  apple_age     = toml::find_or<int>(apple_block, "age", 0);
  auto  apple_married = toml::find<bool>(apple_block, "married");
  auto  apple_friends = toml::find<std::vector<std::string>>(apple_block, "friends");

  CHECK(apple_name == "Mr. Apple");
  CHECK(apple_age == 0);
  CHECK(!apple_married);
  CHECK(apple_friends[0] == "Mr. Orange");
  CHECK(apple_friends.size() == 1);
}

#endif
