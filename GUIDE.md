# Codestyle guide

## General

* Use `const` and `auto` declarations where possible.
* For real-valued literals, use `ONE`, `ZERO`, `HALF` etc. instead of `1.0`, `0.0`, `0.5` to ensure the compiler will not need to cast. If the value is not defined as a macro, use `static_cast<real_t>(123.4)`.

## Users

## Developers

* Use `{}` in declarations to signify a null (placeholder) value for the given variable:
  ```cpp
  auto a { -1 }; // <- value of `a` will be changed later (-1 is a placeholder)
  auto b = -1; // <- value of `b` is known at the time of declaration (but may change later)
  const auto b = -1; // <- value of `b` is not expected to change later
  ```
* Each header file has to have a description at the top, consisting of the following fields:
  * `@file` **[required]** the name of the file (as it should be imported)
  * `@brief` **[required]** brief description of what the file contains
  * `@implements` list of class/function/macros implementations
    * structs/classes in this section have no prefix (templates are marked with `<>`)
    * functions are marked with their return type, e.g. ` -> void`
    * type aliases have a prefix `type`
    * enums or enum-like objects are marked with `enum`
    * macros have a prefix `macro`
    * all of the above are also marked with their respective namespaces (if any): `namespace::`
  * `@depends:` internal header dependencies (not including std or other external libraries)
  * `@cpp:` list of cpp files that implement the header
  * `@namespaces:` list of namespaces defined in the file
  * `@macros:` list of macros that the file depends on
  * `@note` any additional notes (stack as many as necessary)
  
  Example:
  ```cpp
  /**
   * @file utils/formatting.h
   * @brief String formatting utilities
   * @implements
   *   - fmt::format<> -> std::string
   *   - fmt::toLower -> std::string
   *   - fmt::splitString -> std::vector<std::string>
   * @namespaces:
   *   - fmt::
   */

* `#ifdef` macros should be avoided. Use C++17 type traits or `if constexpr ()` expressions to specialize functions and classes instead (ideally, specialize them explicitly). `#ifdef`-s are only acceptable in platform/library-specific parts of the code (e.g., `MPI_ENABLED`, `GPU_ENABLED`, `DEBUG`, etc.).

* Header files should start with `#ifndef ... #define ...` and end with `#endif`; do not use `#pragma` guards. The name of the macro should be the same as the name of the file in uppercase, with underscores instead of dots and slashes. For example, for `global/utils/formatting.h`, the macro should be `GLOBAL_UTILS_FORMATTING_H`.

# Recommendations

* Do assertions on parameters and quantities whenever possible. Outside the kernels, use `raise::Error(message, HERE)` and `raise::ErrorIf(condition, message, HERE)` to throw exceptions. Inside the kernels, use `raise::KernelError(HERE, message, **args)`.

* When writing class or function templates, it is always a good practice to ensure the template argument is valid (depending on the context). When doing that, use SFINAE (see, e.g., `arch/traits.h`) to test whether the type is valid. For example:
  ```cpp
  template <typename T, typename = void>
  struct has_Foo : std::false_type {};

  template <typename T>
  struct has_Foo<T, std::void_t<decltype(&T::Foo)>>
    : std::true_type {};

  template <class B>
  class A {
    static_assert(has_Foo<B>::value, "B must have a Foo method");
  };
  ```