if(${PGEN_FOUND})
  printchoices(
    "Problem generator"
    "pgen"
    "${problem_generators}"
    ${PGEN}
    ${default_pgen}
    "${Blue}"
    PGEN_REPORT
    0)
elseif(${TESTS})
  set(TEST_NAMES "")
  foreach(test_dir IN LISTS TEST_DIRECTORIES)
    get_property(
      LOCAL_TEST_NAMES
      DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir}/tests
      PROPERTY TESTS)
    list(APPEND TEST_NAMES ${LOCAL_TEST_NAMES})
  endforeach()
  printchoices(
    "Test cases"
    ""
    "${TEST_NAMES}"
    ""
    "${ColorReset}"
    ""
    TESTS_REPORT
    0)
endif()

printchoices(
  "Precision"
  "precision"
  "${precisions}"
  ${precision}
  ${default_precision}
  "${Blue}"
  PRECISION_REPORT
  36)
printchoices(
  "Output"
  "output"
  "${ON_OFF_VALUES}"
  ${output}
  ${default_output}
  "${Green}"
  OUTPUT_REPORT
  36)
printchoices(
  "MPI"
  "mpi"
  "${ON_OFF_VALUES}"
  ${mpi}
  OFF
  "${Green}"
  MPI_REPORT
  36)
printchoices(
  "Debug mode"
  "DEBUG"
  "${ON_OFF_VALUES}"
  ${DEBUG}
  OFF
  "${Green}"
  DEBUG_REPORT
  36)

if(NOT ${PROJECT_VERSION_TWEAK} EQUAL 0)
  set(VERSION_SYMBOL "v${PROJECT_VERSION_MAJOR}." "${PROJECT_VERSION_MINOR}.")
  string(APPEND VERSION_SYMBOL
         "${PROJECT_VERSION_PATCH}-rc${PROJECT_VERSION_TWEAK}")
else()
  set(VERSION_SYMBOL "v${PROJECT_VERSION_MAJOR}.")
  string(APPEND VERSION_SYMBOL
         "${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}    ")
endif()

set(REPORT_TEXT
    "${Blue}              __        __
             /\\ \\__  __/\\ \\__
    __    ___\\ \\  _\\/\\_\\ \\  _\\  __  __
  / __ \\ / __ \\ \\ \\/\\/\\ \\ \\ \\/ /\\ \\/\\ \\
 /\\  __//\\ \\/\\ \\ \\ \\_\\ \\ \\ \\ \\_\\ \\ \\_\\ \\  __
 \\ \\____\\ \\_\\ \\_\\ \\__\\\\ \\_\\ \\__\\\\ \\____ \\/\\_\\
  \\/____/\\/_/\\/_/\\/__/ \\/_/\\/__/ \\/___/  \\/_/
                                    /\\___/
Entity ${VERSION_SYMBOL}\t\t    \\/__/")
string(APPEND REPORT_TEXT ${ColorReset} "\n")

string(APPEND REPORT_TEXT ${DASHED_LINE_SYMBOL} "\n" "Configurations" "\n")

if(${PGEN_FOUND})
  string(APPEND REPORT_TEXT "  " ${PGEN_REPORT} "\n")
elseif(${TESTS})
  string(APPEND REPORT_TEXT "  " ${TESTS_REPORT} "\n")
endif()

string(
  APPEND
  REPORT_TEXT
  "  "
  ${PRECISION_REPORT}
  "\n"
  "  "
  ${OUTPUT_REPORT}
  "\n")

string(REPLACE ";" "+" Kokkos_ARCH "${Kokkos_ARCH}")
string(REPLACE ";" "+" Kokkos_DEVICES "${Kokkos_DEVICES}")

string(
  APPEND
  REPORT_TEXT
  "  - ARCH [${Magenta}Kokkos_ARCH_***${ColorReset}]:         ${Kokkos_ARCH}"
  "\n"
  "  - DEVICES [${Magenta}Kokkos_ENABLE_***${ColorReset}]:    ${Kokkos_DEVICES}"
  "\n"
  "  "
  ${MPI_REPORT}
  "\n"
  "  "
  ${DEBUG_REPORT}
  "\n"
  ${DASHED_LINE_SYMBOL}
  "\n"
  "Compilers & dependencies"
  "\n")

string(
  APPEND
  REPORT_TEXT
  "  - C compiler [${Magenta}CMAKE_C_COMPILER${ColorReset}]: v"
  ${CMAKE_C_COMPILER_VERSION}
  "\n"
  "    ${Dim}"
  ${CMAKE_C_COMPILER}
  "${ColorReset}\n"
  "  - C++ compiler [${Magenta}CMAKE_CXX_COMPILER${ColorReset}]: v"
  ${CMAKE_CXX_COMPILER_VERSION}
  "\n"
  "    ${Dim}"
  ${CMAKE_CXX_COMPILER}
  "${ColorReset}\n")

if(${Kokkos_DEVICES} MATCHES "CUDA")
  if("${CMAKE_CUDA_COMPILER}" STREQUAL "")
    execute_process(COMMAND which nvcc OUTPUT_VARIABLE CUDACOMP)
  else()
    set(CUDACOMP ${CMAKE_CUDA_COMPILER})
  endif()
  string(STRIP ${CUDACOMP} CUDACOMP)
  set(cmd "${CUDACOMP} --version |")
  string(APPEND cmd " grep release | sed -e 's/.*release //' -e 's/,.*//'")
  execute_process(
    COMMAND bash -c ${cmd}
    OUTPUT_VARIABLE CUDACOMP_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(
    APPEND
    REPORT_TEXT
    "  - CUDA compiler: v"
    ${CUDACOMP_VERSION}
    "\n"
    "    ${Dim}"
    ${CUDACOMP}
    "${ColorReset}\n")
elseif(${Kokkos_DEVICES} MATCHES "HIP")
  set(cmd "hipcc --version | grep HIP | cut -d ':' -f 2 | tr -d ' '")
  execute_process(
    COMMAND bash -c ${cmd}
    OUTPUT_VARIABLE ROCM_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(APPEND REPORT_TEXT "  - ROCm: v" ${ROCM_VERSION} "\n")
endif()

string(APPEND REPORT_TEXT "  - Kokkos: v" ${Kokkos_VERSION} "\n")
if(${Kokkos_FOUND})
  string(APPEND REPORT_TEXT "    " ${Dim}${Kokkos_DIR}${ColorReset} "\n")
else()
  string(APPEND REPORT_TEXT "    " ${Dim}${Kokkos_BUILD_DIR}${ColorReset} "\n")
endif()

if(${output})
  string(APPEND REPORT_TEXT "  - ADIOS2: v" ${adios2_VERSION} "\n")
  if(${adios2_FOUND})
    string(APPEND REPORT_TEXT "    " "${Dim}${adios2_DIR}${ColorReset}" "\n")
  else()
    string(APPEND REPORT_TEXT "    " "${Dim}${adios2_BUILD_DIR}${ColorReset}"
           "\n")
  endif()
endif()

string(
  APPEND
  REPORT_TEXT
  ${DASHED_LINE_SYMBOL}
  "\n"
  "Notes"
  "\n"
  "    ${Dim}: Set flags with `cmake ... -D "
  "${Magenta}<FLAG>${ColorReset}${Dim}=<VALUE>`, "
  "the ${Underline}default${ColorReset}${Dim} value"
  "\n"
  "    :   will be used unless the variable is explicitly set.${ColorReset}")

if(${TESTS})
  string(
    APPEND
    REPORT_TEXT
    "\n"
    "    ${Dim}: Run the tests with the following command:"
    "\n"
    "    :   ctest --test-dir ${CMAKE_CURRENT_BINARY_DIR}${ColorReset}"
    "\n")
endif()

string(APPEND REPORT_TEXT "\n")

message(${REPORT_TEXT})
