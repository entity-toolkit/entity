function(PadTo Text Padding Target Result)
  set(rt ${Text})
  string(FIND ${rt} "${Magenta}" mg_fnd)

  if(mg_fnd GREATER -1)
    string(REGEX REPLACE "${Esc}\\[35m" "" rt ${rt})
  endif()

  string(LENGTH "${rt}" TextLength)
  math(EXPR PaddingNeeded "${Target} - ${TextLength}")
  set(rt ${Text})

  if(PaddingNeeded GREATER 0)
    foreach(i RANGE 0 ${PaddingNeeded})
      set(rt "${rt}${Padding}")
    endforeach()
  else()
    set(${rt} "${rt}")
  endif()

  set(${Result} "${rt}" PARENT_SCOPE)
endfunction()

function(PrintChoices Label Flag Choices Value Default Color OutputString Multiline Padding)
  list(LENGTH "${Choices}" nchoices)
  set(rstring "")
  set(counter 0)

  foreach(ch ${Choices})
    if(${counter} EQUAL 0)
      set(rstring_i "- ${Label}")

      if(NOT "${Flag}" STREQUAL "")
        set(rstring_i "${rstring_i} [${Magenta}${Flag}${ColorReset}]")
      endif()

      set(rstring_i "${rstring_i}:")
      PadTo("${rstring_i}" " " ${Padding} rstring_i)
    else()
      set(rstring_i "")

      if(NOT ${counter} EQUAL ${nchoices})
        if(${Multiline} EQUAL 1)
          set(rstring_i "${rstring_i}\n")
          PadTo("${rstring_i}" " " ${Padding} rstring_i)
        else()
          set(rstring_i "${rstring_i}/")
        endif()
      endif()
    endif()

    if(${ch} STREQUAL ${Value})
      if(${ch} STREQUAL "ON")
        set(col ${Green})
      elseif(${ch} STREQUAL "OFF")
        set(col ${Red})
      else()
        set(col ${Color})
      endif()
    else()
      set(col ${Dim})
    endif()

    if(${ch} STREQUAL ${Default})
      set(col ${Underline}${col})
    endif()

    set(rstring_i "${rstring_i}${col}${ch}${ColorReset}")
    math(EXPR counter "${counter} + 1")
    set(rstring "${rstring}${rstring_i}")
    set(rstring_i "")
  endforeach()

  set(${OutputString} "${rstring}" PARENT_SCOPE)
endfunction()

set(ON_OFF_VALUES "ON" "OFF")

PrintChoices("Simulation engine"
  "engine"
  "${simulation_engines}"
  ${engine}
  ${default_engine}
  "${Blue}"
  ENGINE_REPORT
  1
  36
)
PrintChoices("Metric"
  "metric"
  "${metrics}"
  ${metric}
  ${default_metric}
  "${Blue}"
  METRIC_REPORT
  1
  36
)

if(${PGEN_FOUND})
  PrintChoices("Problem generator"
    "pgen"
    "${problem_generators}"
    ${pgen}
    ${default_pgen}
    "${Blue}"
    PGEN_REPORT
    1
    36
  )
endif()

PrintChoices("Precision"
  "precision"
  "${precisions}"
  ${precision}
  ${default_precision}
  "${Blue}"
  PRECISION_REPORT
  1
  36
)
PrintChoices("External force"
  "ext_force"
  "${ON_OFF_VALUES}"
  ${ext_force}
  ${default_ext_force}
  "${Green}"
  EXT_FORCE_REPORT
  0
  36
)

PrintChoices("Output"
  "output"
  "${ON_OFF_VALUES}"
  ${output}
  ${default_output}
  "${Green}"
  OUTPUT_REPORT
  0
  36
)
PrintChoices("nttiny GUI"
  "nttiny"
  "${ON_OFF_VALUES}"
  ${nttiny}
  ${default_nttiny}
  "${Green}"
  NTTINY_REPORT
  0
  36
)
PrintChoices("Debug mode"
  "DEBUG"
  "${ON_OFF_VALUES}"
  ${DEBUG}
  OFF
  "${Green}"
  DEBUG_REPORT
  0
  42
)

# get_directory_property(ENABLED_ARCHS
# DIRECTORY ${kokkos_ROOT}/lib/cmake/Kokkos/
# DEFINITION Kokkos_ARCH)
# string(REPLACE ";" " + " ARCHS "${ENABLED_ARCHS}")
# message(STATUS "ARCHS: ${ARCHS}")
# PrintChoices("CPU/GPU architecture"
# ""
# "${ARCHS}"
# "${ARCHS}"
# "N/A"
# "${White}"
# ARCH_REPORT
# 0
# 39
# )
# if(${output})
# if(NOT DEFINED adios2_VERSION OR adios2_VERSION STREQUAL "")
# get_directory_property(adios2_VERSION
# DIRECTORY ${adios2_BUILD_DIR}
# DEFINITION ADIOS2_VERSION)
# endif()
# endif()
PrintChoices("CUDA"
  "Kokkos_ENABLE_CUDA"
  "${ON_OFF_VALUES}"
  ${Kokkos_ENABLE_CUDA}
  "OFF"
  "${Green}"
  CUDA_REPORT
  0
  42
)
PrintChoices("OpenMP"
  "Kokkos_ENABLE_OPENMP"
  "${ON_OFF_VALUES}"
  ${Kokkos_ENABLE_OPENMP}
  "OFF"
  "${Green}"
  OPENMP_REPORT
  0
  42
)

PrintChoices("C++ compiler"
  "CMAKE_CXX_COMPILER"
  "${CMAKE_CXX_COMPILER} v${CMAKE_CXX_COMPILER_VERSION}"
  "${CMAKE_CXX_COMPILER} v${CMAKE_CXX_COMPILER_VERSION}"
  "N/A"
  "${White}"
  CXX_COMPILER_REPORT
  0
  42
)

PrintChoices("C compiler"
  "CMAKE_C_COMPILER"
  "${CMAKE_C_COMPILER} v${CMAKE_C_COMPILER_VERSION}"
  "${CMAKE_C_COMPILER} v${CMAKE_C_COMPILER_VERSION}"
  "N/A"
  "${White}"
  C_COMPILER_REPORT
  0
  42
)

if(${Kokkos_ENABLE_CUDA})
  # check if empty
  if("${CMAKE_CUDA_COMPILER}" STREQUAL "")
    execute_process(COMMAND which nvcc OUTPUT_VARIABLE CUDACOMP)
  else()
    set(CUDACOMP ${CMAKE_CUDA_COMPILER})
  endif()

  string(STRIP ${CUDACOMP} CUDACOMP)

  message(STATUS "CUDA compiler: ${CUDACOMP}")
  execute_process(COMMAND bash -c "${CUDACOMP} --version | grep release | sed -e 's/.*release //' -e 's/,.*//'"

    OUTPUT_VARIABLE CUDACOMP_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  PrintChoices("CUDA compiler"
    "CMAKE_CUDA_COMPILER"
    "${CUDACOMP} v${CUDACOMP_VERSION}"
    "${CUDACOMP} v${CUDACOMP_VERSION}"
    "N/A"
    "${White}"
    CUDA_COMPILER_REPORT
    0
    42
  )
endif()

set(DOT_SYMBOL "${ColorReset}.")
set(DOTTED_LINE_SYMBOL "${ColorReset}. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ")

set(DASHED_LINE_SYMBOL "${ColorReset}- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")

if(NOT ${PROJECT_VERSION_TWEAK} EQUAL 0)
  set(VERSION_SYMBOL "v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-rc${PROJECT_VERSION_TWEAK}")
else()
  set(VERSION_SYMBOL "v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}    ")
endif()

message("
${DOTTED_LINE_SYMBOL}
${DOT_SYMBOL}${Blue}                          __        __                               ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}                         /\\ \\__  __/\\ \\__                            ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}                __    ___\\ \\  _\\/\\_\\ \\  _\\  __  __                   ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}              / __ \\ / __ \\ \\ \\/\\/\\ \\ \\ \\/ /\\ \\/\\ \\                  ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}             /\\  __//\\ \\/\\ \\ \\ \\_\\ \\ \\ \\ \\_\\ \\ \\_\\ \\  __             ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}             \\ \\____\\ \\_\\ \\_\\ \\__\\\\ \\_\\ \\__\\\\ \\____ \\/\\_\\            ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}              \\/____/\\/_/\\/_/\\/__/ \\/_/\\/__/ \\/___/  \\/_/            ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}                                                /\\___/               ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}                                                \\/__/                ${DOT_SYMBOL}
${DOT_SYMBOL}                                                                     ${DOT_SYMBOL}
${DOT_SYMBOL}${Blue}                              ${VERSION_SYMBOL}                             ${DOT_SYMBOL}
${DOTTED_LINE_SYMBOL}")
message("${DASHED_LINE_SYMBOL}
Main configurations ${Dim}[1]${ColorReset}
${DASHED_LINE_SYMBOL}")
message("  ${ENGINE_REPORT}\n")
message("  ${METRIC_REPORT}\n")

if(${PGEN_FOUND})
  message("  ${PGEN_REPORT}\n")
endif()

message("  ${EXT_FORCE_REPORT}\n")

message("  ${PRECISION_REPORT}\n")
message("  ${OUTPUT_REPORT}\n")
message("  ${NTTINY_REPORT}\n")
message("${DASHED_LINE_SYMBOL}
Framework configurations
${DASHED_LINE_SYMBOL}")

# if(NOT ${Kokkos_FETCHED})
# string(REPLACE ${CMAKE_SOURCE_DIR}/ "./" Kokkos_ROOT_rel "${Kokkos_DIR}")
# message("  - Kokkos [${Magenta}Kokkos_DIR${ColorReset}]:\t\t  ${Kokkos_ROOT_rel} v${Kokkos_VERSION}\n")
# else()
message("  - Kokkos:\t\t\t\t  v${Kokkos_VERSION}\n")

# endif()

# if(NOT "${adios2_DIR}" STREQUAL "" AND ${output} STREQUAL "ON")
# string(REPLACE ${CMAKE_SOURCE_DIR}/ "./" adios2_ROOT_rel "${adios2_DIR}")
# message("  - ADIOS2 [${Magenta}adios2_DIR${ColorReset}]:\t\t  ${adios2_DIR} v${adios2_VERSION}\n")
if(${output})
  message("  - ADIOS2:\t\t\t\t  v${adios2_VERSION}\n")
endif()

# endif()

# if(ENABLED_ARCHS)
# message("  ${ARCH_REPORT}\n")
# endif()
message("  ${CUDA_REPORT}\n")
message("  ${OPENMP_REPORT}\n")

message("  ${C_COMPILER_REPORT}\n")

message("  ${CXX_COMPILER_REPORT}\n")

if(NOT "${CUDA_COMPILER_REPORT}" STREQUAL "")
  message("  ${CUDA_COMPILER_REPORT}\n")
endif()

message("  ${DEBUG_REPORT}\n")

# message("  ${FRAMEWORK_REPORT}\n")
message("${DASHED_LINE_SYMBOL}
Notes
${DASHED_LINE_SYMBOL}
  ${Dim}[1] Set with `cmake ... -D ${Magenta}<FLAG>${ColorReset}${Dim}=<VALUE>`, the ${Underline}default${ColorReset}${Dim} value
   :  will be used unless the variable is explicitly set.${ColorReset}
")
