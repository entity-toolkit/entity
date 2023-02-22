if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold "${Esc}[1m")
  set(Underline "${Esc}[4m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
  set(BoldRed "${Esc}[1;31m")
  set(BoldGreen "${Esc}[1;32m")
  set(BoldYellow "${Esc}[1;33m")
  set(BoldBlue "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan "${Esc}[1;36m")
  set(BoldWhite "${Esc}[1;37m")
  set(DarkGray "${Esc}[1;90m")
  set(Dim "${Esc}[2m")
  set(StrikeBegin "${Esc}[9m")
  set(StrikeEnd "${Esc}[0m")
endif()

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
        set(rstring_i "${rstring_i} [${Magenta}${Flag}${ColourReset}]")
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

    # disabled options
    if("${Flag}" STREQUAL "engine")
      if(${ch} STREQUAL "grpic")
        set(ch ${StrikeBegin}grpic${StrikeEnd})
      endif()
    endif()

    set(rstring_i "${rstring_i}${col}${ch}${ColourReset}")
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
  ${default_DEBUG}
  "${Green}"
  DEBUG_REPORT
  0
  42
)

if(${kokkos_FOUND})
  get_directory_property(KOKKOS_VERSION
    DIRECTORY ${kokkos_ROOT}
    DEFINITION Kokkos_VERSION)
  PrintChoices("Kokkos"
    ""
    "v${KOKKOS_VERSION}"
    "v${KOKKOS_VERSION}"
    "N/A"
    "${Blue}"
    FRAMEWORK_REPORT
    0
    39
  )
  get_directory_property(ENABLED_ARCHS
    DIRECTORY ${kokkos_ROOT}
    DEFINITION KOKKOS_ENABLED_ARCH_LIST)
  PrintChoices("CPU/GPU architecture"
    ""
    "${ENABLED_ARCHS}"
    "${ENABLED_ARCHS}"
    "N/A"
    "${Blue}"
    ARCH_REPORT
    0
    39
  )
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
endif()

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

if(${Kokkos_ENABLE_CUDA})
  # check if empty
  if("${CMAKE_CUDA_COMPILER}" STREQUAL "")
    execute_process(COMMAND which nvcc OUTPUT_VARIABLE CUDACOMP)
  else()
    set(CUDACOMP ${CMAKE_CUDA_COMPILER})
  endif()

  message(STATUS "CUDA compiler: ${CUDACOMP}")
  execute_process(COMMAND bash "-c"
    "${CUDACOMP} --version | grep release | sed -e 's/.*release //' -e 's/,.*//'"
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

  # set(COMPILERS_REPORT "${COMPILERS_REPORT}\n\n  ${CUDA_COMPILER_REPORT}")
endif()

set(DOT_SYMBOL "${ColourReset}.")
set(DOTTED_LINE_SYMBOL "${ColourReset}. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ")

set(DASHED_LINE_SYMBOL "${ColourReset}- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")

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
Main configurations ${Dim}[1]${ColourReset}
${DASHED_LINE_SYMBOL}")
message("  ${ENGINE_REPORT}\n")
message("  ${METRIC_REPORT}\n")
message("  ${PGEN_REPORT}\n")
message("  ${PRECISION_REPORT}\n")
message("  ${OUTPUT_REPORT}\n")
message("  ${NTTINY_REPORT}\n")
message("${DASHED_LINE_SYMBOL}
Framework configurations
${DASHED_LINE_SYMBOL}")
message("  ${DEBUG_REPORT}\n")

if(${kokkos_FOUND})
  message("  ${FRAMEWORK_REPORT}\n")

  if(NOT "${ARCH_REPORT}" STREQUAL "")
    message("  ${ARCH_REPORT}\n")
  endif()

  message("  ${CUDA_REPORT}\n")
  message("  ${OPENMP_REPORT}\n")
else()
  message("  - Kokkos [${Magenta}kokkos_ROOT${ColourReset}]:\t\t  ${kokkos_ROOT}\n")
endif()

if(${adios2_FOUND})
else()
  message("  - ADIOS2 [${Magenta}adios2_ROOT${ColourReset}]:\t\t  ${adios2_ROOT}\n")
endif()

message("  ${CXX_COMPILER_REPORT}\n")

if(NOT "${CUDA_COMPILER_REPORT}" STREQUAL "")
  message("  ${CUDA_COMPILER_REPORT}\n")
endif()

message("${DASHED_LINE_SYMBOL}
Notes
${DASHED_LINE_SYMBOL}
  ${Dim}[1] Set with `cmake ... -D ${Magenta}<FLAG>${ColourReset}${Dim}=<VALUE>`, the ${Underline}default${ColourReset}${Dim} value
   :  will be used unless the variable is explicitly set.${ColourReset}
")

# message("This is normal")
# message("${Red}This is Red${ColourReset}")
# message("${Green}This is Green${ColourReset}")
# message("${Yellow}This is Yellow${ColourReset}")
# message("${Blue}This is Blue${ColourReset}")
# message("${Magenta}This is Magenta${ColourReset}")
# message("${Cyan}This is Cyan${ColourReset}")
# message("${White}This is White${ColourReset}")
# message("${BoldRed}This is BoldRed${ColourReset}")
# message("${BoldGreen}This is BoldGreen${ColourReset}")
# message("${BoldYellow}This is BoldYellow${ColourReset}")
# message("${BoldBlue}This is BoldBlue${ColourReset}")
# message("${BoldMagenta}This is BoldMagenta${ColourReset}")
# message("${BoldCyan}This is BoldCyan${ColourReset}")
# message("${BoldWhite}This is BoldWhite\n\n${ColourReset}")

# message()