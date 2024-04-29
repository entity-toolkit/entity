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

if(${PGEN_FOUND})
  PrintChoices("Problem generator"
    "pgen"
    "${problem_generators}"
    ${PGEN}
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
PrintChoices("GUI"
  "gui"
  "${ON_OFF_VALUES}"
  ${gui}
  ${default_gui}
  "${Green}"
  GUI_REPORT
  0
  36
)
PrintChoices("MPI"
  "mpi"
  "${ON_OFF_VALUES}"
  ${mpi}
  OFF
  "${Green}"
  MPI_REPORT
  0
  42
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
  "${ColorReset}"
  CXX_COMPILER_REPORT
  0
  42
)

PrintChoices("C compiler"
  "CMAKE_C_COMPILER"
  "${CMAKE_C_COMPILER} v${CMAKE_C_COMPILER_VERSION}"
  "${CMAKE_C_COMPILER} v${CMAKE_C_COMPILER_VERSION}"
  "N/A"
  "${ColorReset}"
  C_COMPILER_REPORT
  0
  42
)

if(${Kokkos_ENABLE_CUDA})
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
    "${ColorReset}"
    CUDA_COMPILER_REPORT
    0
    42
  )
endif()

set(DOT_SYMBOL "${ColorReset}.")
set(DOTTED_LINE_SYMBOL "${ColorReset}. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ")

set(DASHED_LINE_SYMBOL "${ColorReset}....................................................................... ")

if(NOT ${PROJECT_VERSION_TWEAK} EQUAL 0)
  set(VERSION_SYMBOL "v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-rc${PROJECT_VERSION_TWEAK}")
else()
  set(VERSION_SYMBOL "v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}    ")
endif()

message("${Blue}              __        __
             /\\ \\__  __/\\ \\__
    __    ___\\ \\  _\\/\\_\\ \\  _\\  __  __
  / __ \\ / __ \\ \\ \\/\\/\\ \\ \\ \\/ /\\ \\/\\ \\
 /\\  __//\\ \\/\\ \\ \\ \\_\\ \\ \\ \\ \\_\\ \\ \\_\\ \\  __
 \\ \\____\\ \\_\\ \\_\\ \\__\\\\ \\_\\ \\__\\\\ \\____ \\/\\_\\
  \\/____/\\/_/\\/_/\\/__/ \\/_/\\/__/ \\/___/  \\/_/
                                    /\\___/
Entity ${VERSION_SYMBOL}\t\t    \\/__/")
message("${DASHED_LINE_SYMBOL}
Main configurations")

if(${PGEN_FOUND})
  message("  ${PGEN_REPORT}")
endif()

message("  ${PRECISION_REPORT}")
message("  ${OUTPUT_REPORT}")
message("${DASHED_LINE_SYMBOL}
Compile configurations")

message("  ${CUDA_REPORT}")
message("  ${OPENMP_REPORT}")

message("  ${C_COMPILER_REPORT}")

message("  ${CXX_COMPILER_REPORT}")

if(NOT "${CUDA_COMPILER_REPORT}" STREQUAL "")
  message("  ${CUDA_COMPILER_REPORT}")
endif()

message("  ${MPI_REPORT}")

message("  ${DEBUG_REPORT}")

message("${DASHED_LINE_SYMBOL}\nDependencies")

message("  - Kokkos:\tv${Kokkos_VERSION}")
if(${output})
  message("  - ADIOS2:\tv${adios2_VERSION}")
endif()
if(${HDF5_FOUND})
  message("  - HDF5:\tv${HDF5_VERSION}")
endif()

message("${DASHED_LINE_SYMBOL}
Notes
   ${Dim}: Set flags with `cmake ... -D ${Magenta}<FLAG>${ColorReset}${Dim}=<VALUE>`, the ${Underline}default${ColorReset}${Dim} value
   :   will be used unless the variable is explicitly set.${ColorReset}
")
