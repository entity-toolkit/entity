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
endif()

function(PrintChoices Choices Value Default Color OutputString Multiline)
  list(LENGTH "${Choices}" nchoices)
  set(rstring "")
  set(counter 0)

  foreach(ch ${Choices})
    if(NOT ${counter} EQUAL ${nchoices})
      if(${Multiline} EQUAL 1)
        set(rstring "${rstring}\n\t\t\t\t")
      else()
        set(rstring "${rstring}/")
      endif()

    else()
      set(rstring "${rstring}")
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
      set(col ${col}${Underline})
    endif()
    set(rstring "${rstring}${col}${ch}${ColourReset}")

    math(EXPR counter "${counter} + 1")
  endforeach()

  set(${OutputString} "${rstring}" PARENT_SCOPE)
endfunction()

set(ON_OFF_VALUES "ON" "OFF")

PrintChoices("${ON_OFF_VALUES}" ${output} ${default_output} "${Green}" OUTPUT_REPORT 0)
PrintChoices("${simulation_types}" ${simtype} ${default_simtype} "${Blue}" SIMTYPE_REPORT 1)
PrintChoices("${metrics}" ${metric} ${default_metric} "${Blue}" METRIC_REPORT 1)
PrintChoices("${problem_generators}" ${pgen} ${default_pgen} "${Blue}" PGEN_REPORT 1)

PrintChoices("${precisions}" ${precision} ${default_precision} "${Blue}" PRECISION_REPORT 1)
PrintChoices("${ON_OFF_VALUES}" ${nttiny} ${default_nttiny} "${Green}" NTTINY_REPORT 0)
PrintChoices("${ON_OFF_VALUES}" ${DEBUG} ${default_DEBUG} "${Green}" DEBUG_REPORT 0)

PrintChoices("${ON_OFF_VALUES}" ${Kokkos_ENABLE_CUDA} "OFF" "${Green}" CUDA_REPORT 0)
PrintChoices("${ON_OFF_VALUES}" ${Kokkos_ENABLE_OPENMP} "OFF" "${Green}" OPENMP_REPORT 0)

message("
======================================================
${ColourReset}.${Blue}                 __        __                       ${ColourReset}.
${ColourReset}.${Blue}                /\\ \\__  __/\\ \\__                    ${ColourReset}.
${ColourReset}.${Blue}       __    ___\\ \\  _\\/\\_\\ \\  _\\  __  __           ${ColourReset}.
${ColourReset}.${Blue}     / __ \\ / __ \\ \\ \\/\\/\\ \\ \\ \\/ /\\ \\/\\ \\          ${ColourReset}.
${ColourReset}.${Blue}    /\\  __//\\ \\/\\ \\ \\ \\_\\ \\ \\ \\ \\_\\ \\ \\_\\ \\  __     ${ColourReset}.
${ColourReset}.${Blue}    \\ \\____\\ \\_\\ \\_\\ \\__\\\\ \\_\\ \\__\\\\ \\____ \\/\\_\\    ${ColourReset}.
${ColourReset}.${Blue}     \\/____/\\/_/\\/_/\\/__/ \\/_/\\/__/ \\/___/  \\/_/    ${ColourReset}.
${ColourReset}.${Blue}                                       /\\___/       ${ColourReset}.
${ColourReset}.${Blue}                                       \\/__/        ${ColourReset}.
.                                                    .
${ColourReset}.${Blue}                     v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}                         ${ColourReset}.
======================================================

Main `entity` configurations ${Dim}[1]${ColourReset}
------------------------------------------------------
  Simulation type [${Magenta}simtype${ColourReset}]:\t${SIMTYPE_REPORT}
  Metric [${Magenta}metric${ColourReset}]:\t\t${METRIC_REPORT}
  Problem generator [${Magenta}pgen${ColourReset}]:\t${PGEN_REPORT}
  Precision [${Magenta}precision${ColourReset}]:\t${PRECISION_REPORT}
  Output [${Magenta}output${ColourReset}]:\t\t${OUTPUT_REPORT}
  nttiny GUI [${Magenta}nttiny${ColourReset}]:\t\t${NTTINY_REPORT}

Framework configurations
------------------------------------------------------
  Debug mode [${Magenta}DEBUG${ColourReset}]:\t\t\t${DEBUG_REPORT}
  Main framework:\t\t\t${Blue}Kokkos${ColourReset}
  CUDA [${Magenta}Kokkos_ENABLE_CUDA${ColourReset}]:\t\t${CUDA_REPORT}
  OpenMP [${Magenta}Kokkos_ENABLE_OPENMP${ColourReset}]:\t${OPENMP_REPORT}

======================================================

Notes

  ${Dim}[1] Set with `cmake ... -D ${Magenta}<FLAG>${ColourReset}${Dim}=<VALUE>`,...
  * ... default (${Underline}underlined${ColourReset}${Dim}) value will be used...
  * ... unless a variable is explicitly set.${ColourReset}

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