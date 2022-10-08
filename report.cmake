if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

if (${ENABLE_OUTPUT} STREQUAL "ON")
  set(OUTPUT_COLOR ${Green})
else()
  set(OUTPUT_COLOR ${Red})
endif()

if (${DEBUG} STREQUAL "ON")
  set(DEBUG_COLOR ${Green})
else()
  set(DEBUG_COLOR ${Red})
endif()

if (${Kokkos_ENABLE_CUDA} STREQUAL "ON")
  set(CUDA_COLOR ${Green})
else()
  set(CUDA_COLOR ${Red})
endif()

if (${Kokkos_ENABLE_OPENMP} STREQUAL "ON")
  set(OPENMP_COLOR ${Green})
else()
  set(OPENMP_COLOR ${Red})
endif()

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

Main `entity` configurations
------------------------------------------------------
  Simulation type:\t${Green}${simtype}${ColourReset}
  Metric:\t\t${Green}${metric}${ColourReset}
  Problem generator:\t${Green}${pgen}${ColourReset}
  Precision:\t\t${Green}${precision}${ColourReset}
  Output:\t\t${OUTPUT_COLOR}${ENABLE_OUTPUT}${ColourReset}

Framework configurations
------------------------------------------------------
  Debug mode:\t\t${DEBUG_COLOR}${DEBUG}${ColourReset}
  Main framework:\t${Green}Kokkos${ColourReset}
  CUDA:\t\t\t${CUDA_COLOR}${Kokkos_ENABLE_CUDA}${ColourReset}
  OpenMP:\t\t${OPENMP_COLOR}${Kokkos_ENABLE_OPENMP}${ColourReset}

======================================================
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