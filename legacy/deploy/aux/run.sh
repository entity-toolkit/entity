declare -x REPORT_VARS=()
declare -x REPORT_VALS=()

function common_report {
  # Report cuda
  if [ $enable_cuda = "ON" ]; then
    if [ $use_modules = "ON" ]; then
      REPORT_VALS=(
        "module:${cuda_module}"
        "${REPORT_VALS[@]}"
      )
    else
      REPORT_VALS=(
        "${with_cuda}"
        "${REPORT_VALS[@]}"
      )
    fi
    REPORT_VARS=(
      "CUDA"
      "${REPORT_VARS[@]}"
    )
  fi

  # Report C compiler
  REPORT_VARS=(
    "C compiler"
    "${REPORT_VARS[@]}"
  )
  if [ $use_modules = "ON" ]; then
    REPORT_VALS=(
      "module:${cc_module}"
      "${REPORT_VALS[@]}"
    )
  else
    local cc=$(which gcc)
    REPORT_VALS=(
      "${cc}"
      "${REPORT_VALS[@]}"
    )
  fi

  # Report modulefile path
  if [ $writing_modulefile != "OFF" ]; then
    local module_dir=${modulename_lower}_module
    REPORT_VARS=(
      "modulefile path"
      "${REPORT_VARS[@]}"
    )
    REPORT_VALS=(
      "${!module_dir}"
      "${REPORT_VALS[@]}"
    )
  fi

  # Report src & install directory
  local src_path=${modulename_lower}_src_path
  REPORT_VARS=(
    "src directory"
    "install directory" "${REPORT_VARS[@]}"
  )
  REPORT_VALS=(
    "${!src_path}"
    "${install_path}" "${REPORT_VALS[@]}"
  )

  for i in "${!REPORT_VARS[@]}"; do
    printf "    %-25s%s\n" "${REPORT_VARS[i]}:" "${REPORT_VALS[i]}"
  done
  echo ""
}

run() {
  local configure="$1"
  local build="$2"
  local install="$3"
  local cleanup="$4"
  if [ $writing_modulefile != "OFF" ]; then
    if [[ -z $5 ]]; then
      printf "\n${RED}Missing modulefile function${NC}\n"
      exit 1
    else
      modulefile="$5"
    fi
    if [[ -z $6 ]]; then
      printf "\n${RED}Missing report function${NC}\n"
      exit 1
    else
      report="$6"
    fi
  else
    if [[ -z $5 ]]; then
      report=""
    else
      report="$5"
    fi
  fi

  printf "Installing ${BLUE}${modulename}${NC}\n"
  eval $report
  common_report

  set_spinner spinner1
  declare -x STEPS=(
    'configure'
    'build'
    'install'
    'cleanup'
  )
  declare -x CMDS=(
    'eval $configure'
    'eval $build'
    'eval $install'
    'eval $cleanup'
  )
  if [ $writing_modulefile != "OFF" ]; then
    STEPS+=(
      'modulefile'
    )
    CMDS+=(
      'eval $modulefile'
    )
  fi

  local step=0
  rm -f $logfile

  tput civis -- invisible

  while [ "$step" -lt "${#CMDS[@]}" ]; do
    ${CMDS[$step]} &
    pid=$!

    if [ $verbose = "OFF" ]; then
      while ps -p $pid &>/dev/null; do
        echo -ne "\\r[   ] ${STEPS[$step]}"

        for k in "${!FRAME[@]}"; do
          echo -ne "\\r[ ${FRAME[k]} ]"
          sleep $FRAME_INTERVAL
        done
      done
    fi
    wait $pid
    local exitcode=$?

    if [ $exitcode -eq 0 ]; then
      if [ $deploy = "OFF" ]; then
        # draw up arrow
        echo -ne "\\r[ ${BLUE}↑${NC} ] ${STEPS[$step]}\\n"
        echo ""
      else
        echo -ne "\\r[ ${GREEN}✔${NC} ] ${STEPS[$step]}\\n"
      fi
    else
      echo -ne "\\r[ ${RED}✘${NC} ] ${STEPS[$step]}\\n"
      echo "Failed to install ${modulename} :("
      echo "see ${logfile} for more details"
      exit 1
    fi
    step=$((step + 1))
  done

  tput cnorm -- normal

  if [ $deploy = "ON" ]; then
    echo
    printf "${BLUE}${modulename}${NC} succesfully installed in ${install_prefix}/${modulename_lower}!\n"
  else
    echo
    printf "now run \`${BLUE}bash ${programname} -d${NC}\` to execute the script\n"
  fi
}
