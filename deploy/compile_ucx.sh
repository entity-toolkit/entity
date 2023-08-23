#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

source ${SCRIPT_DIR}/aux/aux.sh

declare -r modulename="UCX"
source ${SCRIPT_DIR}/aux/default.sh
source ${SCRIPT_DIR}/aux/globals.sh

function usage {
  common_help
}

source ${SCRIPT_DIR}/aux/argparse.sh
source ${SCRIPT_DIR}/aux/config.sh

if [ $enable_cuda = "ON" ]; then
  install_path="${install_path}/cuda"
else
  install_path="${install_path}/cpu"
fi

compile_args=(
  --prefix=$install_path
)

if [ $use_modules = "ON" ]; then
  if [ $enable_cuda = "ON" ]; then
    compile_args+=(
      --with-cuda=\$CUDA_HOME
    )
  fi
else
  if [ $enable_cuda = "ON" ]; then
    compile_args+=(
      --with-cuda=$cuda_path
    )
  fi
fi

source ${SCRIPT_DIR}/aux/run.sh

function prebuild {
  if [ $use_modules = "ON" ]; then
    runcommand "module purge"
    runcommand "module load $cc_module"
    runcommand "export CC=\$(which gcc) CXX=\$(which g++)"
    if [ $enable_cuda = "ON" ]; then
      runcommand "module load $cuda_module"
    fi
  fi
}

function configure {
  prebuild
  runcommand "cd $ucx_src_path"
  runcommand "rm -rf build"
  runcommand "./autogen.sh"
  runcommand "mkdir build"
  runcommand "cd build"
  local args=$(printf " %s" "${compile_args[@]}")
  runcommand "../configure$args"
}

function compile {
  prebuild
  runcommand "cd $ucx_src_path/build"
  runcommand "make -j"
}

function install {
  runcommand "cd $ucx_src_path/build"
  runcommand "make install"
}

function cleanup {
  runcommand "cd $ucx_src_path"
  runcommand "rm -rf build"
}

run configure compile install cleanup
