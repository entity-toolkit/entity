#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

source ${SCRIPT_DIR}/aux/aux.sh

declare -r modulename="OMPI"
declare -r has_modulefile="ON"

default_ucx_path="${HOME}/opt/ucx"
declare with_ucx="${default_ucx_path}"

source ${SCRIPT_DIR}/aux/default.sh
source ${SCRIPT_DIR}/aux/globals.sh

function usage {
  common_help
  echo "  --with-ucx <path>              enable UCX support (specify installation path)"
  echo "                                 set to OFF to disable UCX support"
  echo "                                 (default: ${default_ucx_path})"
  echo ""
}

source ${SCRIPT_DIR}/aux/argparse.sh
source ${SCRIPT_DIR}/aux/config.sh

if [ $enable_cuda = "ON" ]; then
  install_path="${install_path}/cuda"
  ompi_module="${ompi_module}/cuda"
else
  install_path="${install_path}/cpu"
  ompi_module="${ompi_module}/cpu"
fi

compile_args=(
  --prefix=${install_path}
  --with-devel-headers
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
      --with-cuda=$with_cuda
    )
  fi
fi

if [ ! $with_ucx = "OFF" ]; then
  if [ $enable_cuda = "ON" ]; then
    compile_args+=(
      --with-ucx=$with_ucx/cuda
    )
  else
    compile_args+=(
      --with-ucx=$with_ucx/cpu
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
  runcommand "cd $ompi_src_path"
  runcommand "rm -rf build"
  runcommand "./autogen.pl"
  runcommand "mkdir build"
  runcommand "cd build"
  local args=$(printf " %s" "${compile_args[@]}")
  runcommand "../configure$args"
}

function compile {
  prebuild
  runcommand "cd $ompi_src_path/build"
  runcommand "make -j"
}

function install {
  runcommand "cd $ompi_src_path/build"
  runcommand "make install"
}

function cleanup {
  runcommand "cd $ompi_src_path"
  runcommand "rm -rf build"
}

function report {
  REPORT_VARS+=(
    "UCX"
  )
  REPORT_VALS+=(
    "${with_ucx}"
  )
}

function modulefile {
  fname=$ompi_module
  description="Open MPI"
  if [ $enable_cuda = "ON" ]; then
    description=$description" @ CUDA"
  fi
  prereqs=""
  if [ $use_modules = "ON" ]; then
    prereqs+="prereq\t\t$cc_module"
  fi
  runcommand "mkdir -p $(dirname $fname)"
  runcommand "rm -f $fname"
  runcommand "echo \"Writing modulefile to $fname\""
  modulecontent='''#%Module1.0######################################################################
##
## $description
##
proc ModulesHelp { } {
    puts stderr \t\"$description\"\n
}
module-whatis   \"$description\"

conflict        ompi openmpi
$prereqs

set             basedir               $install_path
prepend-path    PATH                  \$basedir/bin
prepend-path    LD_LIBRARY_PATH       \$basedir/lib

append-path -d { } LOCAL_LDFLAGS      -L\$basedir/lib
append-path -d { } LOCAL_INCLUDE      -I\$basedir/include
append-path -d { } LOCAL_CFLAGS       -I\$basedir/include
append-path -d { } LOCAL_FCFLAGS      -I\$basedir/include
append-path -d { } LOCAL_CXXFLAGS     -I\$basedir/include

setenv          CXX                   \$basedir/bin/mpicxx
setenv          CC                    \$basedir/bin/mpicc

setenv          SLURM_MPI_TYPE        pmix_v3
setenv          MPIHOME               \$basedir
setenv          MPI_HOME              \$basedir
setenv          OPENMPI_HOME          \$basedir
    '''
  runcommand "echo -e \"$modulecontent\" >>$fname"
}

run configure compile install cleanup modulefile report
