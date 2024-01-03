#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

source ${SCRIPT_DIR}/aux/aux.sh

declare -r modulename="HDF5"
declare -r has_modulefile="ON"

default_mpi_path="module:ompi"
declare with_mpi="${default_mpi_path}"

source ${SCRIPT_DIR}/aux/default.sh
source ${SCRIPT_DIR}/aux/globals.sh

function usage {
  common_help
  echo "  --with-mpi <path|module>       MPI path or modulename (\`module:<modulename>\`)"
  echo "                                 set to OFF to disable MPI support"
  echo "                                 (default: ${default_mpi_path})"
  echo ""
}

source ${SCRIPT_DIR}/aux/argparse.sh
source ${SCRIPT_DIR}/aux/config.sh

if [ ! $with_mpi = "OFF" ]; then
  hdf5_module+="/mpi"
  install_path+="/mpi"
  if [ $enable_cuda = "ON" ]; then
    hdf5_module+="/cuda"
    install_path+="/cuda"
  else
    hdf5_module+="/cpu"
    install_path+="/cpu"
  fi
else
  hdf5_module+="/serial"
  install_path+="/serial"
fi

if [ ! $with_mpi = "OFF" ]; then
  compile_args=(
    -S HDF5config.cmake,HPC=sbatch,MPI=true,BUILD_GENERATOR=Unix,INSTALLDIR=$install_path
  )
else
  compile_args=(
    -S HDF5config.cmake,HPC=sbatch,BUILD_GENERATOR=Unix,INSTALLDIR=$install_path
  )
fi
compile_args+=(
  -C Release
  -V
  -O hdf5.log
)

source ${SCRIPT_DIR}/aux/run.sh

function prebuild {
  if [ $use_modules = "ON" ]; then
    runcommand "module purge"
    runcommand "module load $cc_module"
    if [ $enable_cuda = "ON" ]; then
      runcommand "module load $cuda_module"
    fi
    if [ ! $with_mpi = "OFF" ]; then
      runcommand "module load $mpi_module"
    fi
  fi
}

function configure {
  : # No configuration needed
}

function compile {
  prebuild
  runcommand "cd $hdf5_src_path"
  runcommand "rm -rf build"
  local args=$(printf " %s" "${compile_args[@]}")
  runcommand "ctest$args"
}

function install {
  runcommand "module list"
  runcommand "cd $hdf5_src_path/build"
  runcommand "make install"
  runcommand "cd HDF5_ZLIB-prefix/src/HDF5_ZLIB-build"
  runcommand "make install"
  runcommand "cd ../../../SZIP-prefix/src/SZIP-build"
  runcommand "make install"
}

function cleanup {
  runcommand "cd $hdf5_src_path"
  runcommand "rm -rf build"
}

function report {
  if [ ! $with_mpi = "OFF" ]; then
    REPORT_VARS+=(
      "MPI"
    )
    REPORT_VALS+=(
      "${with_mpi}"
    )
  fi
}

function modulefile {
  fname=$hdf5_module
  description="HDF5"
  if [ ! $with_mpi = "OFF" ]; then
    description=$description" @ MPI"
  fi
  if [ $enable_cuda = "ON" ]; then
    description=$description" @ CUDA"
  fi
  prereqs=""
  if [ $use_modules = "ON" ]; then
    prereqs+="prereq\t\t$cc_module"
    if [ $enable_cuda = "ON" ]; then
      prereqs+=" $cuda_module"
    fi
    if [ ! $with_mpi = "OFF" ]; then
      prereqs+=" $mpi_module"
    fi
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

conflict        hdf5
$prereqs

set             basedir                $install_path
prepend-path    PATH                   \$basedir/bin
prepend-path    LD_LIBRARY_PATH        \$basedir/lib
prepend-path    LIBRARY_PATH           \$basedir/lib
prepend-path    MANPATH                \$basedir/man
prepend-path    HDF5_ROOT              \$basedir
prepend-path    HDF5DIR                \$basedir
append-path     -d { } LDFLAGS         -L\$basedir/lib
append-path     -d { } INCLUDE         -I\$basedir/include
append-path     CPATH                  \$basedir/include
append-path     -d { } FFLAGS          -I\$basedir/include
append-path     -d { } FCFLAGS         -I\$basedir/include
append-path     -d { } LOCAL_LDFLAGS   -L\$basedir/lib
append-path     -d { } LOCAL_INCLUDE   -I\$basedir/include
append-path     -d { } LOCAL_CFLAGS    -I\$basedir/include
append-path     -d { } LOCAL_FFLAGS    -I\$basedir/include
append-path     -d { } LOCAL_FCFLAGS   -I\$basedir/include
append-path     -d { } LOCAL_CXXFLAGS  -I\$basedir/include
  '''
  runcommand "echo -e \"$modulecontent\" >>$fname"
}

run configure compile install cleanup modulefile report
