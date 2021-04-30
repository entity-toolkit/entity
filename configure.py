#! /usr/bin/env python3
#-----------------------------------------------------------------------------------------
# Configure file for the `Entity` code to generate a temporary `Makefile`. 
# ... Parts of the code are adapted from the `K-Athena` MHD code (https://gitlab.com/pgrete/kathena).
#
# Options:
#   -h  --help                    help message
#   -verbose                      enable verbose compilation mode
#   --build=<DIR>                 specify building directory
#   --bin=<DIR>                   specify directory for executables
#   --compiler=<COMPILER>         compiler used (can be a valid path to the binary)
#   --precision=[single|double]   floating point precision used
#   -debug                        compile in `debug` mode
#   -kokkos                       compile with `Kokkos` support
#   --kokkos_arch=<ARCH>          `Kokkos` architecture
#   --kokkos_devices=<DEV>        `Kokkos` devices
#   --kokkos_options=<OPT>        `Kokkos` options
#   --kokkos_cuda_options=<COPT>  `Kokkos` Cuda options
#   --kokkos_loop=[...]           `Kokkos` loop layout
#   --kokkos_vector_length=<VLEN> `Kokkos` vector length
#   --nvcc_wrapper_cxx=<COMPILER> `NVCC_WRAPPER_DEFAULT_COMPILER` flag for `Kokkos`
# ----------------------------------------------------------------------------------------

import argparse
import glob
import re
import subprocess
import os
from pathlib import Path

# Global Settings
# ---------------
# Default values:
DEF_build_dir = 'build'
DEF_bin_dir = 'bin'
DEF_compiler = 'g++'
DEF_cppstandard = 'c++17'

# Set template and output filenames
makefile_input = 'Makefile.in'
makefile_output = 'Makefile'

# Options:
Precision_options = ['double', 'single']
Kokkos_loop_options = ['default', '1DRange', 'MDRange', 'TP-TVR', 'TP-TTR', 'TP-TTR-TVR', 'for']

# . . . auxiliary functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . --> 
def defineOptions():
  parser = argparse.ArgumentParser()
  # system
  parser.add_argument('-verbose', action='store_true', default=False, help='enable verbose compilation mode')
  parser.add_argument('--build', default=DEF_build_dir, help='specify building directory')
  parser.add_argument('--bin', default=DEF_bin_dir, help='specify directory for executables')
  parser.add_argument('--compiler', default=DEF_compiler, help='choose the compiler')
  parser.add_argument('--precision', default='double', choices=Precision_options, help='code precision')
  parser.add_argument('-debug', action='store_true', default=False, help='compile in `debug` mode')
  # `Kokkos` specific arguments -- >
  parser.add_argument('-kokkos', action='store_true', default=False, help='compile with `Kokkos` support')
  parser.add_argument('--kokkos_arch', default='', help='`Kokkos` architecture')
  parser.add_argument('--kokkos_devices', default='', help='`Kokkos` devices')
  parser.add_argument('--kokkos_options', default='', help='`Kokkos` options')
  parser.add_argument('--kokkos_cuda_options', default='', help='`Kokkos` CUDA options')
  parser.add_argument('--kokkos_loop', default='default', choices=Kokkos_loop_options, help='`Kokkos` loop layout')
  parser.add_argument('--nvcc_wrapper_cxx', default='g++', help='Sets the `NVCC_WRAPPER_DEFAULT_COMPILER` flag for `Kokkos`')
  parser.add_argument('--kokkos_vector_length', default=-1, type=int, help='`Kokkos` vector length')
  # < -- `Kokkos` specific arguments
  return vars(parser.parse_args())

def parseKokkosDev(kokkos_dev):
  openmp_proper = 'OpenMP'
  cuda_proper = 'Cuda'
  aliases = {'omp': openmp_proper, 'openmp': openmp_proper, 'cuda': cuda_proper} 
  def swapAlias(expr, aliases):
    for al in aliases:
      if expr.lower() == al:
        expr = aliases[al]
    return expr
  kokkos_dev = [kd.strip() for kd in kokkos_dev.split(',')]
  kokkos_dev = [swapAlias(kd, aliases) for kd in kokkos_dev]
  return ','.join(kokkos_dev)

def configureKokkos(arg, mopt):
  if arg['kokkos']:
    # using Kokkos
    # custom flag to recognize that the code is compiled with `Kokkos`
    arg['kokkos_devices'] = parseKokkosDev(arg['kokkos_devices'])
    mopt['KOKKOS_ARCH'] = arg['kokkos_arch']
    mopt['KOKKOS_DEVICES'] = arg['kokkos_devices']
    
    mopt['KOKKOS_OPTIONS'] = arg['kokkos_options']
    if mopt['KOKKOS_OPTIONS'] != '':
      mopt['KOKKOS_OPTIONS'] += ','
    mopt['KOKKOS_OPTIONS'] += 'disable_deprecated_code'
    
    mopt['NVCC_WRAPPER_DEFAULT_COMPILER'] = arg['nvcc_wrapper_cxx']
    mopt['KOKKOS_CUDA_OPTIONS'] = arg['kokkos_cuda_options']
    
    if 'Cuda' in mopt['KOKKOS_DEVICES']:
      # using Cuda
      mopt['KOKKOS_CUDA_OPTIONS'] = arg['kokkos_cuda_options']
      if mopt['KOKKOS_CUDA_OPTIONS'] != '':
        mopt['KOKKOS_CUDA_OPTIONS'] += ','
      mopt['KOKKOS_CUDA_OPTIONS'] += 'enable_lambda'

      # no MPI (TODO)
      mopt['NVCC_WRAPPER_DEFAULT_COMPILER'] = mopt['COMPILER']
      mopt['COMPILER'] = '${KOKKOS_PATH}/bin/nvcc_wrapper'
      # add with MPI here (TODO)
    
    if arg['kokkos_loop'] == 'default':
      arg['kokkos_loop'] = '1DRange' if 'Cuda' in arg['kokkos_devices'] else 'for'

    mopt['KOKKOS_VECTOR_LENGTH'] = '-1'
    if arg['kokkos_loop'] == '1DRange':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D MANUAL1D_LOOP'
    elif arg['kokkos_loop'] == 'MDRange':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D MDRANGE_LOOP'
    elif arg['kokkos_loop'] == 'for':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D FOR_LOOP'
    elif arg['kokkos_loop'] == 'TP-TVR':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D TP_INNERX_LOOP -D INNER_TVR_LOOP'
      mopt['KOKKOS_VECTOR_LENGTH'] = ('32' if arg['kokkos_vector_length'] == -1 else str(arg['kokkos_vector_length']))
    elif arg['kokkos_loop'] == 'TP-TTR':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D TP_INNERX_LOOP -D INNER_TTR_LOOP'
      mopt['KOKKOS_VECTOR_LENGTH'] = ('1' if arg['kokkos_vector_length'] == -1 else str(arg['kokkos_vector_length']))
    elif arg['kokkos_loop'] == 'TP-TTR-TVR':
      mopt['KOKKOS_LOOP_LAYOUT'] = '-D TPTTRTVR_LOOP'
      mopt['KOKKOS_VECTOR_LENGTH'] = ('32' if arg['kokkos_vector_length'] == -1 else str(arg['kokkos_vector_length']))

    mopt['KOKKOS_VECTOR_LENGTH'] = '-D KOKKOS_VECTOR_LENGTH=' + mopt['KOKKOS_VECTOR_LENGTH']

    settings = f'''
  `Kokkos`:
    {'Architecture':30} {arg['kokkos_arch'] if arg['kokkos_arch'] else '-'}
    {'NVCC wrapper compiler':30} {mopt['NVCC_WRAPPER_DEFAULT_COMPILER']}
    {'Devices':30} {arg['kokkos_devices'] if arg['kokkos_devices'] else '-'}
    {'Options':30} {arg['kokkos_options'] if arg['kokkos_options'] else '-'}
    {'Loop':30} {arg['kokkos_loop']}
    {'Vector length':30} {arg['kokkos_vector_length']}
  '''
    return settings
  else:
    return ''

def createMakefile(m_in, m_out, mopt):
  with open(m_in, 'r') as current_file:
    makefile_template = current_file.read()
  for key, val in mopt.items():
    makefile_template = re.sub(r'@{0}@'.format(key), val, makefile_template)
  makefile_template = re.sub('# Template for ', '# ', makefile_template)
  with open(args['build'] + '/' + m_out, 'w') as current_file:
    current_file.write(makefile_template)
# <-- auxiliary functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

# Step 1. Prepare parser, add each of the arguments
args = defineOptions() 

# Step 2. Set definitions and Makefile options based on above arguments

makefile_options = {}

# Settings
makefile_options['VERBOSE'] = ('y' if args['verbose'] else 'n')
makefile_options['DEBUGMODE'] = ('y' if args['debug'] else 'n')
makefile_options['USEKOKKOS'] = ('y' if args['kokkos'] else 'n')

# Compilation commands
makefile_options['COMPILER'] = args['compiler']
makefile_options['CXXSTANDARD'] = f'{DEF_cppstandard}'

# Target names
makefile_options['NTT_TARGET'] = "ntt.exec"
makefile_options['TEST_TARGET'] = "test.exec"

# Paths
makefile_options['BUILD_DIR'] = args['build']
makefile_options['BIN_DIR'] = args['bin']
makefile_options['NTT_DIR'] = 'ntt'
makefile_options['TEST_DIR'] = 'tests'
makefile_options['SRC_DIR'] = 'lib'
makefile_options['EXTERN_DIR'] = 'extern'

Path(args['build']).mkdir(parents=True, exist_ok=True)

# `Kokkos` settings
Kokkos_details = configureKokkos(args, makefile_options)

# Configuration flags for the performance build (TODO: compiler specific)
makefile_options['RELEASE_CONF_FLAGS'] = "-O3 -Ofast"
makefile_options['RELEASE_PP_FLAGS'] = "-DNDEBUG"

# Configuration flags for the debug build (TODO: compiler specific)
makefile_options['DEBUG_CONF_FLAGS'] = ""
makefile_options['DEBUG_PP_FLAGS'] = "-O0 -g -DDEBUG"

# Warning flags (TODO: compiler specific)
makefile_options['WARNING_FLAGS'] = "-Wall -Wextra -pedantic"

# Code fonfigurations
makefile_options['PRECISION'] = ("" if (args['precision'] == 'double') else "-D SINGLE_PRECISION")

# Step 3. Create new files, finish up
createMakefile(makefile_input, makefile_output, makefile_options)

#  Finish with diagnostic output
report = f'''
====================================================
                 __        __                   
                /\ \__  __/\ \__                
       __    ___\ \  _\/\_\ \  _\  __  __       
     / __ \/  _  \ \ \/\/\ \ \ \/ /\ \/\ \      
    /\  __//\ \/\ \ \ \_\ \ \ \ \_\ \ \_\ \  __ 
    \ \____\ \_\ \_\ \__\\\\ \_\ \__\\\\ \____ \/\_\\
     \/____/\/_/\/_/\/__/ \/_/\/__/ \/___/  \/_/
                                       /\___/   
                                       \/__/    

====================================================
Code has been configured with the following options:

Setup configurations ...............................

Computational details ..............................
  {'Precision':32} {args['precision']}
  {'Use `Kokkos` Library':32} {args['kokkos']}

Physics ............................................

Technical details ..................................
  {'Compiler':32} {makefile_options['COMPILER']}
  {'Debug mode':32} {args['debug']}
  {Kokkos_details}
====================================================
'''

print (report)

with open(args['build'] + "/REPORT", 'w') as reportfile:
  reportfile.write(report)
