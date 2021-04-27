#! /usr/bin/env python3
#-----------------------------------------------------------------------------------------
# Configure file for the `Entity` code to generate a temporary `Makefile`. 
# ... Parts of the code are adapted from the `K-Athena` MHD code (https://gitlab.com/pgrete/kathena).
#
# Options:
#   -h  --help                    help message
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

# Global Settings
# ---------------
# Default values:
DEF_compiler = 'icc'
DEF_cppstandard = 'c++17'
# Options:
Precision_options = ['double', 'single']
Kokkos_loop_options = ['default', '1DRange', 'MDRange', 'TP-TVR', 'TP-TTR', 'TP-TTR-TVR', 'for']

# . . . auxiliary functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . --> 
def defineOptions():
  parser = argparse.ArgumentParser()
  # system
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

def configureKokkos(arg, mopt):
  if arg['kokkos']:
    # using Kokkos
    # custom flag to recognize that the code is compiled with `Kokkos`
    mopt['KOKKOS_FLAG'] = "-D KOKKOS"
    mopt['KOKKOS_ARCH'] = arg['kokkos_arch']
    mopt['KOKKOS_DEVICES'] = arg['kokkos_devices']
    
    if 'Cuda' in mopt['KOKKOS_DEVICES']:
      # using Cuda
      if mopt['KOKKOS_CUDA_OPTIONS'] != '':
        mopt['KOKKOS_CUDA_OPTIONS'] += ','
      mopt['KOKKOS_CUDA_OPTIONS'] += 'enable_lambda'

      # no MPI (TODO)
      mopt['NVCC_WRAPPER_DEFAULT_COMPILER'] = mopt['COMPILER_COMMAND']
      mopt['COMPILER_COMMAND'] = mopt['KOKKOS_PATH'] + '/bin/nvcc_wrapper'

    #  makefile_options['NVCC_WRAPPER_DEFAULT_COMPILER'] = args['nvcc_wrapper_cxx']
    
    mopt['KOKKOS_OPTIONS'] = arg['kokkos_options']
    if mopt['KOKKOS_OPTIONS'] != '':
      mopt['KOKKOS_OPTIONS'] += ','
    mopt['KOKKOS_OPTIONS'] += 'disable_deprecated_code'

    mopt['KOKKOS_CUDA_OPTIONS'] = arg['kokkos_cuda_options']
    # this needs to be rewritten (also added to the Makefile.in)
    # TODO
    #  makefile_options['KOKKOS_VECTOR_LENGTH'] = '-1'
    #  if args['kokkos_loop'] == 'default':
      #  args['kokkos_loop'] = '1DRange' if 'Cuda' in args['kokkos_devices'] else 'for'
      #  makefile_options['KOKKOS_VECTOR_LENGTH'] = '-1'
    #  if args['kokkos_loop'] == '1DRange':
      #  makefile_options['KOKKOS_LOOP_LAYOUT'] = 'MANUAL1D_LOOP'
    #  elif args['kokkos_loop'] == 'MDRange':
      #  makefile_options['KOKKOS_LOOP_LAYOUT'] = 'MDRANGE_LOOP'
    #  elif args['kokkos_loop'] == 'for':
      #  makefile_options['KOKKOS_LOOP_LAYOUT'] = 'FOR_LOOP'
    #  elif args['kokkos_loop'] == 'TP-TVR':
      #  makefile_options['KOKKOS_LOOP_LAYOUT'] = 'TP_INNERX_LOOP\n#define INNER_TVR_LOOP'
      #  makefile_options['KOKKOS_VECTOR_LENGTH'] = ('32' if args['kokkos_vector_length'] == -1
                                             #  else str(args['kokkos_vector_length']))
    #  elif args['kokkos_loop'] == 'TP-TTR':
      #  makefile_options['KOKKOS_LOOP_LAYOUT'] = 'TP_INNERX_LOOP\n#define INNER_TTR_LOOP'
      #  makefile_options['KOKKOS_VECTOR_LENGTH'] = ('1' if args['kokkos_vector_length'] == -1
                                             #  else str(args['kokkos_vector_length']))
    #  elif args['kokkos_loop'] == 'TP-TTR-TVR':
      #  makefile_options['KOKKOS_LOOP_LAYOUT'] = 'TPTTRTVR_LOOP'
      #  makefile_options['KOKKOS_VECTOR_LENGTH'] = ('32' if args['kokkos_vector_length'] == -1
                                             #  else str(args['kokkos_vector_length']))
    settings = f'''
                `Kokkos`:
                  {'Architecture':30} {arg['kokkos_arch'] if arg['kokkos_arch'] else '-'}
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
  with open(m_out, 'w') as current_file:
    current_file.write(makefile_template)
# <-- auxiliary functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

# Set template and output filenames
makefile_input = 'Makefile.in'
makefile_output = 'Makefile'

# Step 1. Prepare parser, add each of the arguments
args = defineOptions() 

# Step 2. Set definitions and Makefile options based on above arguments

makefile_options = {}

# Settings
makefile_options['DEBUGMODE'] = ('y' if args['debug'] else 'n')
makefile_options['USEKOKKOS'] = ('y' if args['kokkos'] else 'n')

# Compilation commands
makefile_options['CXX'] = args['compiler']
makefile_options['CXXSTANDARD'] = f'{DEF_cppstandard}'

# Target names
makefile_options['NTT_TARGET'] = "ntt.exec"
makefile_options['TEST_TARGET'] = "test.exec"

# Paths
makefile_options['NTT_DIR'] = "ntt"
makefile_options['TEST_DIR'] = "tests"
makefile_options['SRC_DIR'] = "lib"
makefile_options['EXTERN_DIR'] = "extern"

# `Kokkos` settings
Kokkos_details = configureKokkos(args, makefile_options)

# Configuration flags for the performance build (TODO: compiler specific)
makefile_options['RELEASE_CONF_FLAGS'] = "-O3 -Ofast"
makefile_options['RELEASE_PP_FLAGS'] = "-DNDEBUG"

# Configuration flags for the debug build (TODO: compiler specific)
makefile_options['DEBUG_CONF_FLAGS'] = ""
makefile_options['DEBUG_PP_FLAGS'] = "-g -DDEBUG"

# Warning flags (TODO: compiler specific)
makefile_options['WARNING_FLAGS'] = "-Wall -Wextra -pedantic"

# Code fonfigurations
makefile_options['PRECISION'] = ("" if (args['precision'] == 'double') else "-D SINGLE_PRECISION")

# Step 3. Create new files, finish up
createMakefile(makefile_input, makefile_output, makefile_options)

#  Finish with diagnostic output
print(
f'''
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
  {'Compiler':32} {args['compiler']}
  {'Debug mode':32} {args['debug']}
  {Kokkos_details}
====================================================
'''
)
