#! /usr/bin/env python3
#-----------------------------------------------------------------------------------------
#
# Options:
#   -h  --help                    help message
#   --cluster=<CLUSTER>           shortcut to choose cluster-specific configurations
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
# ----------------------------------------------------------------------------------------

import argparse
import glob
import re
import subprocess
import os

# Set template and output filenames
makefile_input = 'Makefile.in'
makefile_output = 'Makefile'

# Default values:
DEF_compiler = 'icc'
DEF_cppstandard = 'c++17'

# Options:
ALL_clusters = ['stellar', 'frontera']

# Step 1. Prepare parser, add each of the arguments
parser = argparse.ArgumentParser()

# system
parser.add_argument('--cluster',
                    default=None,
                    choices=ALL_clusters,
                    help='shortcut to choose cluster-specific configurations')

parser.add_argument('--compiler',
                    default=DEF_compiler,
                    help='choose the compiler')

parser.add_argument('--precision',
                    default='double',
                    choices=['double', 'single'],
                    help='code precision')

parser.add_argument('-debug',
                    action='store_true',
                    default=False,
                    help='compile in `debug` mode')

# `Kokkos` specific arguments -- >
parser.add_argument('-kokkos',
                    action='store_true',
                    default=False,
                    help='compile with `Kokkos` support')

parser.add_argument('--kokkos_arch',
                    default='',
                    help='`Kokkos` architecture')

parser.add_argument('--kokkos_devices',
                    type=str,
                    default='',
                    help='`Kokkos` devices')

parser.add_argument('--kokkos_options',
                    type=str,
                    default='',
                    help='`Kokkos` options')

parser.add_argument('--kokkos_cuda_options',
                    type=str,
                    default='',
                    help='`Kokkos` CUDA options')

parser.add_argument('--kokkos_loop',
                    default='default',
                    choices=['default', '1DRange', 'MDRange', 'TP-TVR', 'TP-TTR', 'TP-TTR-TVR', 'for'],
                    help='`Kokkos` loop layout')

parser.add_argument('--kokkos_vector_length',
                    default=-1,
                    type=int,
                    help='`Kokkos` vector length')

# < -- `Kokkos` specific arguments

args = vars(parser.parse_args())

# Step 2. Set definitions and Makefile options based on above arguments

makefile_options = {}

# specific cluster:
specific_cluster = False
if args['cluster']:
  specific_cluster = True
  clustername = args['cluster'].capitalize()
  # cluster specific options here
  # TODO

# Settings
makefile_options['DEBUGMODE'] = ('y' if args['debug'] else 'n')

# `Kokkos` settings
makefile_options['USEKOKKOS'] = ('y' if args['kokkos'] else 'n')
Kokkos_details = ''
if args['kokkos']:
  # custom flag to recognize that the code is compiled with `Kokkos`
  makefile_options['KOKKOS_FLAG'] = "-D KOKKOS"
  makefile_options['KOKKOS_ARCH'] = args['kokkos_arch']
  makefile_options['KOKKOS_DEVICES'] = args['kokkos_devices']
  makefile_options['KOKKOS_OPTIONS'] = args['kokkos_options']
  if makefile_options['KOKKOS_OPTIONS'] != '':
    makefile_options['KOKKOS_OPTIONS'] += ','
  makefile_options['KOKKOS_OPTIONS'] += 'disable_deprecated_code'
  makefile_options['KOKKOS_CUDA_OPTIONS'] = args['kokkos_cuda_options']
  # this needs to be rewritten (also added to the Makefile.in)
  # TODO
  makefile_options['KOKKOS_VECTOR_LENGTH'] = '-1'
  if args['kokkos_loop'] == 'default':
    args['kokkos_loop'] = '1DRange' if 'Cuda' in args['kokkos_devices'] else 'for'
    makefile_options['KOKKOS_VECTOR_LENGTH'] = '-1'
  if args['kokkos_loop'] == '1DRange':
    makefile_options['KOKKOS_LOOP_LAYOUT'] = 'MANUAL1D_LOOP'
  elif args['kokkos_loop'] == 'MDRange':
    makefile_options['KOKKOS_LOOP_LAYOUT'] = 'MDRANGE_LOOP'
  elif args['kokkos_loop'] == 'for':
    makefile_options['KOKKOS_LOOP_LAYOUT'] = 'FOR_LOOP'
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
  Kokkos_details = f'''
  `Kokkos`:
    {'Architecture':30} {args['kokkos_arch'] if args['kokkos_arch'] else '-'}
    {'Devices':30} {args['kokkos_devices'] if args['kokkos_devices'] else '-'}
    {'Options':30} {args['kokkos_options'] if args['kokkos_options'] else '-'}
    {'Loop':30} {args['kokkos_loop']}
    {'Vector length':30} {args['kokkos_vector_length']}
  '''

# Target names
makefile_options['NTT_TARGET'] = "ntt.exec"
makefile_options['TEST_TARGET'] = "test.exec"

# Paths
makefile_options['NTT_DIR'] = "ntt"
makefile_options['TEST_DIR'] = "tests"
makefile_options['SRC_DIR'] = "lib"
makefile_options['EXTERN_DIR'] = "extern"

# Compilation commands
# test if the compiler exists
compiler_found = True
try:
  devnull = open(os.devnull, 'w')
  subprocess.call(args['compiler'], stdout=devnull, stderr=devnull)
except FileNotFoundError:
  compiler_found = False
if not compiler_found:
  raise NameError(f"Compiler `{args['compiler']}` not found on this system.")

makefile_options['CXX'] = args['compiler']
makefile_options['CXXSTANDARD'] = f'-std={DEF_cppstandard}'

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
with open(makefile_input, 'r') as current_file:
  makefile_template = current_file.read()
for key, val in makefile_options.items():
  makefile_template = re.sub(r'@{0}@'.format(key), val, makefile_template)
makefile_template = re.sub('# Template for ', '# ', makefile_template)
with open(makefile_output, 'w') as current_file:
  current_file.write(makefile_template)

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
