#! /usr/bin/env python3
#-----------------------------------------------------------------------------------------
#
# Options:
#   -h  --help            help message
#   --cluster             shortcut to choose cluster-specific configurations
#   --compiler            choose the compiler
#   --precision           code precision
#   -debug                compile in `debug` mode
#   -kokkos               copmile with `Kokkos` support
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

parser.add_argument('-kokkos',
                    action='store_true',
                    default=False,
                    help='compile with `Kokkos` support')

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
makefile_options['USEKOKKOS'] = ('y' if args['kokkos'] else 'n')

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
makefile_options['WARNING_FLAGS'] = "-Wall -Wextra"

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
   __    ___\ \ ,_\/\_\ \ ,_\ __  __
 /'__`\/' _ `\ \ \/\/\ \ \ \//\ \/\ \ 
/\  __//\ \/\ \ \ \_\ \ \ \ \\\\ \ \_\ \ 
\ \____\ \_\ \_\ \__\\\\ \_\ \__\/`____ \ 
 \/____/\/_/\/_/\/__/ \/_/\/__/`/___/> \ 
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

====================================================
'''
)
