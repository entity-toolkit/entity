# -----------------------------------------------------------------------------------------
# Configure file for the `Entity` code to generate a temporary `Makefile`.
#
# Options:
#   -h  --help                    help message
#
# [ Compilation flags ]
#   -verbose                      enable verbose compilation mode
#   -debug                        compile in `debug` mode
#   --compiler=<COMPILER>         compiler used (can be a valid path to the binary)
#   --build=<DIR>                 specify building directory
#   --bin=<DIR>                   specify directory for executables
#
# [ Nttiny flags ]
#   -nttiny                       enable visualizer compilation
#   --nttiny_path=<DIR>           specify path for `Nttiny` (relative to current dir or absolute)
#
# [ Simulation flags ]
#   --pgen=<PROBLEM_GENERATOR>    specify the problem generator to be used
#   --precision=[single|double]   floating point precision used [default: single]
#   --metric=<METRIC>             select metric to be used [default: minkowski]
#   --simtype=<SIM_TYPE>          select simulation type [default: pic]
#
# [ Kokkos-specific flags ]
#   --kokkos_devices=<DEV>        `Kokkos` devices
#   --kokkos_arch=<ARCH>          `Kokkos` architecture
#   --kokkos_options=<OPT>        `Kokkos` options
#   --kokkos_cuda_options=<OPT>   `Kokkos` Cuda options
# ----------------------------------------------------------------------------------------

import argparse
import glob
import re
import subprocess
import os
import sys
import textwrap
from pathlib import Path

assert sys.version_info >= (3, 7), "Requires python 3.7 or higher"

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
Metric_options = ['minkowski', 'spherical',
                  'qspherical', 'kerr_schild', 'qkerr_schild']
Simtype_options = ['pic', 'grpic']


def findFiles(directory, extension):
    return glob.glob(directory + '/*/*.' + extension) + glob.glob(directory + '/*.' + extension)


Pgen_options = [f.replace('.hpp', '').replace('\\', '/').replace('pgen/', '')
                for f in findFiles('pgen', 'hpp')]
Kokkos_devices = dict(host=['Serial', 'OpenMP', 'PThreads'], device=['Cuda'])
Kokkos_arch = dict(host=["AMDAVX", "EPYC", "ARMV80", "ARMV81", "ARMV8_THUNDERX",
                         "ARMV8_THUNDERX2", "WSM", "SNB", "HSW", "BDW", "SKX",
                         "KNC", "KNL", "BGQ", "POWER7", "POWER8", "POWER9"],
                   device=["KEPLER30", "KEPLER32", "KEPLER35", "KEPLER37",
                           "MAXWELL50", "MAXWELL52", "MAXWELL53", "PASCAL60",
                           "PASCAL61", "VOLTA70", "VOLTA72", "TURING75",
                           "AMPERE80", "VEGA900", "VEGA906", "INTEL_GE"])
Kokkos_devices_options = Kokkos_devices["host"] + Kokkos_devices["device"]
Kokkos_arch_options = Kokkos_arch["host"] + Kokkos_arch["device"]
Kokkos_loop_options = ['default', '1DRange',
                       'MDRange', 'TP-TVR', 'TP-TTR', 'TP-TTR-TVR', 'for']

# . . . auxiliary functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . -->
use_nvcc_wrapper = False


def findCompiler(compiler):
    find_command = subprocess.run(
        ['which', compiler], capture_output=True, text=True)
    return find_command.stdout.strip() if (find_command.returncode == 0) else 'N/A'


def pathNotEmpty(path):
    ls_path = subprocess.run(['ls', path], capture_output=True, text=True)
    return path if (ls_path.returncode == 0) else 'N/A'


def defineOptions():
    parser = argparse.ArgumentParser()
    # compilation
    parser.add_argument('-verbose', action='store_true',
                        default=False, help='enable verbose compilation mode')
    parser.add_argument('--build', default=DEF_build_dir,
                        help='specify building directory')
    parser.add_argument('--bin', default=DEF_bin_dir,
                        help='specify directory for executables')
    parser.add_argument('--compiler', default=DEF_compiler,
                        help='choose the compiler')
    parser.add_argument('-debug', action='store_true',
                        default=False, help='compile in `debug` mode')

    # visualizer
    parser.add_argument('-nttiny', action='store_true',
                        default=False, help='enable nttiny visualizer compilation')
    parser.add_argument('--nttiny_path', default="extern/nttiny",
                        help='specify path for `Nttiny`')

    # simulation
    parser.add_argument('--precision', default='single',
                        choices=Precision_options, help='code precision (default: `single`)')
    parser.add_argument(
        '--metric', default=Metric_options[0], choices=Metric_options, help='select metric to be used (default: `minkowski`)')
    parser.add_argument(
        '--simtype', default=Simtype_options[0], choices=Simtype_options, help='select simulation type (default: `pic`)')
    parser.add_argument('--pgen', default="", choices=Pgen_options,
                        help='problem generator to be used (default: `ntt_dummy`)')

    # `Kokkos` specific
    parser.add_argument(
        '--kokkos_devices', default=Kokkos_devices['host'][0], help='`Kokkos` devices')
    parser.add_argument('--kokkos_arch', default='',
                        help='`Kokkos` architecture')
    parser.add_argument('--kokkos_options', default='',
                        help='`Kokkos` options')
    parser.add_argument('--kokkos_cuda_options', default='',
                        help='`Kokkos` CUDA options')
    return vars(parser.parse_args())


def configureKokkos(arg, mopt):
    global use_nvcc_wrapper
    kokkos_configs = {}

    def parseArchDevice(carg, kokkos_list):
        _ = carg.split(',')
        assert len(_) <= 2, "Wrong arch/device specified"
        if len(_) == 2:
            _1, _2 = _
            if _2 in kokkos_list['host']:
                _1 = _[1]
                _2 = _[0]
            return _1, _2
        elif len(_) == 1:
            _1 = _[0]
            _2 = None
            if _1 in kokkos_list['device']:
                # enabling openmp if CUDA is enabled
                _2 = 'OpenMP'
            elif (not (_1 in kokkos_list['host'])):
                if _1 != '':
                    raise ValueError("Wrong arch/device specified")
                else:
                    _1 = None
            return _1, _2
        else:
            return None, None
    host_d, device_d = parseArchDevice(arg['kokkos_devices'], Kokkos_devices)
    host_a, device_a = parseArchDevice(arg['kokkos_arch'], Kokkos_arch)
    if host_d is not None:
        assert (host_d in Kokkos_devices['host']), 'Wrong host'
        kokkos_configs['devices'] = host_d
    if device_d is not None:
        assert (device_d in Kokkos_devices['device']), 'Wrong device'
        kokkos_configs['devices'] += ',' + device_d
    if host_a is not None:
        assert (host_a in Kokkos_arch['host']), 'Wrong host architecture'
        kokkos_configs['arch'] = host_a
    if device_a is not None:
        assert (device_a in Kokkos_arch['device']), 'Wrong device architecture'
        try:
            kokkos_configs['arch'] += ',' + device_a
        except:
            kokkos_configs['arch'] = device_a

    mopt['KOKKOS_DEVICES'] = kokkos_configs['devices']
    mopt['KOKKOS_ARCH'] = kokkos_configs.get('arch', '')
    if 'Cuda' in kokkos_configs['devices']:
        mopt['DEFINITIONS'] += '-DGPUENABLED '
    if 'OpenMP' in kokkos_configs['devices']:
        mopt['DEFINITIONS'] += '-DOMPENABLED '

    mopt['KOKKOS_OPTIONS'] = arg['kokkos_options']
    if mopt['KOKKOS_OPTIONS'] != '':
        mopt['KOKKOS_OPTIONS'] += ','
    mopt['KOKKOS_OPTIONS'] += 'disable_deprecated_code'

    mopt['KOKKOS_CUDA_OPTIONS'] = arg['kokkos_cuda_options']

    if 'Cuda' in mopt['KOKKOS_DEVICES']:
        # using Cuda
        mopt['KOKKOS_CUDA_OPTIONS'] = arg['kokkos_cuda_options']
        if mopt['KOKKOS_CUDA_OPTIONS'] != '':
            mopt['KOKKOS_CUDA_OPTIONS'] += ','
        mopt['KOKKOS_CUDA_OPTIONS'] += 'enable_lambda'

        use_nvcc_wrapper = True

        # no MPI (TODO)
        arg['nvcc_wrapper_cxx'] = arg['compiler']
        mopt['HOST_COMPILER'] = arg["nvcc_wrapper_cxx"]
        mopt['COMPILER'] = f'NVCC_WRAPPER_DEFAULT_COMPILER={arg["nvcc_wrapper_cxx"]} '\
            + '${KOKKOS_PATH}/bin/nvcc_wrapper'
        # + 'NVCC_WRAPPER_TMPDIR=${BUILD_DIR}/tmp '\
        # add with MPI here (TODO)

    settings = f'''
  `Kokkos`:
    {'Devices':30} {mopt['KOKKOS_DEVICES']}
    {'Architecture':30} {mopt['KOKKOS_ARCH']}
    {'Options':30} {mopt['KOKKOS_OPTIONS'] if mopt['KOKKOS_OPTIONS'] is not None else '-'}
    {'Cuda options':30} {mopt['KOKKOS_CUDA_OPTIONS'] if mopt['KOKKOS_CUDA_OPTIONS'] is not None else '-'}'''
    return settings


def createMakefile(m_in, m_out, mopt):
    with open(m_in, 'r') as current_file:
        makefile_template = current_file.read()
    # print(makefile_template)
    for key, val in mopt.items():
        makefile_template = makefile_template.replace(f'@{key}@', val)
    if not args['nttiny']:
        makefile_template = re.sub(
            "# for nttiny />[\S\s]*?</ for nttiny", '', makefile_template)
    with open(args['build'] + '/' + m_out, 'w') as current_file:
        current_file.write(makefile_template)
# <-- auxiliary functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


# Step 1. Prepare parser, add each of the arguments
args = defineOptions()

# Step 2. Set definitions and Makefile options based on above arguments

makefile_options = {}

if sys.platform.startswith('win'):
    makefile_options['FIND'] = '/usr/bin/find.exe'
else:
    makefile_options['FIND'] = 'find'

# Settings
makefile_options['VERBOSE'] = ('y' if args['verbose'] else 'n')
makefile_options['DEBUGMODE'] = ('y' if args['debug'] else 'n')

# Compilation commands
makefile_options['COMPILER'] = args['compiler']
makefile_options['HOST_COMPILER'] = args['compiler']
makefile_options['CXXSTANDARD'] = f'{DEF_cppstandard}'

# Target names
makefile_options['NTT_TARGET'] = "ntt.exec"
makefile_options['TEST_TARGET'] = "test.exec"

# Paths
makefile_options['BUILD_DIR'] = args['build']
makefile_options['BIN_DIR'] = args['bin']
makefile_options['NTT_DIR'] = 'ntt'
makefile_options['PGEN_DIR'] = 'pgen'
makefile_options['TEST_DIR'] = 'tests'
makefile_options['SRC_DIR'] = 'src'
makefile_options['EXTERN_DIR'] = 'extern'

if args['nttiny']:
    if (args['nttiny_path']) != '':
        args['nttiny_path'] = os.path.abspath(args['nttiny_path'])
    makefile_options['NTTINY_DIR'] = args['nttiny_path']
    makefile_options['VIS_DIR'] = "vis"

makefile_options['DEFINITIONS'] = ''

makefile_options['PGEN'] = args['pgen']

Path(args['build']).mkdir(parents=True, exist_ok=True)

# `Kokkos` settings
Kokkos_details = configureKokkos(args, makefile_options)

# Configuration flags for the performance build (TODO: compiler specific)
makefile_options['RELEASE_CFLAGS'] = "-O3 -DNDEBUG"

# Configuration flags for the debug build (TODO: compiler specific)
makefile_options['DEBUG_CFLAGS'] = "-O0 -g -DDEBUG"

# Warning flags (TODO: compiler specific)
makefile_options['WARNING_FLAGS'] = "-Wall -Wextra -pedantic"

# Code configurations
makefile_options['PRECISION'] = (
    "" if (args['precision'] == 'double') else "-DSINGLE_PRECISION")
makefile_options['METRIC'] = args['metric'].upper()
makefile_options['SIMTYPE'] = args['simtype'].upper()

# Step 3. Create new files, finish up
createMakefile(makefile_input, makefile_output, makefile_options)

makedemo = subprocess.run(['make', 'demo'],
                          capture_output=True,
                          text=True,
                          cwd=makefile_options['BUILD_DIR'])
if makedemo.returncode != 0:
    raise Exception(f'Error creating demo: {makedemo.stdout}')
try:
    makedemo = makedemo.stdout.strip()
    compiledemo = makedemo.split('\n')[1]
    linkdemo = makedemo.split('\n')[4]
except Exception as e:
    raise Exception(f'Failed to compile demo: {e}')


def beautifyCommands(command):
    i = command.index(' -')
    cmd = command[:i]
    cmd = re.sub('\s*(NVCC_.*?)\s', '\\1\n', cmd)
    cmd = re.sub('\n', '\n  ', cmd)
    flags = list(set(re.sub('<.o> *', '',
                            re.sub(r'-[I|L|o|c].+?[ |>|$]', '',
                                   re.sub(r'-([I|D|c|o|W|O|L]) ', r'-\1',
                                          command[i + 1:]))
                            ).strip().split(' ')))
    order = ['-std', '-D', '-W', '-l', '--diag', '']
    accounted_flags = []
    ordered_flags = {key: [] for key in order}
    for o in order:
        for flag in flags:
            if (o in flag) and (not flag in accounted_flags):
                accounted_flags.append(flag)
                ordered_flags[o].append(re.sub('-D', '-D ', flag))
    fstring = ""
    fstring += "  " + cmd + "\n"
    for o in order:
        fstring += "      "
        for f in ordered_flags[o]:
            fstring += f + " "
        fstring += "\n"
    fstring = "".join(filter(str.strip, fstring.splitlines(True)))[:-1]
    return fstring

# add some useful notes


def makeNotes():
    notes = ''
    cxx = args['nvcc_wrapper_cxx'] if use_nvcc_wrapper else makefile_options['COMPILER']
    if use_nvcc_wrapper:
        notes += f"* nvcc recognized as:\n    $ {findCompiler('nvcc')}\n  "
    notes += f"* {'nvcc wrapper ' if use_nvcc_wrapper else ''}compiler recognized as:\n    $ {findCompiler(cxx)}\n  "
    if 'OpenMP' in args['kokkos_devices']:
        notes += f"* when using OpenMP set the following environment variables:\n    $ export OMP_PROC_BIND=spread OMP_PLACES=threads\n  "
    if args['nttiny']:
        notes += f"* `nttiny` path:\n    $ {pathNotEmpty(args['nttiny_path'])}"
    return notes.strip()


short_compiler = (
    f"nvcc_wrapper [{args['nvcc_wrapper_cxx']}]" if use_nvcc_wrapper else makefile_options['COMPILER'])

full_command = " ".join(sys.argv[:])

#  Finish with diagnostic output
w = 80
full_command = ' \\\n'.join(textwrap.wrap(full_command, w - 4,
                                          subsequent_indent="      ",
                                          initial_indent="  "))
report = f'''
{'':=<{w}}
             __        __
            /\ \__  __/\ \__
   __    ___\ \  _\/\_\ \  _\  __  __
 / __ \/  _  \ \ \/\/\ \ \ \/ /\ \/\ \\
/\  __//\ \/\ \ \ \_\ \ \ \ \_\ \ \_\ \  __
\ \____\ \_\ \_\ \__\\\\ \_\ \__\\\\ \____ \/\_\\
 \/____/\/_/\/_/\/__/ \/_/\/__/ \/___/  \/_/
                                   /\___/
                                   \/__/

{'':=<{w}}
{'Full configure command ':.<{w}}

{full_command}

{'Setup configurations ':.<{w}}

  {'Simulation type':32} {args['simtype'].upper()}
  {'Problem generator':32} {args['pgen'] if args['pgen'] != '' else '--'}
  {'Precision':32} {args['precision']}
  {'Metric':32} {args['metric']}

{'Physics ':.<{w}}

{'Technical details ':.<{w}}

  {'Compiler':32} {short_compiler}
  {'Debug mode':32} {args['debug']}
  {Kokkos_details}

{'Notes ':.<{80}}

  {makeNotes()}

{'Compilation command ':.<{w}}

{beautifyCommands(compiledemo)}

{'Linking command ':.<{w}}

{beautifyCommands(linkdemo)}

{'':=<{w}}
'''

print(report)

with open(args['build'] + "/REPORT", 'w') as reportfile:
    reportfile.write(report)
