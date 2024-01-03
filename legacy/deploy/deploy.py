import argparse
import pathlib
from pip._vendor import tomli
from string import Template
import os
from typing import Final

gpu_archs = [
    "VOLTA",
    "TURING",
    "AMPERE",
    "MAXWELL",
    "PASCAL",
    "KEPLER",
    "INTEL",
    "VEGA",
    "NAVI",
]


class ColoredText(str):
    def __new__(cls, text, color):
        return super().__new__(cls, text)

    def __init__(self, text, color):
        self.color = color
        self.colors = {
            "red": "\033[0;31m",
            "green": "\033[0;32m",
            "blue": "\033[0;34m",
            "gray": "\033[0;30m",
            "nc": "\033[0m",
        }

    def __str__(self):
        return self.colors[self.color] + self + self.colors["nc"]

    def __repr__(self):
        return str(self)


# Dependency not implemented error

dependency_error: Final[str] = ColoredText(
    "Dependency deployment not implemented yet. You can run the included .sh scripts manually. Run with `bash compile_<lib>.sh` -h for more info.",
    "red",
)

arg_parser = argparse.ArgumentParser(description="Deploy Entity modulefiles")

arg_parser.add_argument(
    "-c",
    "--config",
    help="Path to the specific configuration file",
    default="config.toml",
    type=pathlib.Path,
    required=True,
)
arg_parser.add_argument(
    "-d",
    "--deploy",
    help="Execute the stript",
    default=False,
    action="store_true",
)
arg_parser.add_argument(
    "-v",
    "--verbose",
    help="Print verbose output",
    default=False,
    action="store_true",
)
arg_parser.add_argument(
    "--depends",
    help="Also build, install & deploy the dependencies",
    default=False,
    action="store_true",
)

modulefile_template = Template(
    """
#%Module1.0######################################################################
##
## Entity ${configuration}
##
################################################################################
proc ModulesHelp { } {
  puts stderr "\\tEntity ${configuration}\\n"
}

module-whatis   "Entity ${configuration}"

conflict        entity

${kokkos_setenvs}
${openmp_setenvs}

${entity_setenvs}

${modules}
"""
)


def get_suffix(debug=False, mpi=False, cuda=False, archs=[]):
    return (
        ("/debug" if debug else "")
        + ("/mpi" if mpi else "")
        + ("/cuda" if cuda else "")
        + "/"
        + "/".join(
            f"{arch.lower()}"
            for arch in sorted(
                archs, key=lambda x: not any([any(ar in x for ar in gpu_archs)])
            )
        )
    )


if __name__ == "__main__":
    args = arg_parser.parse_args()
    configfname = args.config
    dependency_build_scripts = []
    with open(configfname, "r", encoding="utf-8") as f:
        config = tomli.loads(f.read())
        modulepath = pathlib.Path(os.path.expandvars(config["entity"]["modulepath"]))
        instances = config["entity"]["instances"]
        dependencies = config["dependencies"]
        cc_module = None
        if (cc_path := dependencies["cc"]).startswith("module:"):
            cc_module = cc_path.split(":")[1]
        for debug in instances["debug"]:
            for mpi in instances["with_mpi"]:
                for cuda in instances["with_cuda"]:
                    for architectures in instances["archs"]:
                        archs = architectures.split(",")
                        isgpu = any([any(ar in a for ar in gpu_archs) for a in archs])
                        if cuda != isgpu:
                            continue

                        entity_setenvs = []
                        kokkos_setenvs = []
                        openmp_setenvs = []
                        modules = [cc_module] if cc_module else []
                        entity_setenvs += [
                            ["Entity_ENABLE_DEBUG", "ON" if debug else "OFF"]
                        ]
                        entity_setenvs += [
                            ["Entity_ENABLE_MPI", "ON" if mpi else "OFF"]
                        ]
                        if cuda and (cuda_path := dependencies["cuda"]).startswith(
                            "module:"
                        ):
                            modules += [(cuda_module := cuda_path.split(":")[1])]
                        for arch in archs:
                            kokkos_setenvs += [[f"Kokkos_ARCH_{arch}", "ON"]]

                        if mpi and (mpi_path := dependencies["mpi"]).startswith(
                            "module:"
                        ):
                            modules += [
                                (
                                    mpi_module := mpi_path.split(":")[1]
                                    + ("/cuda" if cuda else "/cpu")
                                )
                            ]
                        else:
                            openmp_setenvs += [
                                ["Kokkos_ENABLE_OPENMP", "ON"],
                                ["OMP_PROC_BIND", "spread"],
                                ["OMP_PLACES", "threads"],
                                ["OMP_NUM_THREADS", "[exec nproc]"],
                            ]
                        if (hdf5_path := dependencies["hdf5"]).startswith("module:"):
                            hdf5_path += (
                                ("/mpi" if cuda else "/mpi") if mpi else "/serial"
                            )
                            modules += [
                                (hdf5_module := hdf5_path.split(":")[1])
                                + ("/cuda" if cuda else "/cpu")
                                if mpi
                                else ""
                            ]
                        kokkos_setenvs += [
                            ["Kokkos_ENABLE_CUDA", "ON" if cuda else "OFF"]
                        ]

                        suffix = get_suffix(debug, mpi, cuda, archs)
                        if (kokkos_path := dependencies["kokkos"]).startswith(
                            "module:"
                        ):
                            modules += [
                                (
                                    kokkos_module := (
                                        kokkos_path := kokkos_path + suffix
                                    ).split(":")[1]
                                )
                            ]
                        if (adios2_path := dependencies["adios2"]).startswith(
                            "module:"
                        ):
                            modules += [
                                (
                                    adios2_module := (
                                        adios2_path := adios2_path + suffix
                                    ).split(":")[1]
                                )
                            ]
                        configuration = suffix.upper().replace("/", " @ ")[1:]
                        entity_setenvs = "\n".join(
                            f"{'setenv':<16}{e[0]:<27}{e[1]}" for e in entity_setenvs
                        )
                        kokkos_setenvs = "\n".join(
                            f"{'setenv':<16}{e[0]:<27}{e[1]}" for e in kokkos_setenvs
                        )
                        openmp_setenvs = "\n".join(
                            f"{'setenv':<16}{e[0]:<27}{e[1]}" for e in openmp_setenvs
                        )
                        modules = "\n".join(
                            f"{'module load':<16}" + os.path.expandvars(m)
                            for m in modules
                        )
                        modulefile = pathlib.Path.joinpath(modulepath, suffix[1:])
                        modulefile_content = modulefile_template.substitute(
                            configuration=configuration,
                            entity_setenvs=entity_setenvs,
                            kokkos_setenvs=kokkos_setenvs,
                            openmp_setenvs=openmp_setenvs,
                            modules=modules,
                        )
                        if args.deploy:
                            modulefile.parent.mkdir(parents=True, exist_ok=True)
                            with open(modulefile, "w") as f:
                                f.write(modulefile_content.strip())
                        print(modulefile)
                        if args.verbose or not args.deploy:
                            print(
                                ColoredText(modulefile_content, "nc"),
                                sep="\n",
                            )

                        if args.depends:
                            dlm = " \\\n  "
                            arch_flag = f"--arch {architectures}"
                            with_debug = f"--debug ON{dlm}" if debug else ""
                            with_cc = f"--with-cc {cc_path}{dlm}"
                            with_cuda = (
                                f"--with-cuda {cuda_path}{dlm}"
                                if cuda
                                else f"--with-cuda OFF{dlm}"
                            )
                            with_mpi = (
                                f"--with-mpi {mpi_path}{dlm}"
                                if mpi
                                else f"--with-mpi OFF{dlm}"
                            )
                            with_hdf5 = f"--with-hdf5 {hdf5_path}{dlm}"
                            with_kokkos = f"--with-kokkos {kokkos_path}{dlm}"
                            flags_kokkos = "{with_debug}{with_cc}{with_cuda}{with_mpi}{with_hdf5}{arch_flag}".format(
                                **locals()
                            )
                            flags_adios2 = "{with_debug}{with_cc}{with_cuda}{with_mpi}{with_hdf5}{with_kokkos}{arch_flag}".format(
                                **locals()
                            )
                            dependency_build_scripts += [
                                f"bash compile_kokkos.sh{dlm}{flags_kokkos}"
                            ]
                            dependency_build_scripts += [
                                f"bash compile_adios2.sh{dlm}{flags_adios2}"
                            ]
    if args.depends:
        print(
            ColoredText("Use the following commands to build the dependencies:", "blue")
        )
        print()
        for group in ["kokkos", "adios2"]:
            print(ColoredText(f"{group}", "green"))
            print("---")
            for script in dependency_build_scripts:
                if f"compile_{group}" in script:
                    print(script)
                    print()
