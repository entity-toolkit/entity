default_with_cuda="module:cudatoolkit/12.0"
declare with_cuda="${default_with_cuda}"

default_with_cc="module:gcc-toolset/10"
declare with_cc="${default_with_cc}"

default_install_prefix="${HOME}/opt"
declare install_prefix="${default_install_prefix}"

default_module_path="${HOME}/opt/.modules"
default_src_path="${HOME}/opt/src"

declare -r modulename_lower=${modulename,,}

declare ${modulename_lower}_module="${default_module_path}/${modulename_lower}"
declare ${modulename_lower}_src_path="${default_src_path}/${modulename_lower}"

declare deploy="OFF"
declare verbose="OFF"

declare -r programname=$0