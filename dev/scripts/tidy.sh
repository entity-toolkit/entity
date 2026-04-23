#!/usr/bin/env bash
set -u

build_dir=""
file_filter=""

while [[ $# -gt 0 ]]; do
	case "$1" in
	--build)
		if [[ -z "${2:-}" || "$2" == --* ]]; then
			echo "Error: --build requires a value"
			exit 1
		fi
		build_dir="$2"
		shift 2
		;;
	--files)
		if [[ -z "${2:-}" || "$2" == --* ]]; then
			echo "Error: --files requires a value"
			exit 1
		fi
		file_filter="$2"
		shift 2
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: $0 --build <build_dir> [--files <regex>]"
		exit 1
		;;
	esac
done

if [[ -z "$build_dir" ]]; then
	echo "Error: --build is required"
	exit 1
fi

out_dir="tidy"
project_root="$(pwd)"
allowed_re="^${project_root}/(src|pgens)/"
jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

rm -rf "$out_dir"
mkdir -p "$out_dir"

if [[ -n "$file_filter" ]]; then
	file_list=$(jq -r '.[].file' "$build_dir/compile_commands.json" | sort -u | grep -E "$file_filter" || true)
	if [[ -z "$file_list" ]]; then
		echo "No files matched: $file_filter"
		exit 1
	fi
	echo "Matched files:"
	echo "$file_list" | sed 's/^/  /'
else
	file_list=$(jq -r '.[].file' "$build_dir/compile_commands.json" | sort -u)
fi

echo "$file_list" |
	xargs -P "$jobs" -I{} clang-tidy -p "$build_dir" {} 2>&1 |
	awk -v out="$out_dir" -v root="$project_root/" -v allowed="$allowed_re" '
    /^\/?[^:]+:[0-9]+:[0-9]+:/ {
      match($0, /^[^:]+/)
      current_file = substr($0, 1, RLENGTH)
      if (current_file !~ allowed) {
        current_file = ""
        next
      }
      if (index(current_file, root) == 1) {
        current_file = substr(current_file, length(root) + 1)
      }
      logfile = out "/" current_file ".log"
      cmd = "mkdir -p \"$(dirname \"" logfile "\")\""
      system(cmd)
      print $0 >> logfile
      next
    }
    current_file != "" {
      logfile = out "/" current_file ".log"
      print $0 >> logfile
    }
  '

find "$out_dir" -name '*.log' | while read log; do
	awk '!seen[$0]++' "$log" >"$log.tmp" && mv "$log.tmp" "$log"
done

echo "Done. Logs written to $out_dir/"
