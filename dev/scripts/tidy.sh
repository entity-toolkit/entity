#!/usr/bin/env bash
set -u

build_dir=""
file_filter=""
fast_mode=false
use_changed=false
changed_ref=""
verify=false

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
	--changed)
		use_changed=true
		if [[ -n "${2:-}" && "$2" != --* ]]; then
			changed_ref="$2"
			shift 2
		else
			shift
		fi
		;;
	--fast)
		# Skip clang-analyzer-* (inter-procedural analysis; ~5-10x slower than other checks)
		fast_mode=true
		shift
		;;
	--verify)
		verify=true
		shift
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: $0 --build <build_dir> [--files <regex>] [--changed [<ref>]] [--fast] [--verify]"
		exit 1
		;;
	esac
done

if [[ -z "$build_dir" ]]; then
	echo "Error: --build is required"
	exit 1
fi

if [[ "$use_changed" == true && -n "$file_filter" ]]; then
	echo "Error: --changed and --files are mutually exclusive"
	exit 1
fi

out_dir="tidy"
project_root="$(pwd)"
allowed_re="^${project_root}/(src|pgens)/"
jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

# Use cltcache if available for incremental runs (pip install cltcache)
# cltcache requires explicit compiler flags via '--'; it does not support -p <build_dir>.
# We preprocess compile_commands.json once at startup to extract per-file flags.
tidy_bin="clang-tidy"
tidy_prefix=""
_cltcache=""
if command -v cltcache &>/dev/null; then
	_cltcache="cltcache"
elif [[ -x "${project_root}/.venv/bin/cltcache" ]]; then
	_cltcache="${project_root}/.venv/bin/cltcache"
fi
if [[ -n "$_cltcache" ]]; then
	tidy_prefix="$_cltcache"
	echo "Using cltcache"
fi

extra_checks=""
if [[ "$fast_mode" == true ]]; then
	extra_checks="--checks=-clang-analyzer-*"
	echo "Fast mode: skipping clang-analyzer-*"
fi

# Build file_filter from git diff when --changed is given.
# .cpp files match directly; changed headers match all .cpp files in the same directory
# (since clang-tidy has no include graph, this is the best available heuristic).
if [[ "$use_changed" == true ]]; then
	if [[ -n "$changed_ref" ]]; then
		diff_files=$(git diff --name-only "${changed_ref}...HEAD" 2>/dev/null || true)
	else
		diff_files=$(git diff --name-only HEAD 2>/dev/null || true)
	fi

	diff_files=$(echo "$diff_files" | grep -E '\.(cpp|h|hpp)$' || true)

	if [[ -z "$diff_files" ]]; then
		if [[ -z "$changed_ref" ]]; then
			echo "No changed .cpp/.h files in working tree. Try --changed <ref> (e.g. --changed master)."
		else
			echo "No changed .cpp/.h files vs ${changed_ref}."
		fi
		exit 0
	fi

	file_filter=$(
		{
			echo "$diff_files" | grep '\.cpp$' | while IFS= read -r f; do
				printf '%s\n' "${project_root}/${f}" | sed 's/[.]/\\./g'
			done
			echo "$diff_files" | grep -E '\.(h|hpp)$' | while IFS= read -r f; do
				dir="${project_root}/$(dirname "$f")"
				printf '%s/[^/]+\\.cpp\n' "$(printf '%s' "$dir" | sed 's/[.]/\\./g')"
			done
		} | sort -u | paste -sd'|' -
	)
fi

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

total=$(echo "$file_list" | wc -l | tr -d ' ')

# Precompute per-file compiler flags so cltcache can receive them via '--'.
# Uses shlex to handle shell-quoted flags (e.g. "-D FOO=\"bar\"") correctly.
# Strips flags irrelevant to compilation: -o, -c, -MF/-MT/-MQ/-MD/-MMD/-MP,
# and the source file path itself.
flags_db=""
if [[ -n "$tidy_prefix" ]]; then
	flags_db=$(mktemp /tmp/tidy_flags_XXXXXX.json)
	python3 - "$build_dir/compile_commands.json" >"$flags_db" <<'PYEOF'
import json, shlex, sys

SKIP_NEXT = {"-o", "-MF", "-MT", "-MQ"}
SKIP_SELF = {"-MD", "-MMD", "-MP"}

data = json.load(open(sys.argv[1]))
result = {}
for e in data:
    args = shlex.split(e.get("command", ""))
    src = e["file"]
    filtered, skip = [], False
    for a in args[1:]:   # drop compiler binary
        if skip:
            skip = False
            continue
        if a in SKIP_NEXT:
            skip = True
            continue
        if a in SKIP_SELF:
            continue
        filtered.append(a)
    result[src] = filtered

print(json.dumps(result))
PYEOF
fi

# Temp dir: each job touches a file here when done — race-condition-free counter
progress_dir=$(mktemp -d /tmp/tidy_progress_XXXXXX)

# Write a helper script so each parallel job writes its own per-diagnostic-file logs
# without routing through a single serial process
tmpscript=$(mktemp /tmp/tidy_run_XXXXXX.sh)

cleanup() {
	rm -f "$tmpscript" "$flags_db"
	rm -rf "$progress_dir"
}
trap cleanup EXIT

cat >"$tmpscript" <<'ENDSCRIPT'
#!/usr/bin/env bash
file="$1"
build_dir="$2"
out_dir="$3"
project_root="$4"
tidy_bin="$5"
tidy_prefix="${6:-}"
extra_checks="${7:-}"
progress_dir="${8:-}"
flags_db="${9:-}"
allowed_re="^${project_root}/(src|pgens)/"

tmpout=$(mktemp /tmp/tidy_out_XXXXXX)
trap 'rm -f "$tmpout"' EXIT

if [[ -n "$tidy_prefix" && -n "$flags_db" ]]; then
	# cltcache requires explicit compiler flags via '--'
	mapfile -t compile_args < <(jq -r --arg f "$file" '(.[$f] // [])[]' "$flags_db")
	if [[ ${#compile_args[@]} -gt 0 ]]; then
		# shellcheck disable=SC2086
		$tidy_prefix "$tidy_bin" --quiet $extra_checks "$file" -- "${compile_args[@]}" > "$tmpout" 2>&1 || true
	else
		# File not in flags db; fall back to -p (no caching for this file)
		"$tidy_bin" --quiet -p "$build_dir" $extra_checks "$file" > "$tmpout" 2>&1 || true
	fi
else
	# shellcheck disable=SC2086
	$tidy_prefix "$tidy_bin" --quiet -p "$build_dir" $extra_checks "$file" > "$tmpout" 2>&1 || true
fi

# Route each diagnostic to the log for the file it occurs in (not the analyzed file)
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
' "$tmpout"

[[ -n "$progress_dir" ]] && touch "$progress_dir/$$.$RANDOM"
ENDSCRIPT
chmod +x "$tmpscript"

# Progress bar — runs in background, polls the counter dir every 0.2s
bar_full="########################################"
bar_empty="----------------------------------------"
(
	while true; do
		done_n=$(find "$progress_dir" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
		filled=$((done_n * 40 / total))
		printf "\r[%s%s] %d/%d" \
			"${bar_full:0:$filled}" "${bar_empty:0:$((40 - filled))}" "$done_n" "$total"
		[[ "$done_n" -ge "$total" ]] && break
		sleep 0.2
	done
	printf "\n"
) &
progress_pid=$!

echo "$file_list" | xargs -P "$jobs" -I{} \
	"$tmpscript" {} "$build_dir" "$out_dir" "$project_root" "$tidy_bin" "$tidy_prefix" "$extra_checks" "$progress_dir" "$flags_db"

kill "$progress_pid" 2>/dev/null || true
wait "$progress_pid" 2>/dev/null || true
printf "\r[%s] %d/%d\n" "$bar_full" "$total" "$total"

# Dedup (parallel jobs may write overlapping header diagnostics from different TUs)
find "$out_dir" -name '*.log' | while read -r log; do
	awk '!seen[$0]++' "$log" >"$log.tmp" && mv "$log.tmp" "$log"
	[[ ! -s "$log" ]] && rm -f "$log"
done

if [[ "$verify" == true ]]; then
	logs=$(find "$out_dir" -name '*.log' | sort)
	if [[ -z "$logs" ]]; then
		echo "OK: no warnings or errors."
		exit 0
	else
		echo "FAILED: clang-tidy reported issues in:"
		echo "$logs" | sed 's/^/  /'
		exit 1
	fi
else
	echo "Done. Logs written to $out_dir/"
fi
