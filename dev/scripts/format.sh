#!/usr/bin/env bash

verify=false

for arg in "$@"; do
  case $arg in
  --verify) verify=true ;;
  esac
done

if $verify; then
  diff_output=""

  if command -v cmake-format &>/dev/null; then
    while IFS= read -r -d '' f; do
      if ! diff -q <(cmake-format "$f") "$f" &>/dev/null; then
        diff_output+="  $f\n"
      fi
    done < <(find cmake/ src/ minimal/ tests/ -type f \( -name "*.cmake" -o -name "*.txt" \) -print0)
  fi

  if command -v clang-format &>/dev/null; then
    while IFS= read -r -d '' f; do
      if ! clang-format --style=file --dry-run --Werror "$f" &>/dev/null; then
        diff_output+="  $f\n"
      fi
    done < <(find pgens/ examples/ tutorials/ src/ minimal/ tests/ -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) -print0)
  fi

  if [ -n "$diff_output" ]; then
    echo "Formatting check failed. The following files need formatting:"
    printf '%s' "$diff_output"
    exit 1
  else
    echo "All files are properly formatted."
  fi
else
  if command -v cmake-format &>/dev/null; then
    find cmake/ src/ minimal/ tests/ \( -type f -name "*.cmake" -o -name "*.txt" \) -exec cmake-format -i {} \;
  fi

  if command -v clang-format &>/dev/null; then
    find pgens/ src/ minimal/ tests/ examples/ pgens/ tutorials/ \( -type f -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) -exec clang-format --style=file -i {} \;
  fi
fi
