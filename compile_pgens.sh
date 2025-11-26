#!/usr/bin/env bash

pgens=("magnetosphere" "reconnection" "turbulence" "shock" "streaming" "accretion" "wald")
flags=("OFFLINE=ON")

for pgen in "${pgens[@]}"; do
  echo "Compiling pgen: $pgen"
  flags_d="-D pgen=${pgen} "
  for flag in "${flags[@]}"; do
    flags_d+="-D ${flag}"
  done

  (
    cmake -B "builds/build-${pgen}" $flags_d &&
      cmake --build "builds/build-${pgen}" -j &&
      mkdir -p "temp/${pgen}" &&
      cp "builds/build-${pgen}/src/entity.xc" "temp/${pgen}/" &&
      cp "pgens/${pgen}/"*.toml "temp/${pgen}/"
  ) || {
    echo "Failed to compile pgen: $pgen"
    exit 1
  }
done

for pgen in "${pgens[@]}"; do
  cd "temp/${pgen}" || {
    echo "no temp directory for $pgen"
    exit 1
  }
  tomls=$(find . -type f -name "*.toml")
  for toml in "${tomls[@]}"; do
    (
      echo "Running pgen: $pgen with config $toml" &&
        ./entity.xc -input "$toml" &&
        cd ../../
    ) || {
      echo "Failed to run $pgen with config $toml"
      exit 1
    }
  done
done
