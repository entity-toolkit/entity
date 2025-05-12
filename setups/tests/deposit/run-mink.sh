#!/usr/bin/env bash

for i in {1..8}; do
  if [ $i -eq 7 ]; then
    continue
  fi
  run=$(echo "np${i}")
  cp deposit-mink.toml deposit-mink-${run}.toml && \
    sed -i 's/name[[:space:]]*=[[:space:]]*".*\?"/name = "mink-'${run}'"/g' deposit-mink-${run}.toml && \
    mpiexec -np ${i} ./entity.xc -input deposit-mink-${run}.toml && \
    rm deposit-mink-${run}.toml
done

rm *.info *.err *.log *.csv
