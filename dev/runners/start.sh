#!/bin/bash

./config.sh --unattended --url https://github.com/entity-toolkit/entity --token ${TOKEN} --labels ${LABEL}

if [[ ${LABEL} == "amd-gpu" ]]; then
  echo "AMD GPU runner detected"
  export HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
else
  echo "Non-AMD runner"
fi

cleanup() {
  echo "Removing runner..."
  ./config.sh remove --unattended --token ${TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

./run.sh &
wait $!
