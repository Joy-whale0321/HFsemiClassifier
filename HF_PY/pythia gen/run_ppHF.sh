#!/bin/bash

source /opt/sphenix/core/bin/sphenix_setup.sh -n new
# Additional commands for my local environment
export SPHENIX=/sphenix/u/jzhang1
export MYINSTALL=$SPHENIX/install
source /opt/sphenix/core/bin/setup_local.sh $MYINSTALL

date

cd /sphenix/user/jzhang1/HFsemiClassifier/HF_PY/Generate

nEvents=${1:-10000}
PYSet=${2:-py_HF.cmnd}
outName=${3:-ppHF_eXDecay_test.root}
Process=${4:-12345}

now=$(date +%s)
pid=$$
RandSeed=$(( (now + Process * 1000 + pid) % 9000000 ))
if [ "$RandSeed" -le 0 ]; then
  RandSeed=$((RandSeed + 9000000))
fi

./ppHF_eXDecay "${nEvents}" "${PYSet}" "${outName}" "${RandSeed}"

date
