#!/usr/bin/env bash

# Copyright 2021 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0


 . utils/parse_options.sh || exit 1;


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir ${FISHER_CALLHOME_SPANISH}
if [ -z "${FISHER_CALLHOME_SPANISH}" ]; then
    log "Fill the value of 'FISHER_CALLHOME_SPANISH' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp.es
train_set_prefix=train_sp
train_dev=train_dev.es
recog_set="fisher_dev.es fisher_dev2.es fisher_test.es callhome_devtest.es callhome_evltest.es"

# Assume the file structures as
# - ${FISHER_CALLHOME_SPANISH}
#     - LDC2010S01 # (for fisher speech)
#     - LDC2010T04 # (for fisher transcripts)
#     - LDC96S35   # (for callhome speech)
#     - LDC96T17   # (for callhome transcripts)

sfisher_speech=${FISHER_CALLHOME_SPANISH}/LDC2010S01
sfisher_transcript=${FISHER_CALLHOME_SPANISH}/LDC2010T04
sfisher_transcripts=/export/corpora/LDC/LDC2010T04
split=local/splits/split_fisher
callhome_speech=${FISHER_CALLHOME_SPANISH}/LDC96S35
callhome_transcripts=${FISHER_CALLHOME_SPANISH}/LDC96T17
split_callhome=local/splits/split_callhome


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Make sure you have fisher_callhome_spanish at ${sfisher_speech}, ${sfisher_transcript}, \
             ${callhome_speech}, ${callhome_transcripts}"
    log "stage 0: Data Preparation"
    local/fsp_data_prep.sh ${sfisher_speech} ${sfisher_transcripts}
    local/callhome_data_prep.sh ${callhome_speech} ${callhome_transcripts}

    # split data
    local/create_splits.sh ${split}
    local/callhome_create_splits.sh ${split_callhome}

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Concatenate Multiple Utterances"

    # concatenate multiple utterances
    local/normalize_trans.sh ${sfisher_transcripts} ${callhome_transcripts}
fi


