#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=it # de, en, es, fr, it, nl, pt, ru
train_set="tr_${lang}"
valid_set="dt_${lang}"
test_sets="dt_${lang} et_${lang}"

# asr_config=conf/train_asr_rnn.yaml
# asr_config=conf/tuning/transducer/train_asr_rnn_transducer.yaml
# asr_config=conf/tuning/transducer/train_asr_rnn_transducer_win400_hop160.yaml
# asr_config=conf/tuning/transducer/train_asr_rnn_transducer_adam.yaml
asr_config=conf/tuning/transducer/train_asr_rnn_transducer_adam_aux.yaml
inference_config=conf/tuning/transducer/decode_transducer_default.yaml

# FIXME(kamo):
# The results with norm_vars=True is odd.
# I'm not sure this is due to bug.
    # --stop_stage 11 \
    # --inference_nj 1 \

./asr.sh \
    --stage 11 \
    --ngpu 8 \
    --lang "${lang}" \
    --local_data_opts "--lang ${lang}" \
    --use_lm false \
    --token_type char \
    --feats_type raw \
    --asr_args "--normalize_conf norm_vars=False " \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model valid.loss.best.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" "$@"
