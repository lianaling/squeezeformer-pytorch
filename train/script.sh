#! /bin/sh
DATASET_PATH="..\data\LibriSpeech\train-clean-100"
MANIFEST_FILE_PATH="..\data\manifests\train-clean-100-proc.csv"

python ../openspeech/openspeech_cli/hydra_train.py \
    dataset=librispeech \
    dataset.dataset_download=False \
    dataset.dataset_path=$DATASET_PATH \
    dataset.manifest_file_path=$MANIFEST_FILE_PATH \
    tokenizer=libri_subword \
    model=conformer_lstm \
    audio=fbank \
    lr_scheduler=warmup_reduce_lr_on_plateau \
    trainer=cpu \
    criterion=cross_entropy