#!/bin/bash

source ~/.bash_profile
conda activate stc3

echo $(which python)
echo $(which pip)
echo $PYTHONPATH
echo $(python --version)

rep_dir=$HOME/NLP/stc3-baseline/decision-tree
data_dir=$HOME/NLP/stc3-baseline/stc3dataset/data
cd ${rep_dir}/scripts/train/train_a_global_model

lang=$1

case "$lang" in
  "en" ) echo Language $lang; model_path=$rep_dir"/models/word2vec/GoogleNews-vectors-negative300.bin" ;;
  "zh" ) echo Language $lang; model_path=$rep_dir"/models/word2vec/jawiki.bin" ;;
  * ) echo Unsupported language $lang . The program will exit.; exit 1 ;;
esac

# set -l output_dir "outputs/optimize_threashold/dev/"$lang
# mkdir -p $output_dir

mkdir "logs"

echo "python train_a_global_model.py"
env PYTHONPATH=$rep_dir/libraries \
  python train_a_global_model.py \
    --dev-dir $data_dir \
    --eval-dir "" \
    --lang $lang \
    --model-path $model_path\
    --max-bef 3 \
    --keyword-n 10 \
    --save-dir $rep_dir"/scripts/train/train_a_global_model/models/"$lang \
    --log-mode a \
    --log-dir "logs"
