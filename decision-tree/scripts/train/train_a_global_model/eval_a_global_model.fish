#!/usr/bin/env bash

. $HOME/NLP/dbdc_venv/bin/activate.fish

mkdir logs

set -l rep_dir $HOME/NLP/dbdc4/dbdc
set -g global_time (date +"%Y%m%d%H%M%S")

set -g global_lang $argv[1]

switch "$global_lang";
  case "en"
    echo Language $global_lang
  case "en.unrevised"
    echo Language $global_lang
  case "en4"
    echo Language $global_lang
  case "jp4"
    echo Language $global_lang
  case "jp4_compe"
    echo Language $global_lang
  case "jp"
    echo Language $global_lang
  case '*'
    echo Unsupported language $global_lang . The program will exit.
    exit 1
end

set -g global_mode $argv[2]

switch "$global_mode";
  case "av"
    echo Mode $global_mode
  case "otx"
    echo Mode $global_mode
  case '*'
    echo Unsupported mode $global_mode . The program will exit.
    exit 1
end

set -l eval_dir $rep_dir"/scripts/baseline/eval/"$global_lang
set -l output_dir $rep_dir"/scripts/train/train_a_global_model/out/"$global_lang"/"$global_mode
set -l file_path "./logs/eval.py."$global_lang"."$global_mode"."$global_time".log"

python $rep_dir"/libraries/dbdc/eval/eval.py" -p $eval_dir -o $output_dir -t 0.0 > $file_path