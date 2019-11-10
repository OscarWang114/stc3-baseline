#!/usr/local/bin/fish

. $HOME/NLP/dbdc_venv/bin/activate.fish

echo (which python)
echo (which pip)
echo $PYTHONPATH
echo (python --version)

set -l rep_dir $HOME/NLP/dbdc4/dbdc
cd $rep_dir/scripts/train/train_a_global_model

set -g global_lang $argv[1]

switch "$global_lang";
  case "en"
    echo Language $global_lang
    set -g model_path $rep_dir"/models/word2vec/GoogleNews-vectors-negative300.bin"
    set -g model_types "global" # "CIC" "YI" "TickTock" "IRIS"
  case "en4"
    echo Language $global_lang
    set -g model_path $rep_dir"/models/word2vec/GoogleNews-vectors-negative300.bin"
    set -g model_types "global"
  case "en.unrevised"
    echo Language $global_lang
    set -g model_path $rep_dir"/models/word2vec/GoogleNews-vectors-negative300.bin"
    set -g model_types "global.unrevised" # "CIC.unrevised" "YI.unrevised" "TickTock.unrevised" "IRIS.unrevised"
  case "jp"
    echo Language $global_lang
    set -g model_path $rep_dir"/models/word2vec/jawiki.bin"
    set -g model_types "global"
  case "jp4"
    echo Language $global_lang
    set -g model_path $rep_dir"/models/word2vec/jawiki.bin"
    set -g model_types "global"
  case "jp4_compe"
    echo Language $global_lang
    set -g model_path $rep_dir"/models/word2vec/jawiki.bin"
    set -g model_types "global"
  case '*'
    echo Unsupported language $global_lang . The program will exit.
    exit 1
end

set -l dbdc $argv[2]

switch "$dbdc";
  case "3"
    set -g dev_dir  $rep_dir"/data/dbdc3/dev"
    set -g eval_dir $rep_dir"/data/dbdc3/evaluation_data_with_reference_labels"
  case "4"
    set -g dev_dir  $rep_dir"/data/dbdc4/dev"
    set -g eval_dir $rep_dir"/data/dbdc4/eval_allOs"
  case '*'
    echo Unsupported dbdc $dbdc . The program will exit.
    exit 1
end


set -g global_output_dir $rep_dir"/results/"$global_lang
mkdir -p $global_output_dir

function predict
  set -l rep_dir $HOME/NLP/dbdc4/dbdc
  set -l model_type $argv[1]
  set -l lang $argv[2]
  set -l output_dir $argv[3]

  echo "python predict_a_global_model.py --model-type "$model_type
  env PYTHONPATH=$rep_dir/libraries \
    python predict_a_global_model.py \
      --dev-dir $dev_dir \
      --eval-dir $eval_dir \
      --lang $lang \
      --model-path $model_path \
      --max-bef 3 \
      --keyword-n 10 \
      --model-dir $rep_dir"/scripts/train/train_a_global_model/models"/$lang \
      --model-type $model_type \
      --output-dir $output_dir \
      --log-mode a \
      --log-dir "logs"

end

for model_type in $model_types
  predict $model_type $global_lang $global_output_dir
end
