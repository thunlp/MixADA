export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=/home/sichenglei/imdb
PWWS_DATA_PATH=/home/sichenglei/imdb/pwws-bert
TF_DATA_PATH=/home/sichenglei/imdb/textfooler-bert
# DATA_PATH=/home/sichenglei/contrast_imdb_ori/textfooler
# DATA_PATH=/home/sichenglei/sst-2
MODEL_PATH=/home/sichenglei/bert-base-uncased
# ALPHA=0.4
EPOCHS=5
# OUTPUT_PATH=/data2/private/clsi/bert-imdb-iterADASmix-textfooler
SEQLEN=256

# mix_option: 0: no mix, 1: TMix, 2: SimMix


# BERT-ADA-PWWS
python run_simMix.py \
--model_type bert \
--mix_type nomix \
--adv_ratio 0.50 \
--task_name sst-2 \
--data_dir ${DATA_PATH} \
--second_data_dir $PWWS_DATA_PATH \
--model_name_or_path ${MODEL_PATH} \
--output_dir /data2/private/clsi/bert-imdb-ada-pwws-new \
--max_seq_length $SEQLEN \
--mix-layers-set 7 9 12 \
--alpha 0.4 \
--num_labels 2 \
--do_lower_case \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--weight_decay 0.0 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--num_train_epochs $EPOCHS \
--warmup_steps 0 \
--logging_steps 200 \
--eval_all_checkpoints \
--seed 2020 \
--overwrite_output_dir \
--overwrite_cache \
--do_train \
--fp16 
# --second_data_dir $SECOND_DATA_PATH \
# --third_data_dir $THIRD_DATA_PATH \



python attackEval.py  \
--model_name_or_path /data2/private/clsi/bert-imdb-ada-pwws-new/final-checkpoint \
--model_type bert \
--attacker pwws \
--data_dir /home/sichenglei/contrast_imdb_ori/test.tsv \
--max_seq_len $SEQLEN



## BERT-ADA-TF
python run_simMix.py \
--model_type bert \
--mix_type nomix \
--adv_ratio 0.75 \
--task_name sst-2 \
--data_dir ${DATA_PATH} \
--second_data_dir $TF_DATA_PATH \
--model_name_or_path ${MODEL_PATH} \
--output_dir /data2/private/clsi/bert-imdb-ada-textfooler-new \
--max_seq_length $SEQLEN \
--mix-layers-set 7 9 12 \
--alpha 0.4 \
--num_labels 2 \
--do_lower_case \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--weight_decay 0.0 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--num_train_epochs $EPOCHS \
--warmup_steps 0 \
--logging_steps 200 \
--eval_all_checkpoints \
--seed 2020 \
--overwrite_output_dir \
--overwrite_cache \
--do_train \
--fp16 
# --second_data_dir $SECOND_DATA_PATH \
# --third_data_dir $THIRD_DATA_PATH \



python attackEval.py  \
--model_name_or_path /data2/private/clsi/bert-imdb-ada-textfooler-new/final-checkpoint \
--model_type bert \
--attacker textfooler \
--data_dir /home/sichenglei/contrast_imdb_ori/test.tsv \
--max_seq_len $SEQLEN







## BERT-TMixADA-PWWS
python run_simMix.py \
--model_type bert \
--mix_type tmix \
--adv_ratio 0.50 \
--task_name sst-2 \
--data_dir ${DATA_PATH} \
--second_data_dir $PWWS_DATA_PATH \
--model_name_or_path ${MODEL_PATH} \
--output_dir /data2/private/clsi/bert-imdb-tmixada-pwws-new \
--max_seq_length $SEQLEN \
--mix-layers-set 7 9 12 \
--alpha 8.0 \
--num_labels 2 \
--do_lower_case \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--weight_decay 0.0 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--num_train_epochs $EPOCHS \
--warmup_steps 0 \
--logging_steps 200 \
--eval_all_checkpoints \
--seed 2020 \
--overwrite_output_dir \
--overwrite_cache \
--do_train \
--fp16 
# --second_data_dir $SECOND_DATA_PATH \
# --third_data_dir $THIRD_DATA_PATH \



python attackEval.py  \
--model_name_or_path /data2/private/clsi/bert-imdb-tmixada-pwws-new/final-checkpoint \
--model_type bert \
--attacker pwws \
--data_dir /home/sichenglei/contrast_imdb_ori/test.tsv \
--max_seq_len $SEQLEN



## BERT-TMixADA-TF
python run_simMix.py \
--model_type bert \
--mix_type tmix \
--adv_ratio 0.75 \
--task_name sst-2 \
--data_dir ${DATA_PATH} \
--second_data_dir $TF_DATA_PATH \
--model_name_or_path ${MODEL_PATH} \
--output_dir /data2/private/clsi/bert-imdb-tmixada-textfooler-new \
--max_seq_length $SEQLEN \
--mix-layers-set 7 9 12 \
--alpha 0.2 \
--num_labels 2 \
--do_lower_case \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--weight_decay 0.0 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--num_train_epochs $EPOCHS \
--warmup_steps 0 \
--logging_steps 200 \
--eval_all_checkpoints \
--seed 2020 \
--overwrite_output_dir \
--overwrite_cache \
--do_train \
--fp16 
# --second_data_dir $SECOND_DATA_PATH \
# --third_data_dir $THIRD_DATA_PATH \



python attackEval.py  \
--model_name_or_path /data2/private/clsi/bert-imdb-tmixada-textfooler-new/final-checkpoint \
--model_type bert \
--attacker textfooler \
--data_dir /home/sichenglei/contrast_imdb_ori/test.tsv \
--max_seq_len $SEQLEN








