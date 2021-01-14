export CUDA_VISIBLE_DEVICES=2,3

DATA_PATH=/home/sichenglei/sst-2
PWWS_DATA_PATH=/home/sichenglei/sst-2/pwws-rbt
TF_DATA_PATH=/home/sichenglei/sst-2/textfooler-rbt
# DATA_PATH=/home/sichenglei/contrast_imdb_ori/textfooler
# DATA_PATH=/home/sichenglei/sst-2
MODEL_PATH=/home/sichenglei/roberta-base
# ALPHA=0.4
EPOCHS=5
# OUTPUT_PATH=/data2/private/clsi/bert-imdb-iterADASmix-textfooler
SEQLEN=128

# mix_option: 0: no mix, 1: TMix, 2: SimMix




## BERT-TMixADA-PWWS
python run_simMix.py \
--model_type roberta \
--mix_type tmix \
--iterative true \
--attacker pwws \
--num_adv 300 \
--task_name sst-2 \
--data_dir ${DATA_PATH} \
--model_name_or_path ${MODEL_PATH} \
--output_dir /data1/private/clsi/rbt-sst-tmixada-pwws-iterative \
--max_seq_length $SEQLEN \
--mix-layers-set 7 9 12 \
--alpha 2.0 \
--num_labels 2 \
--do_lower_case \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
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
--model_name_or_path /data2/private/clsi/rbt-sst-tmixada-pwws/final-checkpoint \
--model_type roberta \
--attacker pwws \
--data_dir /home/sichenglei/sst-2/test.tsv \
--max_seq_len $SEQLEN






## BERT-TMixADA-Textfooler
python run_simMix.py \
--model_type roberta \
--mix_type tmix \
--iterative true \
--attacker textfooler \
--num_adv 1500 \
--task_name sst-2 \
--data_dir ${DATA_PATH} \
--model_name_or_path ${MODEL_PATH} \
--output_dir /data1/private/clsi/rbt-sst-tmixada-textfooler-iterative \
--max_seq_length $SEQLEN \
--mix-layers-set 7 9 12 \
--alpha 0.4 \
--num_labels 2 \
--do_lower_case \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
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
--model_name_or_path /data1/private/clsi/rbt-sst-tmixada-textfooler-iterative/final-checkpoint \
--model_type roberta \
--attacker textfooler \
--data_dir /home/sichenglei/sst-2/test.tsv \
--max_seq_len $SEQLEN






