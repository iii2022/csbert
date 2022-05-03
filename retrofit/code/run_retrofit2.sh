# train_batch_size是总的batch不是per device
# 1e-5, 10 epoch是最好的设置
# 3e-5 synonym
# /home/name/.local/lib/python3.6/site-packages/transformers/models/bert
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 retrofit2.py \
--model_name_or_path n \
--sleep 0 \
--seed 3 \
--do_train \
--do_eval \
--do_test \
--data_dir data-xiaoqi-2loss \
--feature_dir ./feature-2-xiaoqi-2loss \
--task_name hownet \
--learning_rate 2e-3 \
--max_seq_length 10 \
--gradient_accumulation_steps 1 \
--train_batch_size 512 \
--eval_batch_size 128 \
--num_train_epochs 100 \
--name_save_steps 100 \
--output_dir ./ckpt-2-xiaoqi-2loss/ 