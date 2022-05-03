
# 3e-5 synonym
# /home/name/.local/lib/python3.6/site-packages/transformers/models/bert
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 run_classifier_noxiaoqi.py \
--model_name_or_path bert-base-uncased \
--classifier_path 12 \
--sleep 0 \
--seed 3 \
--do_train \
--do_eval \
--do_test \
--data_dir /ldata/name/common/MannerOf_pandn_instance_noxiaoqi/ \
--feature_dir ./feature-mannerof-noxiaoqi \
--task_name hownet \
--learning_rate 3e-5 \
--max_seq_length 150 \
--max_path_num 10 \
--gradient_accumulation_steps 1 \
--train_batch_size 600 \
--eval_batch_size 600 \
--num_train_epochs 20 \
--name_save_steps 20 \
--output_dir ./ckpt-mannerof-noxiaoqi-mt-0.003/ 