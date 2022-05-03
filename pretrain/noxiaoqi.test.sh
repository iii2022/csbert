
# /home/name/.local/lib/python3.6/site-packages/transformers/models/bert
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 run_classifier_noxiaoqi.py \
--model_name_or_path ./ckpt-hasa-noxiaoqi/48/ \
--sleep 0 \
--seed 3 \
--do_test \
--use_ft \
--data_dir /ldata/name/common/hasa_pandn_instance_noxiaoqi/all \
--feature_dir ./feature-hasa-noxiaoqi-test \
--task_name hownet \
--learning_rate 1e-5 \
--max_seq_length 240 \
--max_path_num 10 \
--gradient_accumulation_steps 16 \
--train_batch_size 128 \
--eval_batch_size 512 \
--num_train_epochs 10 \
--name_save_steps 20 \
--output_dir ./ckpt-hasa-noxiaoqi/ &> test.out