
# 3e-5 synonym
# /home/name/.local/lib/python3.6/site-packages/transformers/models/bert
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_classifier_noxiaoqi.py \
--model_name_or_path ./ckpt-causes-clean/260/ \
--classifier_path 12 \
--sleep 0 \
--seed 3 \
--do_test \
--data_dir /ldata/name/common/Causes_pandn_instance_noxiaoqi/all/ \
--feature_dir ./feature-causes-onlytriple \
--task_name hownet \
--learning_rate 3e-5 \
--max_seq_length 150 \
--max_path_num 10 \
--gradient_accumulation_steps 1 \
--train_batch_size 600 \
--eval_batch_size 1200 \
--num_train_epochs 20 \
--name_save_steps 20 \
--output_dir ./predictions/causes-onlytriple/