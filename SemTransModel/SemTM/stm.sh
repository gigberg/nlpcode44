#!/bin/bash
source /home/zhoujiaming/anaconda3/bin/activate hitrans
CUDA_VISIBLE_DEVICES=1

for flow_type in QA_W
do
    for dataset in navigate weather calendar citod
    do
        python main.py --dataset $dataset
    done
done


# for nli in cnli
# do
#     for num in 4
#     do
#         for model in bert # roberta ernie
#         do
#             mkdir ./checkpoints_${model}
#             mkdir ./checkpoints_${model}/cdconv_cgim
#             mkdir ./checkpoints_${model}/cdconv_cgim/${num}class_${nli}

#             for seed in 42 # 23  42 133 233
#             do
#                 for type in intra # role hist
#                 do
#                     save_dir="./checkpoints_${model}/cdconv_cgim/${num}class_${nli}/${seed}"
#                     mkdir $save_dir

#                     if [ -n "$1" ] && [ $1 == "--debug" ]
#                     then
#                         CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m debugpy \
#                             --listen 5679 \
#                             --wait-for-client \
#                             main.py \
#                             --do_train \
#                             --model_name_or_path ${model} \
#                             --save_dir $save_dir \
#                             --train_file ./data/cdconv/${num}class_train.tsv \
#                             --dev_file ./data/cdconv/${num}class_dev.tsv \
#                             --num_classes ${num} \
#                             | tee $save_dir/log_debug.txt
#                     fi

#                     if [ -n "$1" ] && [ $1 == "--train" ]
#                     then
#                         CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
#                             --do_train \
#                             --model_name_or_path ${model} \
#                             --save_dir $save_dir \
#                             --train_file ./data/cdconv/${num}class_train.tsv \
#                             --dev_file ./data/cdconv/${num}class_dev.tsv \
#                             --num_classes ${num} \
#                             | tee -a $save_dir/log_train.txt
#                     fi

#                     if [ -n "$1" ] && [ $1 == "--test" ]
#                     then
#                         CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
#                             --do_test \
#                             --model_name_or_path ${model} \
#                             --init_from_ckpt  ${save_dir}/best_model/model_state.pdparams \
#                             --test_file ./data/cdconv/${num}class_test.tsv \
#                             --num_classes ${num} \
#                             | tee -a $save_dir/log_test.txt
#                     fi
#                 done
#             done
#         done
#     done
# done