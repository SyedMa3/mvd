#!/bin/bash
GPUS=2
OUTPUT_DIR='OUTPUT/mvd_vit_base_with_vit_large_teacher_k400_epoch_400/cross-val_finetune_on_all_cases'
MODEL_PATH='OUTPUT/mvd_vit_base_with_vit_large_teacher_k400_epoch_400/checkpoint-399.pth'
DATA_PATH='data/Annotations'
DATA_ROOT='data'

# train on 16 V100 GPUs (2 nodes x 8 GPUs)
torchrun --nproc_per_node=2 \
    --nnodes=1 \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set SSV2 --nb_classes 11 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size 24 --update_freq 1 --num_sample 2 \
    --save_ckpt_freq 25 \
    --num_frames 16 \
    --lr 5e-4 --epochs 100 \
    --dist_eval --test_num_segment 2 --test_num_crop 3 \
    --enable_deepspeed \
    --use_checkpoint
    # --no_auto_resume

# deepspeed --master_port 13835 run_class_finetuning.py \
#     --model vit_base_patch16_224 \
#     --data_set SSV2 --nb_classes 11 \
#     --data_path ${DATA_PATH} \
#     --data_root ${DATA_ROOT} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --input_size 224 --short_side_size 224 \
#     --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
#     --batch_size 24 --update_freq 1 --num_sample 2 \
#     --save_ckpt_freq 5 --no_save_best_ckpt \
#     --num_frames 16 \
#     --lr 5e-4 --epochs 10 \
#     --dist_eval --test_num_segment 2 --test_num_crop 3 \
#     --use_checkpoint \
#     --enable_deepspeed
