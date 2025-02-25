#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="path/to/trainging_data"
EVAL_DATA="path/to/test_data"
MODEL="openbmb/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm
DS_CONFIG="ds_config_zero3.json" #, use zero3 for v2.5

# override
# geminipro 1.5 on agiqa-3k
OUTPUT_DIR="output/gemini_agiqa-3k_lora_07_31_2024_lr2e-6_bf16"    # this is the gemini pro1.5 on agiqa-3k
DATA="./data_processed/agiqa-3k/directly_train_cls_minicpm_quality_multiround.jsonl"
EVAL_DATA="./data_processed/agiqa-3k/directly_test_cls_minicpm_quality_multiround.jsonl"


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length 512 \
    --max_slice_nums 9 \
    --max_steps 22000 \
    --eval_steps 400 \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 20 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed $DS_CONFIG \
    --report_to "tensorboard" # wandb
