#python finetune_clm.py \
torchrun --nproc_per_node=4 finetune_clm.py \
    --base_model '/home/runs/models/alpaca-7b-lora-merged' \
    --train_data_path './alpaca_data_train.jsonl' \
    --dev_data_path './alpaca_data_dev.jsonl' \
    --output_dir './lora-alpaca-clm' \
    --batch_size 92 \
    --micro_batch_size 8 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --cutoff_len 2048 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]'
