#python3 finetune.py \
torchrun --nproc_per_node=4 finetune.py \
    --base_model 'weiqis/llama-7b-hf' \
    --data_path './alpaca_data_cleaned.json' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]'
