#python finetune_clm.py \
torchrun --nproc_per_node=4 finetune_clm.py \
    --base_model 'distilgpt2' \
    --train_data_path './sample_data_train.jsonl' \
    --dev_data_path './sample_data_dev.jsonl' \
    --output_dir './test_output' \
    --batch_size 8 \
    --micro_batch_size 2 \
    --micro_eval_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 1024 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[]'
