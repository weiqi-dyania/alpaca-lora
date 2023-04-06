import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import evaluate

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.0
}

eval_metric = evaluate.load("./pajm.py", completion_format="labeled")

class LoRATrainer(transformers.Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = "cuda" if self.args.local_rank < 0 else f"cuda:{self.args.local_rank}"
        predictions = []
        references = []
        with torch.no_grad():
            for inp, mask, label in zip(inputs['input_ids'], inputs['attention_mask'], inputs['labels']):
                pad_len = (mask == 0).sum()
                pad_and_prompt_len = (label == -100).sum()
                # get prompt tokens and send them to device
                prompt_inp = self._prepare_inputs(inp[pad_len:pad_and_prompt_len].unsqueeze(0))
                output = self.model.generate(input_ids=prompt_inp, **DEFAULT_GENERATION_CONFIG)
                pred = self.tokenizer.decode(output[0][prompt_inp.shape[1]:], skip_special_tokens=True)
                refe = self.tokenizer.decode(label[pad_and_prompt_len:], skip_special_tokens=True)
                predictions.append(pred)
                references.append(refe)
        score = eval_metric.compute(predictions=predictions, references=references)["score"]
        # a temporary hack to squeeze score into loss
        return (torch.tensor(1.0 - score).to(device), None, None)

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: str = "",
    dev_data_path: str = "",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    micro_eval_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    #val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "alpaca-lora",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_data_path: {train_data_path}\n"
            f"dev_data_path: {dev_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"micro_dev_batch_size: {micro_eval_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            #f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    tokenizer_kwargs = {
        "model_max_length": cutoff_len,
        "padding_side": "left",
        "use_fast": False,
    }
    tokenizer = LlamaTokenizer.from_pretrained(base_model, **tokenizer_kwargs)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )


    raw_datasets = load_dataset('json', data_files={"train": train_data_path, "validation": dev_data_path})

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names

    # prepare the tokenize function to tokenize and concatenate prompt and completion
    def tokenize_function(examples):
        # tokenize prompt and completion separately
        prompt_tokens = tokenizer(examples['prompt'])
        completion_tokens = tokenizer(examples['completion'])

        results = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for p_input, p_mask, c_input, c_mask in zip(prompt_tokens['input_ids'],
                                                    prompt_tokens['attention_mask'],
                                                    completion_tokens['input_ids'],
                                                    completion_tokens['attention_mask']):
            # concatenate prompt and completion tokens and attention masks
            results['input_ids'].append(
                (p_input + c_input[1:]  + [tokenizer.eos_token_id])[:cutoff_len]
            )
            results['attention_mask'].append(
                (p_mask + c_mask[1:] + [1])[:cutoff_len]
            )
            # set tokens of the prompt to `IGNORED_LABEL_TOKEN_ID` in labels to avoid calculating loss
            results['labels'].append(
                ([-100] * len(p_input) + c_input[1:] + [tokenizer.eos_token_id])[:cutoff_len]
            )
        return results

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_data = tokenized_datasets["train"]
    val_data = tokenized_datasets["validation"]
    val_set_size = len(val_data)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    #trainer = transformers.Trainer(
    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="epoch" if val_set_size > 0 else "no",
            save_strategy="epoch",
            #eval_steps=200 if val_set_size > 0 else None,
            #save_steps=200,
            output_dir=output_dir,
            #save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            prediction_loss_only=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    tokenizer.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
