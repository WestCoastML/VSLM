from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gc
import math
import os
import torch
from transformers import Trainer, TrainingArguments,DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel, EvalPrediction
import wandb
from utils import AsciiTokenizer, DynamicEvalCallback, preprocess_and_cache_dataset, save_prior_version, save_tokenizer_model
import numpy as np


def train(dataset, tokenizer, model,
            rundir="runs",
            logging_dir="./logs",
            num_epochs=1,
            per_device_batch_size=8,
            gradient_accumulation_steps=4,
            fp16=True,
            name="{repr(model)}_{config.dataset_name}",
            project='newmodel',
            eval_steps=500,
            max_length=512,
            run="{model_name}_{lef}",
            ):



    # Training arguments
    training_args = TrainingArguments(
        output_dir="temp",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=500,  # Adjust as needed
        weight_decay=0.01,  # Adjust as needed
        logging_dir=logging_dir,
        logging_steps=100,
        save_steps=1000,
        fp16=fp16,
        dataloader_num_workers=4,  # Adjust as needed
        eval_strategy="no",  # Set evaluation strategy to steps
        # eval_steps=eval_steps, 
        report_to=['wandb'],
        remove_unused_columns=False,
        #metric_for_best_model = "eval_loss",
        #load_best_model_at_end=True,
        save_total_limit=2,
        # prediction_loss_only=True
    )

    print(f"{training_args}")
    config = training_args.to_dict()
    wrapped_model=model.module if hasattr(model, 'module') else model
    config.update({
        "max_length": max_length,
        "tokenizer_name": tokenizer.__class__.__name__,
        "model_name": wrapped_model.__class__.__name__,
        "dataset_name": dataset.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": wrapped_model.config.n_embd,
        "num_layers": wrapped_model.config.n_layer,
        "num_heads": wrapped_model.config.n_head,        
    })

    # Initialize wandb
    try:
        lef="{num_layers}_{embed_dim}_{num_heads}".format(**config)
        name = run.format(**config,lef=lef)
    except Exception as e:
        name = f"{config.model_name}_{lef}"
    print(f"run name {name}")

    wandb.init(project=project, config=config, name=name)

    # manage the output directory
    outdir=os.path.join(rundir,name)
    save_prior_version(outdir)
    # save config of tokenizer and model for reconstruction
    save_tokenizer_model(outdir, tokenizer, model)
    # This is a little tricky because we want the logs to be in a directory with the run name
    # but the run name is created using the training_args. So we create a new trainging_args
    # with the updated output_dir
    training_args = TrainingArguments(**{**training_args.to_dict(), "output_dir": outdir})

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],  # Access the 'train' split
        eval_dataset=dataset["validation"],  # Access the 'validation' split
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        #callbacks=[DynamicEvalCallback()],
        # compute_metrics=compute_metrics,  
    )

    # Start training
    print("Skipping training")
    trainer.train()
    # Free up memory
    print("Clean up after training")
    wandb.finish()



def run_experiment(n_layer, n_embd, n_head):
    print(f'run experiment {n_layer=},{n_embd=},{n_head=}')
    max_length=512
    tokenizer=AsciiTokenizer()
    dataset = preprocess_and_cache_dataset("roneneldan/TinyStories",tokenizer,max_length=max_length,reload=True)

    config = GPT2Config(vocab_size=tokenizer.vocab_size,
                        n_embd=n_embd,
                        n_layer=n_layer,
                        n_head=n_head,
                        n_positions=max_length)
    model = GPT2LMHeadModel(config)

    # Move the model to the primary GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Wrap the model with DataParallel
    model = torch.nn.DataParallel(model)

    print(f" train {n_layer=},{n_embd=},{n_head=}")
    train(dataset,tokenizer,model,per_device_batch_size=128,max_length=max_length)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Running experiments")
    tokenizer = AsciiTokenizer()
    dataset_name="roneneldan/TinyStories"
    dataset = preprocess_and_cache_dataset(dataset_name,tokenizer)

    experiments = [
        (2,128,2),
        (3,192,3),
        (4,256,4),
        (5,320,5),
        (6,384,6),
        (7,448,7),
        (8,512,8),
        (9,576,9),
        (10,640,10),
        (11,704,11),
        (12,768,12),
        (13,832,13),
        (14,896,14),
        (15,960,15),
        (16,1024,16),
    ]

    for i, (n_layer, n_embd, n_head) in enumerate(experiments):
        run_experiment(n_layer, n_embd, n_head)
    #     for future in futures:
    # # Run experiments in parallel using ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = [executor.submit(run_experiment, n_layer, n_embd, n_head, i % 2) for i, (n_layer, n_embd, n_head) in enumerate(experiments)]
    #     for future in futures:
    #         future.result()