import torch
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset  # Type hint for dataset

def fine_tune_model(model, dataset: Dataset,  lora_config: LoraConfig, training_args: TrainingArguments, tokenizer=None):
    """Fine-tunes a language model using PEFT and LoRA.

    Args:
        model: The pre-trained Hugging Face model (e.g., AutoModelForCausalLM instance).
        dataset: The training dataset (Hugging Face Dataset object).  Must have a "text" column.
        lora_config: The PEFT LoRA configuration (LoraConfig instance).
        training_args: The Hugging Face TrainingArguments.
        tokenizer: The tokenizer to use. If none is provided the model's tokenizer will be used.

    Returns:
        The fine-tuned PEFT model.
    """

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)  # Adjust max length as needed

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    return model, tokenizer  # Return the PEFT model


# Example usage:
from transformers import AutoModelForCausalLM

# 1. Load your pre-trained model
model_name = "lmsys/vicuna-7b-v1.5"  # Replace with your model
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

# 2. Load your dataset
dataset_name = "your_dataset_name"  # Replace with your dataset
dataset = load_dataset(dataset_name, split="train")

# 3. Configure PEFT and Training
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

training_args = TrainingArguments(
    output_dir="fine_tuned_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    push_to_hub=False,
)

# 4. Fine-tune the model
fine_tuned_model, tokenizer = fine_tune_model(model, dataset, lora_config, training_args)

# 5. Save the fine-tuned model
fine_tuned_model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")