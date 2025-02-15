import logging
import torch
from torch.utils.data import Dataset 
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer 
from transformers import DataCollatorForLanguageModeling 
from datasets import load_dataset 
import numpy as np 

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenizer, max_length=1024, split="train",nrecs=None):
        self.tokenizer = tokenizer 
        self.max_length = max_length # Load TinyStories dataset 
        if nrecs:
            tmp = load_dataset("roneneldan/TinyStories")
            self.dataset = tmp[split].select(range(nrecs))
        else:
            self.dataset = load_dataset("roneneldan/TinyStories",split=split)

    def __len__(self):
        return len(self.dataset)
    
    def chunk_text(self, text): # Tokenize the full text 
        tokens = self.tokenizer.encode(text) # Split into chunks of max_length - 2 to account for special tokens 
        chunk_size = self.max_length - 2 
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)] 
        # Process each chunk to ensure proper format 
        processed_chunks = [] 
        for chunk in chunks: 
            if len(chunk) > 0: # Only process non-empty chunks 
                chunk = chunk[:chunk_size] # Ensure we don't exceed max length 
                processed_chunks.append(chunk)
        return processed_chunks 
    
    def __getitem__(self, idx): 
        story = self.dataset[idx]["text"] 
        chunks = self.chunk_text(story) # If no valid chunks, return a minimal valid input 
        if not chunks: 
            print("Not Chunks")
            return {
            "input_ids": torch.tensor(self.tokenizer.encode(".", max_length=self.max_length, truncation=True)),
            #"attention_mask": torch.tensor([1] * len(self.tokenizer.encode("."))) 
        } 
        # Randomly select one chunk for training 
        #chunk = chunks[np.random.randint(0, len(chunks))] 
        chunk = chunks[0]

        # # Create attention mask 
        # attention_mask = [1] * len(chunk) 
        # Convert to tensor 
        input_ids = torch.tensor(chunk) 
        #attention_mask = torch.tensor(attention_mask) 
        return { "input_ids": input_ids} #, "attention_mask": attention_mask }
    
def train_model(mname,tokenizer,model,train_dataset):  
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Add padding token to tokenizer 
    tokenizer.pad_token = tokenizer.eos_token 

    # Create data collator 
    data_collator = DataCollatorForLanguageModeling( tokenizer=tokenizer, mlm=False )

    # Define training arguments 
    training_args = TrainingArguments( 
        output_dir="./out",
        overwrite_output_dir=True, 
        num_train_epochs=3, 
        per_device_train_batch_size=8, 
        #per_device_eval_batch_size=4, 
        eval_steps=1000, 
        save_steps=1000, 
        warmup_steps=500, 
        learning_rate=5e-5, 
        lr_scheduler_type="cosine", 
        optim="adamw_torch", 
        weight_decay=0.01, 
        logging_dir='./logs', 
        logging_steps=100, 
        eval_strategy="no", 
        save_total_limit=2, 
        #load_best_model_at_end=True, 
        report_to="none", # "wandb",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        torch_compile=True,
    ) 

    # # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to('cuda')

   
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, 
                        train_dataset=train_dataset)
    
    
    #, eval_dataset=eval_dataset )
     
    # Train the model 
    trainer.train() 

    # Save the final model 
    trainer.save_model(f"./out/{mname}-final") 
    tokenizer.save_pretrained(f"./out/{mname}-final")

if __name__ == "__main__": 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
    config = GPT2Config(
        vocab_size=50257,  # Adjust as needed (default is 50257)
        n_positions=1024,   # Adjust sequence length (default is 1024)
        n_embd=512,       # Adjust embedding dimension (default is 768)
        n_layer=8,       # Adjust number of layers (default is 12)
        n_head=8,        # Adjust number of attention heads (default is 12)
        # ... other config parameters you want to customize ...
        # Example:
        attn_pdrop=0.1,    # Attention dropout
        resid_pdrop=0.1,   # Residual dropout
        # ... and many more!  Refer to the GPT2Config documentation
        #     for the full list of configurable parameters.
    )

    model = GPT2LMHeadModel(config=config)

    print(model.config)
    print(f"Number of parameters: {model.num_parameters()}")

    train_dataset=TinyStoriesDataset(tokenizer,split="train",nrecs=40000)
    #eval_dataset=TinyStoriesDataset(tokenizer,split="validation")
    train_model("gpt2_8_512",tokenizer,model,train_dataset)