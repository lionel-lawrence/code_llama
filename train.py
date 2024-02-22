import torch
from config import *
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import default_data_collator, Trainer, TrainingArguments
import copy

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, torch_dtype=torch.bfloat16, device_map='auto')

class NSText2SQLDataset(Dataset):
    def __init__(self, size=None, max_seq_length=2048):
        self.dataset = load_dataset("NumbersStation/NSText2SQL",split="train")
        if size:
            self.dataset = self.dataset.select(range(size))
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instruction = torch.tensor(tokenizer.encode(self.dataset[index]['instruction']), dtype=torch.int64)
        example = self.dataset[index]['instruction'] + self.dataset[index]["output"]
        example = tokenizer.encode(example)
        example.append(tokenizer.eos_token_id)
        padding = self.max_seq_length - len(example)
        example = torch.tensor(example, dtype=torch.int64)

        if padding < 0:
            example = example[:self.max_seq_length]
        else:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64)))
            
        labels = copy.deepcopy(example)
        labels[: len(instruction)] = -100
        
        return {"input_ids": example, "labels": labels}
    
dataset = NSText2SQLDataset(size=1000, max_seq_length=1024)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
)


model.train()

model = prepare_model_for_int8_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05
    ,target_modules = ["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

output_dir = '/home/lionel/projects/chat_llama'

config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 1,
    'gradient_accumulation_steps': 2,
    'gradient_checkpointing': False,
}


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    bf16=True,
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=5,
    optim="adamw_torch_fused",
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator
)

# Start training
trainer.train()

model.save_pretrained(output_dir)

model.eval()

text = """CREATE TABLE stadium (
    stadium_id number,
    location text,
    name text,
    capacity number,
)

-- Using valid SQLite, answer the following questions for the tables provided above.

-- how many stadiums in total?

SELECT"""

model_input = tokenizer(text, return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_input, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
