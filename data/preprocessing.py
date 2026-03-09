from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer,DataCollatorWithPadding
from datasets import load_dataset,Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path



def process_c4_dataset(split: str,
                       language: str,
                       num_examples: int,
                       sequence_length: int,
                       batch_size: int,
                       tokenizer: AutoTokenizer,):
    
    def tokenizer_func(example,
                tokenizer:AutoTokenizer = tokenizer,
                sequence_length = sequence_length):
        return tokenizer(example['text'][:sequence_length],truncation=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    c4 = load_dataset("allenai/c4",language,split=split,streaming=True) 
    c4 = Dataset.from_list(list(c4.take(num_examples)))
    c4 = c4.remove_columns(["timestamp","url"])
    c4 = c4.map(tokenizer_func,batched=(batch_size > 1))
    c4 = c4.remove_columns("text")
    
    return DataLoader(c4,shuffle=True,batch_size=batch_size,collate_fn=data_collator)

def process_stack_dataset(split: str,
                       language: str,
                       num_examples: int,
                       sequence_length: int,
                       batch_size: int,
                       tokenizer: AutoTokenizer,):
    pass