from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer,DataCollatorWithPadding
from datasets import load_dataset,Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path



class c4_dataset():
    def __init__(self,
                split: str,
                language: str,
                tokenizer: AutoTokenizer,):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.c4 = load_dataset("allenai/c4",language,split=split,streaming=True) 
        
    def process(self,
                num_examples: int,
                sequence_length: int,
                batch_size: int,):
        
        def tokenizer_func(example,
                            tokenizer:AutoTokenizer = self.tokenizer,
                            sequence_length = sequence_length):
            
            return tokenizer(example['text'],truncation=True,max_length=sequence_length)
    
        c4 = Dataset.from_list(list(self.c4.take(num_examples)))
        c4 = c4.remove_columns(["timestamp","url"])
        c4 = c4.map(tokenizer_func,batched=(batch_size > 1))
        c4 = c4.remove_columns("text")
    
        return DataLoader(c4,shuffle=True,batch_size=batch_size,collate_fn=self.data_collator)

def process_stack_dataset(split: str,
                       language: str,
                       num_examples: int,
                       sequence_length: int,
                       batch_size: int,
                       tokenizer: AutoTokenizer,):
    pass