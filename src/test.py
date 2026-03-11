import transformers
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(".")
from Efficient_Architecture.src.utils.metrics import *


test_suite = [(64,16),(128,32),(128,64),(256,128),(512,128),(1024,256),(2048,256)]

def test_model(model: transformers.models,
               data):
    metrics = {}
    print(calculate_memory_footprint(model))
    for read_len,gen_len in test_suite:
        metrics[(read_len,gen_len)] = {}
        data = data.process(20,read_len,1,)
        pm,ttft,tps,ppl = get_metrics(model, data, read_len, gen_len)
        
        metrics[(read_len,gen_len)]["Peak Mem."] = pm
        metrics[(read_len,gen_len)]["TTFT"] = ttft
        metrics[(read_len,gen_len)]["TPS"] = tps
        metrics[(read_len,gen_len)]["PPL"] = ppl
        
        print(f"{read_len}/{gen_len}")
        print(f"Peak Memory: {pm}")
        print(f"Time to First Token: {ttft}")
        print(f"Tokens per second: {tps}")
        print(f"Perplexity: {ppl}")
    return metrics