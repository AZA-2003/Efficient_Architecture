from transformers import AutoProcessor, AutoModelForImageTextToText, \
    AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
'''
This ONLY works in REMOTE SERVERS (datahub, colab, etc)

'''

'''

'''
class Qwen3():
    def __init__(self):
        self.processor = None
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

''' 
 
'''     
class Qwen35():
    def __init__(self): 
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
        self.model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B-Base") 
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
'''
'''
class LFM2():
    def __init__(self):
        self.processor = None
        self.model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-700M")
        self.tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-700M")


'''

'''
class IBM_Granite():
    def __init__(self):
        self.processor = None
        self.model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.0-micro")
        self.tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-micro")
'''

'''
class RWKV7():
    def __init__(self):
        self.processor = None
        self.model = AutoModelForCausalLM.from_pretrained('fla-hub/rwkv7-0.4B-world', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained('fla-hub/rwkv7-0.4B-world', trust_remote_code=True)
        