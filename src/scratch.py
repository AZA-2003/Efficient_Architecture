import torch
import sys
sys.path.append(".")
from src.utils.models import *

try:
    model = Qwen3()
    #model = Qwen35()
    # model = LFM2()
    # model = IBM_Granite()
    # model = RWKV7()
    print("Done!")
except Exception as e:
    import traceback
    traceback.print_exception(e)
    print("Error!")