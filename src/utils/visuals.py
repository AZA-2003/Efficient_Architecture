import json
import matplotlib.pyplot as plt
from typing import List


test_suite = [(64,16),(128,32),(128,64),(256,64),(512,64),(1024,64),(2048,64)]
test_suite_X = [f"({r},{g})" for r,g in test_suite]
def generate_plots(json_files:List[str],
                  plot_name: str):

    #plt.figure(figsize=(20,20)) 
    fig, axs = plt.subplots(2,2,figsize=(12,12))
    plt.subplots_adjust(wspace=0.8, hspace=0.8)
    
    #   plt.tight_layout(h_pad=2)
    for json_file in json_files:
        name = json_file.split("_")[0]
        with open(json_file,"r") as f:
            metrics = json.load(f)
        #print(metrics)
        pm = [metrics[f"({r},{g})"]["Peak Mem."] for r,g in test_suite if (f"({r},{g})" in metrics.keys() and len(metrics[f"({r},{g})"]) > 0)]
        ttft = [metrics[f"({r},{g})"]["TTFT"] for r,g in test_suite if (f"({r},{g})" in metrics.keys() and len(metrics[f"({r},{g})"]) > 0)]
        tps = [metrics[f"({r},{g})"]["TPS"] for r,g in test_suite if (f"({r},{g})" in metrics.keys() and len(metrics[f"({r},{g})"]) > 0)]
        ppl = [metrics[f"({r},{g})"]["PPL"] for r,g in test_suite if (f"({r},{g})" in metrics.keys() and len(metrics[f"({r},{g})"]) > 0)]
        
        axs[0][0].plot(pm,'-o',label=name)
        axs[0][0].set_xlabel("Read/Gen context length")
        axs[0][0].set_ylabel("Peak Memory(GB)")
        axs[0][0].set_xticklabels(test_suite_X, rotation=45, fontsize=10)
        
        axs[0][1].plot(ttft,'-o',label=name)
        axs[0][1].set_xlabel("Read/Gen context length")
        axs[0][1].set_ylabel("Time to First Token (sec.)")
        axs[0][1].set_xticklabels(test_suite_X, rotation=45, fontsize=10)
        
        axs[1][0].plot(tps,'-o',label=name)
        axs[1][0].set_xlabel("Read/Gen context length")
        axs[1][0].set_ylabel("Tokens/sec")
        axs[1][0].set_xticklabels(test_suite_X, rotation=45, fontsize=10)
        
        axs[1][1].plot(ppl,'-o',label=name)
        axs[1][1].set_xlabel("Read/Gen context length")
        axs[1][1].set_ylabel("Perplexity")
        axs[1][1].set_xticklabels(test_suite_X, rotation=45, fontsize=10)
    
    axs[0][0].legend(title="Models", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[0][1].legend(title="Models", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[1][0].legend(title="Models", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    axs[1][1].legend(title="Models", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig.savefig(f'{plot_name}.png')
  
  


generate_plots(["Qwen3_metrics.json","Qwen3.5_metrics.json","IBM-G1B_metrics.json","IBM-G350M_metrics.json","LFM2_metrics.json"],"plots")