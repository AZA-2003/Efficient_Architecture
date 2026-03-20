import json
import matplotlib.pyplot as plt
from typing import List

from test_vectors import benchmark_test_suite, profiling_test_suite, prefill_test_suite, decode_test_suite


def generate_plots(
    json_files: List[str],
    plot_name: str,
    suite: List[tuple[int, int]] | None = None,
):
    # Memory uses the combined suite. Latencies use their intended subsets.
    combined_suite = suite if suite is not None else benchmark_test_suite
    combined_X = [f"({r},{g})" for (r, g) in combined_suite]

    prefill_X = [f"({r},{g})" for (r, g) in prefill_test_suite]
    decode_X = [f"({r},{g})" for (r, g) in decode_test_suite]

    x_combined = list(range(len(combined_suite)))
    x_prefill = list(range(len(prefill_test_suite)))
    x_decode = list(range(len(decode_test_suite)))

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.8, hspace=0.8)

    # Set ticks/labels once per axis
    axs[0][0].set_xlabel("Read/Gen context length")
    axs[0][0].set_ylabel("Peak Memory(GB)")
    axs[0][0].set_xticks(x_combined)
    axs[0][0].set_xticklabels(combined_X, rotation=45, fontsize=10)

    axs[0][1].set_xlabel("Read/Gen context length")
    axs[0][1].set_ylabel("Time to First Token (sec.)")
    axs[0][1].set_xticks(x_prefill)
    axs[0][1].set_xticklabels(prefill_X, rotation=45, fontsize=10)

    axs[1][0].set_xlabel("Read/Gen context length")
    axs[1][0].set_ylabel("Tokens/sec")
    axs[1][0].set_xticks(x_decode)
    axs[1][0].set_xticklabels(decode_X, rotation=45, fontsize=10)

    axs[1][1].set_xlabel("Read/Gen context length")
    axs[1][1].set_ylabel("Perplexity")
    axs[1][1].set_xticks(x_prefill)
    axs[1][1].set_xticklabels(prefill_X, rotation=45, fontsize=10)

    for json_file in json_files:
        name = json_file.split("_")[0]
        with open(json_file, "r") as f:
            metrics = json.load(f)

        # Full peak memory (combined)
        pm = []
        for (r, g) in combined_suite:
            key = f"({r},{g})"
            if key in metrics and isinstance(metrics[key], dict) and len(metrics[key]) > 0:
                pm.append(metrics[key].get("Peak Mem.", float("nan")))
            else:
                pm.append(float("nan"))

        # Prefill-only metrics
        ttft = []
        ppl = []
        for (r, g) in prefill_test_suite:
            key = f"({r},{g})"
            if key in metrics and isinstance(metrics[key], dict) and len(metrics[key]) > 0:
                ttft.append(metrics[key].get("TTFT", float("nan")))
                ppl.append(metrics[key].get("PPL", float("nan")))
            else:
                ttft.append(float("nan"))
                ppl.append(float("nan"))

        # Decode-only metric
        tps = []
        for (r, g) in decode_test_suite:
            key = f"({r},{g})"
            if key in metrics and isinstance(metrics[key], dict) and len(metrics[key]) > 0:
                tps.append(metrics[key].get("TPS", float("nan")))
            else:
                tps.append(float("nan"))

        axs[0][0].plot(x_combined, pm, "-o", label=name)
        axs[0][1].plot(x_prefill, ttft, "-o", label=name)
        axs[1][0].plot(x_decode, tps, "-o", label=name)
        axs[1][1].plot(x_prefill, ppl, "-o", label=name)

    axs[0][0].legend(title="Models", loc="best")
    axs[0][1].legend(title="Models", loc="best")
    axs[1][0].legend(title="Models", loc="best")
    axs[1][1].legend(title="Models", loc="best")

    fig.tight_layout()
    fig.savefig(f"{plot_name}.png")


def generate_incremental_memory_plots(
    json_files: List[str],
    plot_name: str,
    baseline_json_path,
    suite: List[tuple[int, int]] | None = None,
):
    """
    Like `generate_plots`, but for the first subplot plots:
      Peak incremental memory DM(N) = PeakMem(N,G) - M(N0,G)

    Expects:
    - per-model metrics JSON schema from `main.ipynb` with keys like "Peak Mem."
    - baseline JSON produced by `main.ipynb` (see `baseline_peak_mem_gb_by_model`)
    """
    combined_suite = suite if suite is not None else benchmark_test_suite
    combined_X = [f"({r},{g})" for (r, g) in combined_suite]

    prefill_X = [f"({r},{g})" for (r, g) in prefill_test_suite]
    decode_X = [f"({r},{g})" for (r, g) in decode_test_suite]

    x_combined = list(range(len(combined_suite)))
    x_prefill = list(range(len(prefill_test_suite)))
    x_decode = list(range(len(decode_test_suite)))

    from pathlib import Path

    with open(Path(baseline_json_path), "r") as f:
        baseline_obj = json.load(f)

    baseline_peak = baseline_obj.get("baseline_peak_mem_gb_by_model")
    if baseline_peak is None:
        baseline_peak = baseline_obj

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.8, hspace=0.8)

    axs[0][0].set_xlabel("Read/Gen context length")
    axs[0][0].set_ylabel("Peak Incremental Memory (GB)")
    axs[0][0].set_xticks(x_combined)
    axs[0][0].set_xticklabels(combined_X, rotation=45, fontsize=10)

    axs[0][1].set_xlabel("Read/Gen context length")
    axs[0][1].set_ylabel("Time to First Token (sec.)")
    axs[0][1].set_xticks(x_prefill)
    axs[0][1].set_xticklabels(prefill_X, rotation=45, fontsize=10)

    axs[1][0].set_xlabel("Read/Gen context length")
    axs[1][0].set_ylabel("Tokens/sec")
    axs[1][0].set_xticks(x_decode)
    axs[1][0].set_xticklabels(decode_X, rotation=45, fontsize=10)

    axs[1][1].set_xlabel("Read/Gen context length")
    axs[1][1].set_ylabel("Perplexity")
    axs[1][1].set_xticks(x_prefill)
    axs[1][1].set_xticklabels(prefill_X, rotation=45, fontsize=10)

    for json_file in json_files:
        json_file = str(json_file)
        model_name = Path(json_file).stem
        if model_name.endswith("_metrics"):
            model_name = model_name[: -len("_metrics")]

        with open(json_file, "r") as f:
            metrics = json.load(f)

        base_scalar = None
        if isinstance(baseline_peak, dict):
            base_scalar = baseline_peak.get(model_name)

        # Peak incremental memory over the combined suite
        inc_mem_gb = []
        for read_len, gen_len in combined_suite:
            key = f"({read_len},{gen_len})"
            if (
                key in metrics
                and isinstance(metrics[key], dict)
                and len(metrics[key]) > 0
                and base_scalar is not None
            ):
                inc_mem_gb.append(metrics[key]["Peak Mem."] - base_scalar)
            else:
                inc_mem_gb.append(float("nan"))

        # Prefill-only: TTFT + PPL
        ttft = []
        ppl = []
        for read_len, gen_len in prefill_test_suite:
            key = f"({read_len},{gen_len})"
            if key in metrics and isinstance(metrics[key], dict) and len(metrics[key]) > 0:
                ttft.append(metrics[key].get("TTFT", float("nan")))
                ppl.append(metrics[key].get("PPL", float("nan")))
            else:
                ttft.append(float("nan"))
                ppl.append(float("nan"))

        # Decode-only: TPS
        tps = []
        for read_len, gen_len in decode_test_suite:
            key = f"({read_len},{gen_len})"
            if key in metrics and isinstance(metrics[key], dict) and len(metrics[key]) > 0:
                tps.append(metrics[key].get("TPS", float("nan")))
            else:
                tps.append(float("nan"))

        axs[0][0].plot(x_combined, inc_mem_gb, "-o", label=model_name)
        axs[0][1].plot(x_prefill, ttft, "-o", label=model_name)
        axs[1][0].plot(x_decode, tps, "-o", label=model_name)
        axs[1][1].plot(x_prefill, ppl, "-o", label=model_name)

    axs[0][0].legend(title="Models", loc="best")
    axs[0][1].legend(title="Models", loc="best")
    axs[1][0].legend(title="Models", loc="best")
    axs[1][1].legend(title="Models", loc="best")

    fig.tight_layout()
    fig.savefig(f"{plot_name}.png")


def generate_profiling_plots(
    json_files: List[str],
    plot_name: str,
    suite: List[tuple[int, int]] | None = None,
):
    """
    Plot profiling JSONs produced by `profiling_main.ipynb`.

    Each JSON is expected to have:
      {
        "_model": "...",
        "_points": {
          "(read_len,gen_len)": {
            "prefill_s": ...,
            "decode_s": ...,
            "prefill_peak_gpu_gb": ...,
            "decode_peak_gpu_gb": ...,
            "prefill_cpu_peak_mb": ...,
            "decode_cpu_peak_mb": ...
          }
        }
      }
    """
    test_suite = suite if suite is not None else profiling_test_suite
    test_suite_X = [f"({r},{g})" for r, g in test_suite]
    x = list(range(len(test_suite_X)))

    metrics_to_plot = [
        ("prefill_s", "Prefill Latency (s)", "prefill_latency.png"),
        ("decode_s", "Decode Latency (s)", "decode_latency.png"),
        ("prefill_peak_gpu_gb", "Prefill Peak GPU Memory (GB)", "prefill_gpu_peak.png"),
        ("decode_peak_gpu_gb", "Decode Peak GPU Memory (GB)", "decode_gpu_peak.png"),
    ]

    for key, ylabel, filename in metrics_to_plot:
        plt.figure(figsize=(10, 5))
        for json_file in json_files:
            with open(json_file, "r") as f:
                obj = json.load(f)
            model_name = obj.get("_model", json_file.split("/")[-1].replace("_profiling.json", ""))
            pts = obj.get("_points", {})

            ys = []
            for x_label in test_suite_X:
                item = pts.get(x_label, {})
                if isinstance(item, dict) and key in item:
                    ys.append(item[key])
                else:
                    ys.append(float("nan"))

            plt.plot(x, ys, marker="o", label=model_name)

        plt.xticks(x, test_suite_X, rotation=45)
        plt.xlabel("(read_len, gen_len)")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = f"{plot_name}_{filename}"
        plt.savefig(out_path)
        plt.show()

    # Combined overview (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    combined = [
        ("prefill_s", "Prefill Latency (s)", axs[0, 0]),
        ("decode_s", "Decode Latency (s)", axs[0, 1]),
        ("prefill_peak_gpu_gb", "Prefill Peak GPU Mem (GB)", axs[1, 0]),
        ("decode_peak_gpu_gb", "Decode Peak GPU Mem (GB)", axs[1, 1]),
    ]

    for key, title, ax in combined:
        for json_file in json_files:
            with open(json_file, "r") as f:
                obj = json.load(f)
            model_name = obj.get("_model", json_file.split("/")[-1].replace("_profiling.json", ""))
            pts = obj.get("_points", {})

            ys = []
            for x_label in test_suite_X:
                item = pts.get(x_label, {})
                if isinstance(item, dict) and key in item:
                    ys.append(item[key])
                else:
                    ys.append(float("nan"))

            ax.plot(x, ys, marker="o", label=model_name)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(test_suite_X, rotation=45)
        ax.grid(alpha=0.3)

    axs[0, 0].legend(loc="best")
    plt.tight_layout()
    overview_out = f"{plot_name}_profiling_overview.png"
    plt.savefig(overview_out)
    plt.show()


if __name__ == "__main__":
    generate_plots(
        [
            "Qwen3_metrics.json",
            "Qwen3.5_metrics.json",
            "IBM-G1B_metrics.json",
            "IBM-G350M_metrics.json",
            "LFM2_metrics.json",
        ],
        "plots",
    )